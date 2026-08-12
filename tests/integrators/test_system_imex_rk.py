"""Tests for systems of equations coupled through their nonlinear terms.

The fast structural tests run by default; the ones that integrate a complete
problem are marked `integration`, following `test_imex_rk.py`.

Note that the suite runs in float32 unless `--float64` is given, so tolerances
here are chosen well above single-precision round-off.
"""

from typing import cast

import jax.numpy as jnp
import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProduct
from jaxfun.galerkin.arguments import JAXFunction, TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.integrators import (
    ARS222,
    ARS443,
    IMEXRungeKutta,
    SystemIMEXRungeKutta,
)
from jaxfun.integrators.nonlinear import compile_nonlinear_evaluator
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms

# Schnakenberg parameters. The uniform state u = a + b, v = b/(a+b)^2 is an
# exact equilibrium, which the values below make 1.0 and 0.9 -- deliberately
# distinct, so that swapping the two fields is detectable.
A_PARAM = 0.1
B_PARAM = 0.9
GAMMA = 3.0
U_STAR = A_PARAM + B_PARAM
V_STAR = B_PARAM / (A_PARAM + B_PARAM) ** 2


def schnakenberg(N: int = 12, G: float = 30.0, eu: float = 1.0, ev: float = 10.0):
    """Return `(V, eq_u, eq_v)` for the Schnakenberg system on a periodic square."""
    F = Fourier(N, Domain(0, G))
    V = TensorProduct(F, F, name="Vs")
    w = TestFunction(V, name="w")
    u = TrialFunction(V, name="u", transient=True)
    q = TestFunction(V, name="q")
    v = TrialFunction(V, name="v", transient=True)
    t = V.system.base_time()
    eq_u = (u.diff(t) - eu * Div(Grad(u)) - GAMMA * (A_PARAM - u + u**2 * v)) * w
    eq_v = (v.diff(t) - ev * Div(Grad(v)) - GAMMA * (B_PARAM - u**2 * v)) * q
    return V, eq_u, eq_v


# ---------------------------------------------------------------------------
# Symbolic splitting: foreign fields must never reach the implicit operator
# ---------------------------------------------------------------------------


def test_foreign_field_terms_are_forced_explicit() -> None:
    """`explicit` moves every term touching a foreign field to the nonlinear part.

    Without it, `gamma*u**2*v` looks linear in `v` (exactly one factor contains
    `v`) and would be assembled into `v`'s implicit operator with a coefficient
    that silently freezes `u`.
    """
    V, _, eq_v = schnakenberg()
    _, rhs = split_time_derivative_terms(eq_v, V.system.base_time())

    trials = {u.name: u for u in rhs.atoms(TrialFunction)}
    u, v = trials["u"], trials["v"]

    # Documented behaviour without `explicit`: the coupling term is mistaken for
    # a linear term in v.
    linear, nonlinear = split_linear_nonlinear_terms(rhs, v)
    assert linear.has(u)
    assert sp.sympify(nonlinear) == 0

    linear, nonlinear = split_linear_nonlinear_terms(rhs, v, explicit=(u,))
    assert not linear.has(u)
    assert nonlinear.has(u) and nonlinear.has(v)


def test_term_containing_only_a_foreign_field_is_explicit() -> None:
    """A purely linear coupling term must not be assembled either.

    `a*u` in `v`'s equation does not contain `v` at all, so it would otherwise
    pass straight through as a "linear" term and be assembled as a bilinear form
    in the wrong field's trial function.
    """
    V = FunctionSpace(8, Fourier, name="Vc", fun_str="Ec")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="u")
    v = TrialFunction(V, name="v")
    expr = q * (Div(Grad(v)) + 5 * u)

    linear, nonlinear = split_linear_nonlinear_terms(expr, v)
    assert linear.has(u)  # today's behaviour without `explicit`

    linear, nonlinear = split_linear_nonlinear_terms(expr, v, explicit=(u,))
    assert not linear.has(u)
    assert sp.simplify(nonlinear - 5 * q * u) == 0


def test_system_splits_each_equation_about_its_own_field() -> None:
    """Each sub-integrator is implicit in its own field and explicit in the rest."""
    _, eq_u, eq_v = schnakenberg()
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Float(U_STAR), sp.Float(V_STAR)),
        sparse=True,
    )
    for sub in integrator.integrators:
        # No trial function of another equation survives into the operator, and
        # the nonlinear part has been fully replaced by JAXFunction nodes.
        assert not sub.nonlinear_expr.atoms(TrialFunction)
        assert sub.has_nonlinear
        names = {str(node.func) for node in sub.nonlinear_expr.atoms(sp.Function)}
        assert {"u_jax", "v_jax"} <= names

    for sub, own in zip(integrator.integrators, ("u", "v"), strict=True):
        # Implicit in its own field only, and the diffusion stayed implicit.
        assert {u.name for u in sub.linear_expr.atoms(TrialFunction)} == {own}
        assert sub.linear_expr.has(Div)


# ---------------------------------------------------------------------------
# Nonlinear evaluator with several fields
# ---------------------------------------------------------------------------


def _field_node(V, name: str) -> AppliedUndef:
    """Return the symbolic JAXFunction node a nonlinear evaluator reads from."""
    return cast(AppliedUndef, JAXFunction(jnp.zeros(V.num_dofs), V, name=name).doit())


def _two_field_nodes(V):
    return _field_node(V, "u_jax"), _field_node(V, "v_jax")


def test_multi_jaxfunction_evaluator_products_and_derivatives() -> None:
    """Each JAXFunction leaf is evaluated through its own coefficients.

    Guards the rebuilt-leaf hazard: `doit()`/`expand` may rebuild a leaf node,
    and a rebuilt leaf that is not mapped back to the object the evaluator
    mutates would silently read the stale class-level array (all zeros).
    """
    V = FunctionSpace(16, Fourier, name="Vm", fun_str="Em")
    (x,) = V.system.base_scalars()
    u_node, v_node = _two_field_nodes(V)

    xj = V.mesh()
    uh = V.forward(jnp.sin(xj))
    vh = V.forward(jnp.cos(2 * xj))
    bu, bv = V.backward(uh), V.backward(vh)

    product = compile_nonlinear_evaluator(3 * u_node**2 * v_node, V, (u_node, v_node))
    assert jnp.allclose(product((uh, vh), None), 3 * bu**2 * bv, atol=1e-5)
    assert float(jnp.abs(product((uh, vh), None)).max()) > 0.1  # not stale zeros

    derivative = compile_nonlinear_evaluator(
        u_node * sp.Derivative(v_node, x), V, (u_node, v_node)
    )
    expected = bu * V.backward_primitive(vh, k=1)
    assert jnp.allclose(derivative((uh, vh), None), expected, atol=1e-4)

    # Field order matters: swapping the state tuple swaps the roles.
    assert jnp.allclose(product((vh, uh), None), 3 * bv**2 * bu, atol=1e-5)


def test_single_jaxfunction_evaluator_still_takes_a_bare_array() -> None:
    """The scalar signature is unchanged, as `inner` depends on it."""
    V = FunctionSpace(16, Fourier, name="Vs1", fun_str="Es1")
    u_node = _field_node(V, "w_jax")
    uh = V.forward(jnp.sin(V.mesh()))
    evaluate = compile_nonlinear_evaluator(u_node**3, V, u_node)
    assert jnp.allclose(evaluate(uh, None), V.backward(uh) ** 3, atol=1e-5)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_rejects_duplicate_field_names() -> None:
    """Same-named fields are indistinguishable to SymPy and must be refused.

    Applied undefined functions compare and hash by name, so two same-named
    nodes would collide in the evaluator's memo cache while holding independent
    coefficient arrays -- one field would evaluate to the other's values.
    """
    F = Fourier(8, Domain(0, 30.0))
    V = TensorProduct(F, F, name="Vdup")
    w = TestFunction(V, name="w")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="same", transient=True)
    v = TrialFunction(V, name="same", transient=True)
    t = V.system.base_time()
    with pytest.raises(ValueError, match="unique names"):
        SystemIMEXRungeKutta(
            (
                w * (u.diff(t) - Div(Grad(u)) - u**2 * v),
                q * (v.diff(t) - Div(Grad(v)) + u**2 * v),
            ),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=(sp.Float(1), sp.Float(1)),
        )


def test_rejects_different_quadrature_families() -> None:
    """Fields sampled at different physical points cannot be multiplied."""
    Vc = FunctionSpace(12, Chebyshev, name="Vcheb", fun_str="Tc")
    Vl = FunctionSpace(12, Legendre, name="Vleg", fun_str="Ll")
    w = TestFunction(Vc, name="w")
    q = TestFunction(Vl, name="q")
    u = TrialFunction(Vc, name="u", transient=True)
    v = TrialFunction(Vl, name="v", transient=True)
    t = Vc.system.base_time()
    with pytest.raises(NotImplementedError, match="quadrature mesh"):
        SystemIMEXRungeKutta(
            (
                w * (u.diff(t) - Div(Grad(u)) - u * v),
                q * (v.diff(t) - Div(Grad(v)) + u * v),
            ),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=(sp.Float(1), sp.Float(1)),
        )


def test_rejects_mismatched_number_of_initial_conditions() -> None:
    _, eq_u, eq_v = schnakenberg(N=8)
    with pytest.raises(ValueError, match="initial"):
        SystemIMEXRungeKutta(
            (eq_u, eq_v),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=(sp.Float(U_STAR),),
        )


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------


def _independent_pair(N: int = 24):
    """Two equations sharing no fields, so the system must decouple exactly."""
    V = FunctionSpace(N, Fourier, name="Vi", fun_str="Ei")
    w = TestFunction(V, name="w")
    u = TrialFunction(V, name="u", transient=True)
    q = TestFunction(V, name="q")
    v = TrialFunction(V, name="v", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    eq_u = w * (u.diff(t) - 0.5 * Div(Grad(u)) - u + u**3)
    eq_v = q * (v.diff(t) - 2.0 * Div(Grad(v)) - v + v**3)
    return V, eq_u, eq_v, sp.sin(x), sp.cos(2 * x)


@pytest.mark.integration
def test_uncoupled_system_matches_two_scalar_integrators() -> None:
    """With no shared fields the system must reproduce two separate runs."""
    T, steps = 0.2, 40
    _, eq_u, eq_v, u0, v0 = _independent_pair()
    dt = T / steps

    system = SystemIMEXRungeKutta(
        (eq_u, eq_v), tableau=ARS443, time=(0.0, T), initial=(u0, v0), sparse=True
    )
    u_sys, v_sys = system.solve(dt=dt, steps=steps, progress=False)

    u_ref = IMEXRungeKutta(
        eq_u, tableau=ARS443, time=(0.0, T), initial=u0, sparse=True
    ).solve(dt=dt, steps=steps, progress=False)
    v_ref = IMEXRungeKutta(
        eq_v, tableau=ARS443, time=(0.0, T), initial=v0, sparse=True
    ).solve(dt=dt, steps=steps, progress=False)

    # Identical operations in identical order, so this holds bit for bit.
    assert jnp.array_equal(u_sys, u_ref)
    assert jnp.array_equal(v_sys, v_ref)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("tableau", "order", "coarse_steps"),
    [(ARS222, 2, 40), (ARS443, 3, 20)],
    ids=["ars222", "ars443"],
)
def test_linearly_coupled_rotation_matches_analytic(
    tableau, order, coarse_steps
) -> None:
    """u_t = nu*u_xx + a*v, v_t = nu*v_xx - a*u has a closed-form solution.

    Each coupling term contains only the *foreign* field, so this is the sharpest
    check that such terms are lagged explicitly rather than assembled into the
    wrong field's operator.
    """
    N, T, nu, a = 32, 1.0, 0.05, 3.0
    V = FunctionSpace(N, Fourier, name="Vrot", fun_str="Er")
    w = TestFunction(V, name="w")
    u = TrialFunction(V, name="u", transient=True)
    q = TestFunction(V, name="q")
    v = TrialFunction(V, name="v", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    eq_u = w * (u.diff(t) - nu * Div(Grad(u)) - a * v)
    eq_v = q * (v.diff(t) - nu * Div(Grad(v)) + a * u)

    xj = V.mesh()
    u_exact = jnp.exp(-nu * T) * jnp.cos(a * T) * jnp.sin(xj)
    v_exact = -jnp.exp(-nu * T) * jnp.sin(a * T) * jnp.sin(xj)

    def error(steps: int) -> float:
        # A fresh integrator per dt: `setup(dt)` bakes the stage operators into
        # the traced step, so reusing one instance across step sizes would keep
        # the operators of the first dt.
        integrator = SystemIMEXRungeKutta(
            (eq_u, eq_v),
            tableau=tableau,
            time=(0.0, T),
            initial=(sp.sin(x), sp.Integer(0)),
            sparse=True,
        )
        u_hat, v_hat = integrator.solve(dt=T / steps, steps=steps, progress=False)
        return max(
            float(
                jnp.linalg.norm(V.backward(u_hat).real - u_exact)
                / jnp.linalg.norm(u_exact)
            ),
            float(
                jnp.linalg.norm(V.backward(v_hat).real - v_exact)
                / jnp.linalg.norm(v_exact)
            ),
        )

    coarse = error(coarse_steps)
    fine = error(2 * coarse_steps)
    assert fine < coarse
    assert fine > 1e-5, "refine less; the error has reached float32 round-off"
    observed = jnp.log2(coarse / fine)
    assert float(observed) > order - 0.4


@pytest.mark.integration
def test_schnakenberg_uniform_steady_state_is_preserved() -> None:
    """The uniform equilibrium is exact, so both fields must stay put.

    Catches sign errors, a swapped field order, and any stage bookkeeping that
    lets an equation see a stale partner: all of them break the cancellation
    between the constant source and the reaction term.
    """
    V, eq_u, eq_v = schnakenberg(N=12)
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Float(U_STAR), sp.Float(V_STAR)),
        sparse=True,
    )
    u_hat, v_hat = integrator.solve(dt=0.01, steps=100, progress=False)
    assert jnp.allclose(V.backward(u_hat).real, U_STAR, atol=1e-5)
    assert jnp.allclose(V.backward(v_hat).real, V_STAR, atol=1e-5)


@pytest.mark.integration
def test_constant_source_keeps_its_sign() -> None:
    """u_t = u_xx + 1 from rest must grow to u(T) = T, not -T.

    Coupled reaction systems live on constant sources (Schnakenberg's `gamma*a`
    and `gamma*b`), so the assembled forcing has to enter the right way round.
    """
    T, steps = 0.5, 50
    V = FunctionSpace(16, Fourier, name="Vf", fun_str="Ef")
    w = TestFunction(V, name="w")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    integrator = IMEXRungeKutta(
        w * (u.diff(t) - Div(Grad(u)) - 1),
        tableau=ARS443,
        time=(0.0, T),
        initial=sp.Integer(0),
        sparse=True,
    )
    u_hat = integrator.solve(dt=T / steps, steps=steps, progress=False)
    assert jnp.allclose(V.backward(u_hat).real, T, atol=1e-5)


@pytest.mark.integration
def test_snapshots_and_restart_preserve_the_tuple_state() -> None:
    """Snapshots stack per field, and a tuple restart continues the same run."""
    T, steps = 0.2, 40
    _, eq_u, eq_v, u0, v0 = _independent_pair(N=16)
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v), tableau=ARS443, time=(0.0, T), initial=(u0, v0), sparse=True
    )
    u_states, v_states = integrator.solve(
        dt=T / steps,
        steps=steps,
        n_batches=4,
        return_batch_snapshots=True,
        progress=False,
    )
    assert u_states.shape[0] == 5
    assert v_states.shape[0] == 5

    halves = integrator.solve(dt=T / steps, steps=steps // 2, progress=False)
    restarted = integrator.solve(
        dt=T / steps, steps=steps // 2, state0=halves, progress=False
    )
    direct = integrator.solve(dt=T / steps, steps=steps, progress=False)
    for got, want in zip(restarted, direct, strict=True):
        assert jnp.allclose(got, want, atol=1e-5)


@pytest.mark.integration
def test_rejects_a_non_tuple_restart_state() -> None:
    _, eq_u, eq_v, u0, v0 = _independent_pair(N=16)
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v), tableau=ARS443, time=(0.0, 0.1), initial=(u0, v0), sparse=True
    )
    with pytest.raises(TypeError, match="tuple"):
        # Deliberately the wrong shape of state: a system restarts from a tuple.
        integrator.solve(dt=0.01, steps=1, state0=jnp.zeros(16), progress=False)  # ty: ignore[invalid-argument-type]


@pytest.mark.integration
def test_fields_may_use_different_function_spaces() -> None:
    """Different bases/BCs are fine as long as the quadrature mesh is shared.

    `u` carries Dirichlet conditions and `v` Neumann ones, so the two spaces have
    different basis functions and different numbers of degrees of freedom, but
    both reduce to the same Chebyshev quadrature points.
    """
    Vd = FunctionSpace(
        24,
        Chebyshev,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name="Vd",
        fun_str="psi",
    )
    Vn = FunctionSpace(
        24,
        Chebyshev,
        bcs={"left": {"N": 0}, "right": {"N": 0}},
        name="Vn",
        fun_str="phi",
    )
    w = TestFunction(Vd, name="w")
    u = TrialFunction(Vd, name="u", transient=True)
    q = TestFunction(Vn, name="q")
    v = TrialFunction(Vn, name="v", transient=True)
    (x,) = Vd.system.base_scalars()
    t = Vd.system.base_time()

    integrator = SystemIMEXRungeKutta(
        (
            w * (u.diff(t) - 0.1 * Div(Grad(u)) - u + u**2 * v),
            q * (v.diff(t) - 0.05 * Div(Grad(v)) - v - u**2 * v),
        ),
        tableau=ARS443,
        time=(0.0, 0.5),
        initial=(sp.sin(sp.pi * (x + 1) / 2), sp.cos(sp.pi * (x + 1) / 2)),
        sparse=True,
    )
    u_hat, v_hat = integrator.solve(dt=0.005, steps=100, progress=False)
    assert u_hat.shape == (Vd.num_dofs,)
    assert v_hat.shape == (Vn.num_dofs,)
    assert bool(jnp.isfinite(u_hat).all())
    assert bool(jnp.isfinite(v_hat).all())


# ---------------------------------------------------------------------------
# Shared evaluation across the equations of a system
# ---------------------------------------------------------------------------


def _per_equation_scalar_products(integrator, states, N=None):
    """The unshared reference path: every equation evaluates on its own."""
    return tuple(
        sub.nonlinear_rhs_scalar_product(states, N) for sub in integrator.integrators
    )


def test_proportional_reaction_terms_share_one_projection() -> None:
    """Schnakenberg's `+gamma*u^2 v` and `-gamma*u^2 v` project only once.

    Sharing the backward transforms alone buys nothing -- they are common
    subexpressions that XLA merges anyway. Terms differing by a constant reach
    the transform with *different* values, so merging them is what actually
    removes work.
    """
    _, eq_u, eq_v = schnakenberg()
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Float(U_STAR), sp.Float(V_STAR)),
        sparse=True,
    )
    groups = integrator._nonlinear_groups
    assert len(groups) == 1
    _, members = groups[0]
    assert {k for k, _ in members} == {0, 1}
    assert dict(members)[1] == pytest.approx(-1.0)


def test_grouped_evaluation_matches_the_per_equation_path() -> None:
    """Grouping must not change the answer, for any stoichiometry."""
    F = Fourier(12, Domain(0, 30.0))
    V = TensorProduct(F, F, name="Vstoich")
    w = TestFunction(V, name="w")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="u", transient=True)
    v = TrialFunction(V, name="v", transient=True)
    t = V.system.base_time()
    integrator = SystemIMEXRungeKutta(
        (
            w * (u.diff(t) - Div(Grad(u)) - 2 * u**2 * v),
            q * (v.diff(t) - Div(Grad(v)) + 5 * u**2 * v),
        ),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Float(1.0), sp.Float(0.7)),
        sparse=True,
    )
    # 2 : -5 is still a constant ratio, so one projection serves both.
    assert len(integrator._nonlinear_groups) == 1
    assert dict(integrator._nonlinear_groups[0][1])[1] == pytest.approx(-2.5)

    states = integrator.initial_coefficients()
    integrator.setup(0.01)
    for shared, reference in zip(
        integrator.nonlinear_scalar_products(states, None),
        _per_equation_scalar_products(integrator, states),
        strict=True,
    ):
        assert jnp.allclose(shared, reference, rtol=1e-6, atol=1e-6)


def test_non_proportional_terms_are_not_grouped() -> None:
    """`u**2*v` and `u*v**2` are different functions and each needs its own."""
    F = Fourier(12, Domain(0, 30.0))
    V = TensorProduct(F, F, name="Vnp")
    w = TestFunction(V, name="w")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="u", transient=True)
    v = TrialFunction(V, name="v", transient=True)
    t = V.system.base_time()
    integrator = SystemIMEXRungeKutta(
        (
            w * (u.diff(t) - Div(Grad(u)) - u**2 * v),
            q * (v.diff(t) - Div(Grad(v)) - u * v**2),
        ),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Float(1.0), sp.Float(1.0)),
        sparse=True,
    )
    assert len(integrator._nonlinear_groups) == 2


def test_equations_with_different_test_spaces_are_not_grouped() -> None:
    """Sharing a projection requires sharing the space projected onto.

    Also checks the projection stays real: scaling by a Python `complex` would
    promote a real coefficient array and double its memory traffic.
    """
    Vd = FunctionSpace(
        16,
        Chebyshev,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name="Vdg",
        fun_str="psig",
    )
    Vn = FunctionSpace(
        16,
        Chebyshev,
        bcs={"left": {"N": 0}, "right": {"N": 0}},
        name="Vng",
        fun_str="phig",
    )
    w = TestFunction(Vd, name="w")
    u = TrialFunction(Vd, name="u", transient=True)
    q = TestFunction(Vn, name="q")
    v = TrialFunction(Vn, name="v", transient=True)
    (x,) = Vd.system.base_scalars()
    t = Vd.system.base_time()
    integrator = SystemIMEXRungeKutta(
        (
            w * (u.diff(t) - 0.1 * Div(Grad(u)) - u**2 * v),
            q * (v.diff(t) - 0.05 * Div(Grad(v)) + u**2 * v),
        ),
        tableau=ARS443,
        time=(0.0, 0.5),
        initial=(sp.sin(sp.pi * (x + 1) / 2), sp.cos(sp.pi * (x + 1) / 2)),
        sparse=True,
    )
    # Proportional terms, but the projections target different spaces.
    assert len(integrator._nonlinear_groups) == 2

    states = integrator.initial_coefficients()
    integrator.setup(0.005)
    for shared, reference in zip(
        integrator.nonlinear_scalar_products(states, None),
        _per_equation_scalar_products(integrator, states),
        strict=True,
    ):
        assert shared.dtype == reference.dtype
        assert jnp.allclose(shared, reference, rtol=1e-6, atol=1e-6)


def test_shared_value_cache_does_not_survive_between_calls() -> None:
    """The per-call cache must never serve values from an earlier stage.

    A cache reused across calls would silently evaluate every stage with the
    coefficients of the first one.
    """
    _, eq_u, eq_v = schnakenberg()
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Float(U_STAR), sp.Float(V_STAR)),
        sparse=True,
    )
    integrator.setup(0.01)
    first_states = integrator.initial_coefficients()
    first = integrator.nonlinear_scalar_products(first_states, None)

    second_states = tuple(1.7 * c for c in first_states)
    second = integrator.nonlinear_scalar_products(second_states, None)

    # The state changed, so the answer must change ...
    assert not jnp.allclose(first[0], second[0])
    # ... and must equal a fresh per-equation evaluation of the new state.
    for shared, reference in zip(
        second,
        _per_equation_scalar_products(integrator, second_states),
        strict=True,
    ):
        assert jnp.allclose(shared, reference, rtol=1e-6, atol=1e-6)
