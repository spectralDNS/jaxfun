"""Tests for systems holding an algebraic constraint equation.

A constraint carries no time derivative: its field is solved for at every
Runge-Kutta stage from the fields that are integrated, rather than stepped
alongside them.

Two systems carry the tests, both well posed:

- `signal_system`, on a bounded Legendre square, for everything structural. It
  puts the two fields in spaces of different size and gives the constrained one
  an inhomogeneous wall value, so the cross-space coupling operators and the
  boundary lifting are both exercised.
- `decay_system`, on a Fourier line, where eliminating the constraint leaves a
  scalar ODE with a closed-form solution, for everything to do with time.

Note that the suite runs in float32 unless `--float64` is given, so tolerances
here are chosen well above single-precision round-off.
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProduct
from jaxfun.galerkin.arguments import JAXFunction, TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import project
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.integrators import (
    ARK3_2_4L2SA,
    ARS443,
    IMEX_SSP2_222,
    ConstraintSolver,
    SystemIMEXRungeKutta,
)
from jaxfun.utils import get_time_independent

# Every combination path in `SystemIMEXRungeKutta.step`: globally stiffly
# accurate, implicit-only stiffly accurate, and the general weighted sum. Only
# the first returns the last stage unchanged, so this is what checks that the
# constrained field is re-solved from the *accepted* solution.
ALL_COMBINATION_PATHS = [ARS443, ARK3_2_4L2SA, IMEX_SSP2_222]

A_RATE = 1.0
B_RATE = 1.0


def signal_spaces(N: int = 12, modes: int | None = None):
    """Return `(V, U)`: the constrained signal space and the transported one.

    `v` is a fast-equilibrating signal held at a prescribed value on one wall,
    so its space carries an inhomogeneous Dirichlet condition and therefore a
    boundary lifting. `u` reacts pointwise and has no spatial operator of its
    own, so it needs no boundary conditions and takes the full orthogonal space
    -- which is also what lets it represent `v - lap(v)` exactly. `modes`
    truncates it below that, which the recovery then loses.
    """
    x = sp.Symbol("x")
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    V = TensorProduct(
        FunctionSpace(N, Legendre, bcs=hom, name="Vxs", fun_str="Lvx"),
        FunctionSpace(
            N,
            Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 1 - x**2}},
            name="Vys",
            fun_str="Lvy",
        ),
        name="Vsig",
    )
    if modes is None:
        return V, V.get_orthogonal()
    U = TensorProduct(
        FunctionSpace(modes, Legendre, name="Uxs", fun_str="Lux"),
        FunctionSpace(modes, Legendre, name="Uys", fun_str="Luy"),
        system=V.system,
        name="Usig",
    )
    return V, U


def signal_system(V, U):
    """Return `(eq_u, eq_v)` for `u_t = -a*u - b*v` with `0 = lap(v) - v + u`.

    Both terms of the `u` equation are dissipative and the screened Helmholtz
    operator with Dirichlet data is negative definite, so the pair is well posed
    and can be integrated.
    """
    v = TrialFunction(V, name="v")  # constrained: no transient=True
    q = TestFunction(V, name="q")
    u = TrialFunction(U, name="u", transient=True)
    w = TestFunction(U, name="w")
    t = V.system.base_time()
    eq_u = (u.diff(t) + A_RATE * u + B_RATE * v) * w
    eq_v = (Div(Grad(v)) - v + u) * q
    return eq_u, eq_v


def signal_integrator(N: int = 12, modes: int | None = None, tableau=ARS443, **kw):
    """Build the signal system, returning `(V, U, integrator)`."""
    V, U = signal_spaces(N=N, modes=modes)
    integrator = SystemIMEXRungeKutta(
        signal_system(V, U),
        tableau=tableau,
        time=(0.0, 1.0),
        initial=(sp.S.Zero, None),
        sparse=True,
        **kw,
    )
    return V, U, integrator


def decay_system(N: int = 8, c: float = 2.0):
    """Return `(V, eq_u, eq_p, c, u, p)` for `u_t = -c*p` constrained by `p = u`.

    Eliminating the constraint leaves `u_t = -c*u`, so `u = u0*exp(-c*t)` is
    exact -- and `p` only ever enters `u`'s equation as a foreign field, which
    makes this a direct check that the constraint is solved at the right point
    of the stage loop rather than lagged by one.

    The trial functions come back too, because a weak form can only be rebuilt
    against the very objects it was written with.
    """
    V = Fourier(N, Domain(0, 2 * sp.pi), name="Vd", fun_str="Ed")
    g = TestFunction(V, name="g")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="u", transient=True)
    p = TrialFunction(V, name="p")
    t = V.system.base_time()
    return V, g * (u.diff(t) + c * p), q * (p - u), c, u, p


# ---------------------------------------------------------------------------
# Structure: classification and symbolic splitting
# ---------------------------------------------------------------------------


def test_constraint_equation_is_classified_and_split() -> None:
    """The Helmholtz equation is a constraint, linear in v and explicit in u."""
    V, U, integrator = signal_integrator(N=10)
    assert integrator.transient_slots == (0,)
    assert integrator.constraint_slots == (1,)
    assert integrator.num_fields == 2

    (constraint,) = integrator.constraints
    # v is what the constraint is solved for, so it must reach the operator;
    # u belongs to the other equation and must not.
    assert "v" in str(constraint.linear_expr)
    assert "u" not in str(constraint.linear_expr)
    # Square in its own field, over its own test space.
    assert constraint.operator.shape == (V.dim, V.dim)
    # The inhomogeneous wall value lifts into a forcing vector.
    assert constraint.forcing is not None

    # The u coupling is linear in u alone, so it is assembled as an operator
    # between the two spaces rather than evaluated pointwise at every stage,
    # and nothing is left needing the physical-space evaluator.
    assert not constraint.has_nonlinear
    assert sp.sympify(constraint.nonlinear_expr) == 0
    ((coupling, _),) = constraint._couplings
    (slot,) = constraint._coupling_slots
    assert slot == 0  # u is the field of equation 0
    assert coupling.shape == (V.dim, U.dim)

    # The transported equation's `b*v` term is the same story, the other way.
    ((coupling, _),) = integrator.integrators[0]._couplings
    (slot,) = integrator.integrators[0]._coupling_slots
    assert slot == 1
    assert coupling.shape == (U.dim, V.dim)


def test_nonlinear_constraint_coupling_stays_pointwise() -> None:
    """A coupling that is *not* linear in one field keeps the pointwise path."""
    V, U = signal_spaces(N=10)
    v = TrialFunction(V, name="v")
    q = TestFunction(V, name="q")
    u = TrialFunction(U, name="u", transient=True)
    w = TestFunction(U, name="w")
    t = V.system.base_time()
    integrator = SystemIMEXRungeKutta(
        ((u.diff(t) + u + v) * w, (Div(Grad(v)) - v + u**2) * q),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.S.Zero, None),
        sparse=True,
    )
    (constraint,) = integrator.constraints
    assert constraint.has_nonlinear
    assert "u_jax" in str(constraint.nonlinear_expr)
    assert constraint._couplings == ()


def test_constraint_field_appears_in_the_state_in_declaration_order() -> None:
    """`solve` returns one array per equation, constrained fields included."""
    V, U, integrator = signal_integrator(N=10)
    u_hat, v_hat = integrator.solve(dt=0.05, steps=1, progress=False)
    assert u_hat.shape == U.num_dofs
    assert v_hat.shape == V.num_dofs


# ---------------------------------------------------------------------------
# The constraint solve itself
# ---------------------------------------------------------------------------


def _reference_pair(V, U):
    """Return `(v_ref, u_hat)` satisfying the constraint exactly."""
    v_ref = jax.random.normal(jax.random.PRNGKey(0), V.num_dofs) * 1e-1
    vf = JAXFunction(v_ref, V, name="vref")
    # The constraint reads lap(v) - v + u = 0, so u = v - lap(v).
    return v_ref, project(vf - Div(Grad(vf)), U)


@pytest.mark.parametrize("modes", [None, 14])
def test_constraint_recovers_the_constrained_field(modes) -> None:
    """Solving the constraint inverts `u = v - lap(v)` exactly.

    This is the sharpest check of the sign convention, of the boundary-lifting
    forcing and of the assembled coupling at once: given the source belonging to
    a known signal, the constraint must return that signal back.
    """
    V, U, integrator = signal_integrator(N=12, modes=modes)
    integrator.setup(0.01)
    (constraint,) = integrator.constraints

    v_ref, u_hat = _reference_pair(V, U)
    got = constraint.solve_field((u_hat, jnp.zeros(V.num_dofs)), U.shape)
    assert float(jnp.abs(got - v_ref).max()) / float(jnp.abs(v_ref).max()) < 1e-3


def test_truncated_space_cannot_represent_the_source() -> None:
    """`U` must hold `v - lap(v)` exactly, which needs the full N modes.

    Truncating it loses the source and the recovery above degrades badly.
    Pinned here so that the requirement is not silently weakened.
    """
    V, U, integrator = signal_integrator(N=12, modes=8)
    integrator.setup(0.01)
    (constraint,) = integrator.constraints

    v_ref, u_hat = _reference_pair(V, U)
    got = constraint.solve_field((u_hat, jnp.zeros(V.num_dofs)), U.shape)
    assert float(jnp.abs(got - v_ref).max()) / float(jnp.abs(v_ref).max()) > 0.1


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tableau", ALL_COMBINATION_PATHS)
def test_constrained_field_matches_analytic_solution(tableau) -> None:
    """`u_t = -c*p` with `p = u` integrates as `u = u0*exp(-c*t)`."""
    V, eq_u, eq_p, c, _, _ = decay_system()
    T = 0.5
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_p),
        tableau=tableau,
        time=(0.0, T),
        initial=(sp.sin(V.system.base_scalars()[0]), None),
        sparse=True,
    )
    u_hat, p_hat = integrator.solve(dt=T / 64, steps=64, progress=False)

    xj = V.mesh()
    exact = float(jnp.exp(-c * T)) * jnp.sin(xj)
    u = V.backward(u_hat).real
    assert float(jnp.linalg.norm(u - exact) / jnp.linalg.norm(exact)) < 1e-3
    # The constraint is p = u, so the two fields must come back identical.
    assert float(jnp.abs(V.backward(p_hat).real - u).max()) < 1e-4


@pytest.mark.parametrize("tableau", ALL_COMBINATION_PATHS)
def test_constrained_field_is_never_lagged(tableau) -> None:
    """After a step, v is the exact solve of its equation for the returned u.

    A lagged constraint would return the field belonging to some earlier stage.
    The check matters most for the tableaux whose accepted solution is not the
    last stage, which is why all three combination paths are covered.
    """
    _, _, integrator = signal_integrator(N=10, tableau=tableau)
    dt = 0.05
    integrator.setup(dt)
    state = integrator.initial_coefficients()
    # A non-trivial source, so the constraint has something to work with.
    state = integrator.resolve_constraints(
        (state[0] + 0.5, state[1]), integrator.common_padding(None)
    )

    u_hat, v_hat = integrator.step(state, dt)
    (constraint,) = integrator.constraints
    expected = constraint.solve_field(
        (u_hat, jnp.zeros_like(v_hat)), integrator.common_padding(None)
    )
    # Not exactly equal: `step` is jitted as a whole, so XLA fuses the same
    # solve differently there than in the standalone call above. What matters is
    # the ratio of the two scales -- agreement to round-off, against a field
    # that moved by orders of magnitude more over the step it would be lagged by.
    agreement = float(jnp.abs(v_hat - expected).max())
    moved = float(jnp.abs(v_hat - state[1]).max())
    assert agreement < 1e-4 * moved


def test_constrained_field_may_be_stated_or_derived() -> None:
    """`initial` may name the constrained field; omitting it derives the same one.

    Stating the signal and taking the source from it has to agree with letting
    the constraint recover the signal from that source.
    """
    V, U = signal_spaces(N=12)
    v_ref, u_hat = _reference_pair(V, U)

    def build(initial):
        return SystemIMEXRungeKutta(
            signal_system(V, U),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=initial,
            sparse=True,
        ).initial_coefficients()

    u_stated, v_stated = build((u_hat, v_ref))
    u_derived, v_derived = build((u_hat, None))

    assert float(jnp.abs(v_stated - v_ref).max()) == 0.0
    scale = max(float(jnp.abs(v_derived).max()), 1e-12)
    assert float(jnp.abs(v_stated - v_derived).max()) / scale < 1e-3
    assert float(jnp.abs(u_stated - u_derived).max()) == 0.0


def test_restart_state_reresolves_the_constraint() -> None:
    """A restart recomputes constrained fields instead of trusting them."""
    V, eq_u, eq_p, _, _, _ = decay_system()
    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_p),
        tableau=ARS443,
        time=(0.0, 0.1),
        initial=(sp.sin(V.system.base_scalars()[0]), None),
        sparse=True,
    )
    u_hat, p_hat = integrator.initial_coefficients()
    # Hand back a state whose constrained entry is nonsense.
    u_out, p_out = integrator._coerce_state((u_hat, p_hat + 17.0))
    assert float(jnp.abs(u_out - u_hat).max()) == 0.0
    assert float(jnp.abs(p_out - p_hat).max()) < 1e-5


def test_signal_system_integrates_stably() -> None:
    """Both terms of the u equation are dissipative, so this must settle down."""
    V, U, integrator = signal_integrator(N=10)
    u_hats, _ = integrator.solve(
        dt=0.02, steps=100, n_batches=4, return_batch_snapshots=True, progress=False
    )
    peak = jnp.max(jnp.abs(jax.vmap(U.backward)(u_hats)), axis=(1, 2))
    assert bool(jnp.isfinite(peak).all())
    # Monotone approach to equilibrium from rest, and bounded by it.
    assert bool(jnp.all(jnp.diff(peak) > 0))
    assert float(peak[-1]) < 2.0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_rejects_wrong_number_of_initial_conditions() -> None:
    _, _, integrator = signal_integrator(N=8)
    with pytest.raises(ValueError, match="one entry per equation"):
        integrator.initial_coefficients((sp.S.Zero,))


def test_rejects_missing_initial_condition_for_a_transient_field() -> None:
    """Only a constraint may be given None; an integrated field needs a value."""
    V, U = signal_spaces(N=8)
    with pytest.raises(ValueError, match="needs an initial"):
        SystemIMEXRungeKutta(
            signal_system(V, U),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=(None, None),
            sparse=True,
        )


def test_rejects_a_system_of_constraints_only() -> None:
    """Something has to evolve for there to be anything to integrate."""
    _, _, eq_p, _, _, _ = decay_system()
    with pytest.raises(ValueError, match="at least one equation with a time"):
        SystemIMEXRungeKutta(
            (eq_p,), tableau=ARS443, time=(0.0, 1.0), initial=(None,), sparse=True
        )


def test_rejects_a_constraint_with_an_ambiguous_field() -> None:
    """Two fields no equation evolves leaves nothing to solve the constraint for."""
    V = Fourier(8, Domain(0, 2 * sp.pi), name="Va", fun_str="Ea")
    g = TestFunction(V, name="g")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="u", transient=True)
    p = TrialFunction(V, name="p")
    r = TrialFunction(V, name="r")
    t = V.system.base_time()
    with pytest.raises(ValueError, match="exactly one field"):
        SystemIMEXRungeKutta(
            (g * (u.diff(t) + p), q * (p + r - u)),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=(sp.S.Zero, None),
            sparse=True,
        )


def test_rejects_a_constraint_that_is_not_linear_in_its_own_field() -> None:
    """A constraint has to be solvable for its field, so it must appear linearly."""
    V = Fourier(8, Domain(0, 2 * sp.pi), name="Vn", fun_str="En")
    g = TestFunction(V, name="g")
    q = TestFunction(V, name="q")
    u = TrialFunction(V, name="u", transient=True)
    p = TrialFunction(V, name="p")
    t = V.system.base_time()
    with pytest.raises(ValueError, match="assembled no operator"):
        SystemIMEXRungeKutta(
            (g * (u.diff(t) + p), q * (p**2 - u)),
            tableau=ARS443,
            time=(0.0, 1.0),
            initial=(sp.S.Zero, None),
            sparse=True,
        )


def test_constraint_solver_requires_a_field_order_when_coupled() -> None:
    """Without a global order there is no way to read the state tuple."""
    _, _, eq_p, _, u, p = decay_system()
    with pytest.raises(ValueError, match="field_order"):
        ConstraintSolver(
            eq_p,
            own_trial=get_time_independent(p),
            explicit_trials=(u,),
            sparse=True,
        )
