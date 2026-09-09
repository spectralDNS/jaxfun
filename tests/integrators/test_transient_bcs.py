"""Time integration with boundary data that varies in time.

`tests/integrators/test_inhomogeneous_bcs.py` covers the steady case, and its
header states the invariant that makes it work: the lifting enters the weak form
under the time derivative as `inner(v, B)`, which drops out because `B` is
constant. Once `B` moves, that term is `-<v, dB/dt>` and has to be carried, and
the stiffness contribution `a(v, B(t))` has to be re-evaluated rather than
frozen at the value it had when the space was built.

Every exact solution here solves the heat equation *exactly*, so no source term
is needed -- time-dependent sources are a separate feature the integrators do
not have. The boundary data still moves, which is the point.

Convergence order is what these assert, not an error threshold. A frozen lifting
still converges in space, so a fixed tolerance on a single resolution would pass
while the time dependence was being ignored entirely; the order in `dt` is what
actually detects it.
"""

import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Fourier,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.tensorproductspace import DirectSumTPS
from jaxfun.integrators import (
    ARK4_3_6L2SA,
    ETDRK4,
    IMEX_EULER,
    BackwardEuler,
    IMEXRungeKutta,
)
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify

pytestmark = pytest.mark.integration

NU = 0.5
X, Y = sp.symbols("x y", real=True)


def _decaying_mode_1d(N: int, tag: str):
    """`u = exp(-4*nu*t)*sin(2x)` on [-1, 1]: an exact solution with moving walls."""
    t = FunctionSpace(N, Legendre.Legendre).system.base_time()
    ue = sp.exp(-4 * NU * t) * sp.sin(2 * X)
    V = FunctionSpace(
        N,
        Legendre.Legendre,
        bcs={"left": {"D": ue.subs(X, -1)}, "right": {"D": ue.subs(X, 1)}},
        name=f"V{tag}",
    )
    return V, ue, t


def _heat_equation(V, t):
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    return v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)))


def _error_1d(cls, steps, T, N, tag, **kw):
    V, ue, t = _decaying_mode_1d(N, tag)
    (x,) = V.system.base_scalars()
    integrator = cls(
        _heat_equation(V, t), time=(0.0, T), initial=sp.sin(2 * x), sparse=True, **kw
    )
    u_hat = integrator.solve(dt=T / steps, steps=steps, progress=False)
    xj = V.mesh(kind="uniform", N=50)
    got = V.evaluate(xj, u_hat, t=integrator.end_time(T / steps, steps))
    want = lambdify((x,), ue.subs(t, T).subs(X, x))(xj)
    return float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))


def _orders(errors: list[float]) -> list[float]:
    return [float(jnp.log2(a / b)) for a, b in zip(errors, errors[1:], strict=False)]


@pytest.mark.parametrize(
    ("cls", "kw", "tag"),
    [
        (IMEXRungeKutta, {"tableau": IMEX_EULER}, "ie"),
        (BackwardEuler, {}, "be"),
    ],
)
def test_a_moving_wall_converges_at_first_order(cls, kw, tag) -> None:
    """The sharpest probe: a frozen lifting does not converge in `dt` at all.

    Both schemes are first order, so the time error dominates by orders of
    magnitude over the spatial and round-off errors, and the rate is clean.
    """
    steps = [8, 16, 32, 64]
    errors = [_error_1d(cls, s, 1.0, 24, f"{tag}{s}", **kw) for s in steps]
    assert errors[0] > 1e-3, (
        "the time error must dominate for the order to mean anything"
    )
    assert all(0.85 < order < 1.25 for order in _orders(errors)), _orders(errors)


def test_a_high_order_scheme_reaches_round_off_with_a_moving_wall() -> None:
    """Fourth order takes the error straight to the float32 floor."""
    assert _error_1d(IMEXRungeKutta, 16, 1.0, 24, "hi", tableau=ARK4_3_6L2SA) < 1e-5


def test_a_dirichlet_lifting_moves_only_through_the_time_derivative() -> None:
    """Why the problems above probe exactly the term that used to be dropped.

    A two-point Dirichlet lifting is linear in x, so its Laplacian vanishes and
    `a(v, B(t))` is identically zero however the wall moves. Everything the
    boundary contributes therefore arrives as `-<v, dB/dt>` -- the term a steady
    problem is entitled to discard.
    """
    V, ue, t = _decaying_mode_1d(24, "probe")
    (x,) = V.system.base_scalars()
    integrator = IMEXRungeKutta(
        _heat_equation(V, t),
        tableau=ARK4_3_6L2SA,
        time=(0.0, 1.0),
        initial=sp.sin(2 * x),
        sparse=True,
    )
    assert integrator._transient_boundary
    assert integrator._linear_boundary is not None
    assert integrator._mass_boundary is not None
    assert float(jnp.abs(integrator._linear_boundary(0.3)).max()) == 0.0
    assert float(jnp.abs(integrator._mass_boundary.rate(0.3)).max()) > 1e-2


def _error_2d(steps, T, both_axes: bool, tag: str) -> float:
    if both_axes:
        # u = exp(-2*nu*a^2*t) sin(ax) sin(ay): every wall moves.
        a = 2.0
        t = FunctionSpace(8, Legendre.Legendre).system.base_time()
        ue = sp.exp(-2 * NU * a**2 * t) * sp.sin(a * X) * sp.sin(a * Y)
        space = TensorProduct(
            FunctionSpace(
                16,
                Legendre.Legendre,
                bcs={"left": {"D": ue.subs(X, -1)}, "right": {"D": ue.subs(X, 1)}},
                name=f"Dx{tag}",
            ),
            FunctionSpace(
                16,
                Legendre.Legendre,
                bcs={"left": {"D": ue.subs(Y, -1)}, "right": {"D": ue.subs(Y, 1)}},
                name=f"Dy{tag}",
            ),
            name=f"T{tag}",
        )
    else:
        # Fourier(x) x Legendre(y): the wall value moves in time *and* in x.
        k, m = 1, 2.0
        F = Fourier.Fourier(12, name=f"F{tag}")
        t = F.system.base_time()
        ue = sp.exp(-NU * (k**2 + m**2) * t) * sp.cos(k * X) * sp.sin(m * Y)
        space = TensorProduct(
            F,
            FunctionSpace(
                16,
                Legendre.Legendre,
                bcs={"left": {"D": ue.subs(Y, -1)}, "right": {"D": ue.subs(Y, 1)}},
                name=f"D{tag}",
            ),
            name=f"T{tag}",
        )

    assert isinstance(space, DirectSumTPS)  # inhomogeneous, so it holds a lifting
    v = TestFunction(space, name="v")
    u = TrialFunction(space, name="u", transient=True)
    integrator = IMEXRungeKutta(
        v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u))),
        tableau=ARK4_3_6L2SA,
        time=(0.0, T),
        initial=space.system.expr_psi_to_base_scalar(ue.subs(t, 0)),
        sparse=True,
        # The two-axis operator is banded enough to warn but not to pay off.
        solver_options={"auto_threshold": 100_000},
    )
    u_hat = integrator.solve(dt=T / steps, steps=steps, progress=False)
    got = space.backward(u_hat, t=integrator.end_time(T / steps, steps))
    want = lambdify(
        space.system.base_scalars(),
        space.system.expr_psi_to_base_scalar(ue.subs(t, T)),
    )(*space.mesh())
    return float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))


@pytest.mark.parametrize("both_axes", [False, True], ids=["one_axis", "both_axes"])
def test_a_moving_wall_in_two_dimensions(both_axes: bool) -> None:
    """With both axes inhomogeneous the corner block moves too."""
    tag = "b" if both_axes else "o"
    assert _error_2d(16, 0.5, both_axes, tag) < 1e-4


def test_etdrk4_rejects_boundary_data_that_moves() -> None:
    """Its propagators fold a constant forcing in at setup; silence would be worse."""
    V, _, t = _decaying_mode_1d(16, "etd")
    (x,) = V.system.base_scalars()
    with pytest.raises(NotImplementedError, match="time-dependent"):
        ETDRK4(_heat_equation(V, t), time=(0.0, 0.1), initial=sp.sin(2 * x))


def test_steady_boundary_data_keeps_exactly_the_old_forcing() -> None:
    """No new arithmetic on the steady path -- the same object comes back."""
    V = FunctionSpace(
        16, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 1}}, name="Vst"
    )
    t = V.system.base_time()
    (x,) = V.system.base_scalars()
    integrator = IMEXRungeKutta(
        _heat_equation(V, t),
        tableau=ARK4_3_6L2SA,
        time=(0.0, 0.1),
        initial=sp.sin(sp.pi * x),
        sparse=True,
    )
    assert not integrator._transient_boundary
    assert integrator._linear_boundary is None
    assert integrator._mass_boundary is None
    assert integrator.forcing_at(0.7) is integrator.linear_forcing


def test_a_constraint_reads_its_wall_at_the_stage_time() -> None:
    """An algebraic equation has no `dB/dt`, but its lifting still has to move.

    `u` decays towards `-b/a * v`, and `v` is recovered at every stage from a
    screened Helmholtz problem whose wall value is prescribed and moving. If the
    constraint froze its lifting, `v` would settle to the wrong profile and `u`
    with it.
    """
    from jaxfun.integrators import ARS443, SystemIMEXRungeKutta

    N = 12
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    Vy = FunctionSpace(N, Legendre.Legendre, bcs=hom, name="cVy")
    t = Vy.system.base_time()
    wall = (1 - X**2) * sp.exp(-t)
    V = TensorProduct(
        FunctionSpace(N, Legendre.Legendre, bcs=hom, name="cVx", fun_str="Lvx"),
        FunctionSpace(
            N,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": wall}},
            name="cVys",
            fun_str="Lvy",
        ),
        name="cVsig",
    )
    assert isinstance(V, DirectSumTPS)  # the moving wall gives it a lifting
    U = V.get_orthogonal()

    v = TrialFunction(V, name="v")  # constrained: no transient=True
    q = TestFunction(V, name="q")
    u = TrialFunction(U, name="u", transient=True)
    w = TestFunction(U, name="w")
    eq_u = (u.diff(t) + u + v) * w
    eq_v = (Div(Grad(v)) - v + u) * q

    integrator = SystemIMEXRungeKutta(
        (eq_u, eq_v),
        tableau=ARS443,
        time=(0.0, 1.0),
        initial=(sp.Integer(0), None),
        sparse=True,
    )
    (constraint,) = integrator.constraints
    assert constraint._boundary is not None, "the moving wall should force it"
    first, last = constraint.forcing_at(0.0), constraint.forcing_at(1.0)
    assert first is not None and last is not None
    assert not jnp.allclose(first, last)

    u_hats, v_hats = integrator.solve(
        dt=0.02, steps=50, n_batches=5, return_batch_snapshots=True, progress=False
    )
    assert jnp.all(jnp.isfinite(u_hats)) and jnp.all(jnp.isfinite(v_hats))
    # The wall decays like exp(-t), so the recovered field has to decay with it.
    early = float(jnp.abs(V.backward(v_hats[1], t=0.2)).max())
    late = float(jnp.abs(V.backward(v_hats[-1], t=1.0)).max())
    assert late < 0.6 * early, (early, late)
