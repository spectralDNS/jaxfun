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

from jaxfun.coordinates import R
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
from jaxfun.integrators._utils import apply_field_couplings
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify

pytestmark = pytest.mark.integration

NU = 0.5


def _decaying_mode_1d(N: int, tag: str):
    """`u = exp(-4*nu*t)*sin(2x)` on [-1, 1]: an exact solution with moving walls."""
    R1 = R(1)
    x = R1.x
    t = R1.base_time()
    ue = sp.exp(-4 * NU * t) * sp.sin(2 * x)
    V = FunctionSpace(
        N,
        Legendre.Legendre,
        bcs={"left": {"D": ue.subs(x, -1)}, "right": {"D": ue.subs(x, 1)}},
        name=f"V{tag}",
    )
    return V, ue, t


def _heat_equation(V, t):
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    return v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)))


def _error_1d(cls, steps, T, N, tag, **kw):
    V, ue, t = _decaying_mode_1d(N, tag)
    x = V.system.x
    integrator = cls(
        _heat_equation(V, t), time=(0.0, T), initial=sp.sin(2 * x), sparse=True, **kw
    )
    u_hat = integrator.solve(dt=T / steps, steps=steps, progress=False)
    xj = V.mesh(kind="uniform", N=50)
    got = V.evaluate(xj, u_hat, t=integrator.end_time(T / steps, steps))
    want = lambdify((x,), ue.subs(t, T))(xj)
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


def _drifting_ramp_1d(N: int, tag: str):
    """A problem with a moving wall and *no* linear spatial term.

    `u_t = f(x)` with `f` steady, whose exact solution `(1+x)/2 * (1+t)` rides a
    wall that moves. There is nothing for `a(v, B(t))` to come from, so the only
    boundary contribution is `-<v, dB/dt>`.
    """
    R1 = R(1)
    (x,) = R1.base_scalars()
    t = R1.base_time()
    ue = (1 + x) / 2 * (1 + t)
    V = FunctionSpace(
        N,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 1 + t}},
        name=f"ramp{tag}",
        system=R1,
    )
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    integrator = IMEXRungeKutta(
        v * (u.diff(t) - sp.Rational(1, 2) * (1 + x)),
        tableau=ARK4_3_6L2SA,
        time=(0.0, 1.0),
        initial=ue.subs(t, 0),
        sparse=True,
    )
    return V, integrator, ue, x, t


def test_the_lifting_rate_is_kept_without_a_linear_boundary_block() -> None:
    """The two boundary blocks are independent; neither gates the other.

    `a(v, B(t))` comes from the linear spatial term and `-<v, dB/dt>` from the
    time derivative. An equation with no linear spatial term assembles the first
    as `None` while still owing the second, so treating the linear block as the
    gate returned `linear_forcing` and dropped the rate.
    """
    _V, integrator, _ue, _x, _t = _drifting_ramp_1d(16, "gate")
    assert integrator._transient_boundary
    assert integrator._linear_boundary is None, "no linear spatial term to assemble"
    assert integrator._mass_boundary is not None
    assert integrator.linear_forcing is not None, "the steady source is still there"

    got = integrator.forcing_at(0.3)
    assert got is not None
    want = -integrator._mass_boundary.rate(0.3) + jnp.asarray(integrator.linear_forcing)
    assert jnp.allclose(got, want)
    # The rate is substantial here, so returning `linear_forcing` alone is a
    # visibly different answer rather than a rounding difference.
    assert not jnp.allclose(got, jnp.asarray(integrator.linear_forcing))


def test_a_moving_wall_is_exact_without_a_linear_spatial_term() -> None:
    """End-to-end: the dropped rate term was a 49% error, not a small one.

    The exact solution is linear in `x` and in `t`, so a fourth-order scheme on a
    spectral space should reach round-off. It does once the rate is carried; with
    it dropped the answer is off by half.
    """
    V, integrator, ue, x, _t = _drifting_ramp_1d(16, "exact")
    dt, steps = 0.01, 100
    u_hat = integrator.solve(dt=dt, steps=steps, progress=False)
    xj = V.mesh(kind="uniform", N=40)
    got = V.evaluate(xj, u_hat, t=integrator.end_time(dt, steps))
    want = lambdify((x,), ue.subs(_t, 1.0))(xj)
    error = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
    assert error < 1e-5, error
    assert float(jnp.abs(integrator._mass_boundary.rate(0.3)).max()) > 1e-2


def _error_2d(steps, T, both_axes: bool, tag: str) -> float:
    R2 = R(2)
    x, y = R2.base_scalars()
    t = R2.base_time()
    if both_axes:
        # u = exp(-2*nu*a^2*t) sin(ax) sin(ay): every wall moves.
        a = 2.0
        ue = sp.exp(-2 * NU * a**2 * t) * sp.sin(a * x) * sp.sin(a * y)
        space = TensorProduct(
            FunctionSpace(
                16,
                Legendre.Legendre,
                bcs={"left": {"D": ue.subs(x, -1)}, "right": {"D": ue.subs(x, 1)}},
                name=f"Dx{tag}",
            ),
            FunctionSpace(
                16,
                Legendre.Legendre,
                bcs={"left": {"D": ue.subs(y, -1)}, "right": {"D": ue.subs(y, 1)}},
                name=f"Dy{tag}",
            ),
            name=f"T{tag}",
        )
    else:
        # Fourier(x) x Legendre(y): the wall value moves in time *and* in x.
        k, m = 1, 2.0
        F = Fourier.Fourier(12, name=f"F{tag}")
        ue = sp.exp(-NU * (k**2 + m**2) * t) * sp.cos(k * x) * sp.sin(m * y)
        space = TensorProduct(
            F,
            FunctionSpace(
                16,
                Legendre.Legendre,
                bcs={"left": {"D": ue.subs(y, -1)}, "right": {"D": ue.subs(y, 1)}},
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
        initial=ue.subs(t, 0),
        sparse=True,
        # The two-axis operator is banded enough to warn but not to pay off.
        solver_options={"auto_threshold": 100_000},
    )
    u_hat = integrator.solve(dt=T / steps, steps=steps, progress=False)
    got = space.backward(u_hat, t=integrator.end_time(T / steps, steps))
    want = lambdify((x, y), ue.subs(t, T))(*space.mesh())
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
    R2 = R(2)
    x, _ = R2.base_scalars()
    t = R2.base_time()
    wall = (1 - x**2) * sp.exp(-t)
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


def _diffusion_driven_by_a_foreign_wall(N: int = 12):
    """A field with no boundary conditions, coupled to one with a moving wall.

    `u` lives in an orthogonal space -- no boundary data of its own, so nothing
    about *its* equation looks transient. All the motion arrives through the
    coupling to `v`, whose wall decays like `exp(-t)`.
    """
    from jaxfun.integrators import ARS443, SystemIMEXRungeKutta

    hom = {"left": {"D": 0}, "right": {"D": 0}}
    R2 = R(2)
    x, _ = R2.base_scalars()
    t = R2.base_time()
    wall = (1 - x**2) * sp.exp(-t)
    V = TensorProduct(
        FunctionSpace(N, Legendre.Legendre, bcs=hom, name="fVx", fun_str="Lfx"),
        FunctionSpace(
            N,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": wall}},
            name="fVy",
            fun_str="Lfy",
        ),
        name="fVsig",
    )
    assert isinstance(V, DirectSumTPS)
    U = V.get_orthogonal()
    v = TrialFunction(V, name="v")
    q = TestFunction(V, name="q")
    u = TrialFunction(U, name="u", transient=True)
    w = TestFunction(U, name="w")
    integrator = SystemIMEXRungeKutta(
        ((u.diff(t) + u + v) * w, (Div(Grad(v)) - v + u) * q),
        tableau=ARS443,
        time=(0.0, 6.0),
        initial=(sp.Integer(0), None),
        sparse=True,
    )
    return integrator, U


def test_a_coupling_carries_its_foreign_wall_deferred() -> None:
    """A coupling to a field whose wall moves must not be collapsed at one time.

    `inner` folds the foreign space's boundary block into a load vector against
    the space's frozen `bndvals`. Stored that way it is `B_v(0)` forever, while
    the foreign field's own coefficients are solved relative to `B_v(t)` -- the
    two halves of one field, read at different instants.

    Nothing about the coupled equation itself looks transient, which is what
    makes this easy to miss: its own space has no boundary conditions at all.
    """
    integrator, _ = _diffusion_driven_by_a_foreign_wall()
    equation = integrator.integrators[0]
    # The gate on the equation's *own* space is correctly False -- the motion is
    # entirely in the coupling, which is the point.
    assert not equation._transient_boundary
    ((operator, forcing, boundary),) = equation._couplings
    assert boundary is not None, "the foreign wall must be kept deferred"
    assert boundary.is_transient
    # A coupling's only linear contribution is the lifting, so removing the
    # frozen copy leaves nothing behind.
    assert forcing is None or float(jnp.abs(jnp.asarray(forcing)).max()) == 0.0

    # A zero state isolates the boundary block: the operator contributes
    # nothing, so what is left is the lifting alone.
    state = tuple(jnp.zeros_like(c) for c in integrator.initial_coefficients())
    slots = equation._coupling_slots
    early = apply_field_couplings(slots, equation._couplings, state, 0.0)
    late = apply_field_couplings(slots, equation._couplings, state, 1.0)
    assert early is not None and late is not None
    assert not jnp.allclose(early, late)
    # The wall is the only thing moving, and it decays like exp(-t).
    ratio = float(jnp.abs(late).max() / jnp.abs(early).max())
    assert ratio == pytest.approx(float(jnp.exp(jnp.array(-1.0))), rel=1e-4)
    assert operator is not None


def test_a_coupled_field_decays_with_the_foreign_wall() -> None:
    """The end-to-end consequence of freezing a coupling's lifting.

    `u` is driven only through `v`, and `v` is driven only by a wall that decays
    like `exp(-t)`. So `u` has to decay too. Held at `B_v(0)`, the coupling is a
    source that never switches off and `u` instead grows to a steady state of
    its own -- roughly 0.96 here, against the 4e-2 it should have fallen to.

    Asserted as decay rather than a threshold: the wrong answer is not a small
    perturbation of the right one, it has the opposite sign of slope.
    """
    integrator, U = _diffusion_driven_by_a_foreign_wall()
    u_hats, v_hats = integrator.solve(
        dt=0.05, steps=120, n_batches=6, return_batch_snapshots=True, progress=False
    )
    assert jnp.all(jnp.isfinite(u_hats)) and jnp.all(jnp.isfinite(v_hats))
    peak = float(jnp.abs(U.backward(u_hats[1])).max())
    final = float(jnp.abs(U.backward(u_hats[-1])).max())
    assert final < 0.2 * peak, (peak, final)


def test_a_run_starting_after_zero_lifts_its_initial_state_there() -> None:
    """A nonzero `time[0]` has to reach the initial state, not just the stepping.

    The state stores the *homogeneous* part, so building it means taking a
    lifting back out, and which lifting depends on when the run starts. A
    constrained field is the sharper case: it is algebraic, so it is solved
    outright from the wall at that instant rather than merely started there.
    Resolving it at zero while stepping from `time[0]` leaves the first step
    reading a field that belongs to a different time.
    """
    from jaxfun.integrators import ARS443, SystemIMEXRungeKutta

    N = 12
    t_start = 2.0
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    R2 = R(2)
    x, _ = R2.base_scalars()
    t = R2.base_time()
    wall = (1 - x**2) * sp.exp(-t)
    V = TensorProduct(
        FunctionSpace(N, Legendre.Legendre, bcs=hom, name="sVx", fun_str="Lsx"),
        FunctionSpace(
            N,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": wall}},
            name="sVy",
            fun_str="Lsy",
        ),
        name="sVsig",
    )
    assert isinstance(V, DirectSumTPS)
    U = V.get_orthogonal()
    v = TrialFunction(V, name="v")
    q = TestFunction(V, name="q")
    u = TrialFunction(U, name="u", transient=True)
    w = TestFunction(U, name="w")

    integrator = SystemIMEXRungeKutta(
        ((u.diff(t) + u + v) * w, (Div(Grad(v)) - v + u) * q),
        tableau=ARS443,
        time=(t_start, t_start + 1.0),
        initial=(sp.Integer(0), None),
        sparse=True,
    )
    (slot,) = integrator.constraint_slots
    state = integrator.initial_coefficients()
    at_start = integrator.resolve_constraints(tuple(state), t=t_start)
    at_zero = integrator.resolve_constraints(tuple(state), t=0.0)
    # The wall decays by e^-2 over this interval, so the two are far apart --
    # without which the assertion below could not tell them apart.
    assert float(jnp.abs(at_start[slot] - at_zero[slot]).max()) > 1e-2
    assert jnp.array_equal(state[slot], at_start[slot])

    # A restart must land in the same place rather than re-deriving at zero.
    coerced = integrator._coerce_state(tuple(state))
    assert jnp.array_equal(coerced[slot], at_start[slot])


def test_a_trange_override_moves_the_initial_state_with_it() -> None:
    """`solve(trange=...)` changes where the run begins, lifting included.

    `time[0]` is what the integrator was built with; `trange` overrides it per
    call. The start time the stepping uses and the one the initial state is
    lifted against are then two different numbers unless the override is
    threaded through, and the state would be built for a time the run never
    visits.
    """
    V, ue, t = _decaying_mode_1d(24, "trange")
    (x,) = V.system.base_scalars()
    integrator = IMEXRungeKutta(
        _heat_equation(V, t),
        tableau=ARK4_3_6L2SA,
        time=(0.0, 1.0),
        initial=ue.subs(t, 0),
        sparse=True,
    )
    at_two = integrator.initial_coefficients(t=2.0)
    at_zero = integrator.initial_coefficients(t=0.0)
    assert float(jnp.abs(at_two - at_zero).max()) > 1e-2

    started = integrator.solve(
        dt=0.05,
        trange=(2.0, 2.1),
        steps=2,
        n_batches=1,
        return_batch_snapshots=True,
        progress=False,
    )[0]
    assert jnp.array_equal(started, at_two)
