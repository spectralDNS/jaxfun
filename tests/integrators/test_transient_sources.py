"""Time integration with a source term that varies in time.

`tests/integrators/test_transient_bcs.py` covers boundary data that moves; this
covers the other half of a moving right-hand side. The two arrive by different
routes and meet at one place. A moving lifting is assembled as a boundary block
and re-contracted per stage; a moving source cannot reach `inner` at all, since
`_linear_form_scale` coerces a linear form's coefficient with `float()`. It is
therefore lifted out of the weak form before assembly and rebuilt from `t`
instead. Both then flow through `forcing_at`.

Supported sources are separable in time -- `f = sum_k g_k(t) * h_k(x)`. That is
not a restriction on the physics so much as on how it is written: `expand_trig`
brings a travelling wave into the form, and `tests/galerkin/test_inner_source.py`
pins what is accepted and what is refused.

Convergence order in `dt` is what these assert, not an error threshold. A source
frozen at `t=0` still converges in space, so a fixed tolerance at one resolution
would pass while the time dependence was being ignored.
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
from jaxfun.integrators import (
    ARK4_3_6L2SA,
    ARS222,
    ARS443,
    ETDRK4,
    IMEX_EULER,
    RK4,
    BackwardEuler,
    IMEXRungeKutta,
)
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify

pytestmark = pytest.mark.integration

NU = 0.1


def _manufactured_1d(N: int, tag: str):
    """`u_t = nu u_xx + f` whose exact solution decays, with homogeneous walls.

    `sin(pi x)` vanishes at both ends, so the boundary data is steady and every
    moving thing in the problem is the source. That separation is the point: a
    failure here cannot be blamed on the lifting.
    """
    R1 = R(1)
    (x,) = R1.base_scalars()
    t = R1.base_time()
    ue = sp.sin(sp.pi * x) * sp.exp(-3 * t)
    f = sp.diff(ue, t) - NU * sp.diff(ue, x, 2)
    V = FunctionSpace(
        24,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name=f"src{tag}",
        system=R1,
    )
    return V, ue, f, x, t


def _error_1d(cls, steps, tag, **kw) -> float:
    V, ue, f, x, t = _manufactured_1d(24, tag)
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    integrator = cls(
        v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)) - f),
        time=(0.0, 1.0),
        initial=ue.subs(t, 0),
        sparse=True,
        **kw,
    )
    u_hat = integrator.solve(dt=1.0 / steps, steps=steps, progress=False)
    xj = V.mesh(kind="uniform", N=60)
    got = V.evaluate(xj, u_hat)
    want = lambdify((x,), ue.subs(t, 1.0))(xj)
    return float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))


def _orders(errors: list[float]) -> list[float]:
    return [float(jnp.log2(a / b)) for a, b in zip(errors, errors[1:], strict=False)]


@pytest.mark.parametrize(
    ("cls", "kw", "tag", "design"),
    [
        (IMEXRungeKutta, {"tableau": IMEX_EULER}, "ie", 1),
        (BackwardEuler, {}, "be", 1),
        (IMEXRungeKutta, {"tableau": ARS222}, "a2", 2),
        (IMEXRungeKutta, {"tableau": ARS443}, "a4", 3),
    ],
)
def test_a_moving_source_converges_at_the_scheme_order(cls, kw, tag, design) -> None:
    """Each scheme reaches its design order with the source evaluated per stage.

    This is what says the stage times reach the source, not just the stepping:
    the forcing rides the implicit tableau as `sum_j a_i[i][j] f(t + c_i[j] dt)`,
    and a source held at the step's own `t` would cap every one of these at
    first order.
    """
    steps = [8, 16, 32, 64]
    errors = [_error_1d(cls, s, f"{tag}{s}", **kw) for s in steps]
    got = _orders(errors)
    assert all(design - 0.35 < o < design + 0.35 for o in got), (got, errors)


def test_a_high_order_scheme_reaches_round_off_with_a_moving_source() -> None:
    """Fourth order takes the error to the float32 floor, so it has no slope.

    Kept as a threshold rather than an order for exactly that reason: at 16
    steps `ARK4_3_6L2SA` is already at ~4e-7 here, and the next refinement
    measures round-off rather than the scheme.
    """
    assert _error_1d(IMEXRungeKutta, 16, "hi", tableau=ARK4_3_6L2SA) < 1e-5


def test_rk4_carries_a_moving_source() -> None:
    """`RK4` reaches the source through `total_rhs`, with four stage times.

    Given a problem with no spatial operator at all -- `u_t = f(x, t)` -- so
    that the source is the only thing driving it. That also sidesteps the
    stability limit: `RK4` is fully explicit, and spectral diffusion on this
    space would need a step around `1e-4` to stay stable, which measures the
    limit rather than the source.
    """
    R1 = R(1)
    (x,) = R1.base_scalars()
    t = R1.base_time()
    ue = sp.sin(sp.pi * x) * sp.exp(-3 * t)
    V = FunctionSpace(
        24,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name="rk4src",
        system=R1,
    )
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    integrator = RK4(
        v * (u.diff(t) - sp.diff(ue, t)),
        time=(0.0, 1.0),
        initial=ue.subs(t, 0),
        sparse=True,
    )
    assert integrator._transient_source
    u_hat = integrator.solve(dt=0.05, steps=20, progress=False)
    xj = V.mesh(kind="uniform", N=60)
    got = V.evaluate(xj, u_hat)
    want = lambdify((x,), ue.subs(t, 1.0))(xj)
    assert float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)) < 1e-4


def test_a_frozen_source_would_not_converge() -> None:
    """The control for the assertions above.

    Evaluating `f` once at `t=0` and reusing it is exactly the bug this feature
    exists to avoid. The resulting error is independent of `dt`, so an order
    assertion catches it where an error threshold at one resolution would not.
    """
    V, ue, f, x, t = _manufactured_1d(24, "frozen")
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    frozen = f.subs(t, 0)
    errors = []
    for steps in (8, 64):
        integrator = IMEXRungeKutta(
            v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)) - frozen),
            tableau=ARK4_3_6L2SA,
            time=(0.0, 1.0),
            initial=ue.subs(t, 0),
            sparse=True,
        )
        u_hat = integrator.solve(dt=1.0 / steps, steps=steps, progress=False)
        xj = V.mesh(kind="uniform", N=60)
        got = V.evaluate(xj, u_hat)
        want = lambdify((x,), ue.subs(t, 1.0))(xj)
        errors.append(float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want)))
    assert errors[0] > 1.0, errors
    assert abs(_orders(errors)[0]) < 0.1, (
        "a frozen source must not converge in dt",
        errors,
    )


def test_a_moving_source_in_two_dimensions() -> None:
    """Fourier x Legendre, where the source spans both directions."""
    R2 = R(2)
    x, y = R2.base_scalars()
    t = R2.base_time()
    ue = sp.cos(x) * sp.sin(sp.pi * y) * sp.exp(-t)
    f = sp.diff(ue, t) - NU * Div(Grad(ue))
    T = TensorProduct(
        Fourier.Fourier(12, name="F2s"),
        FunctionSpace(
            16,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            name="D2s",
        ),
        name="T2s",
        system=R2,
    )
    v = TestFunction(T)
    u = TrialFunction(T, transient=True)
    integrator = IMEXRungeKutta(
        v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)) - f),
        tableau=ARK4_3_6L2SA,
        time=(0.0, 1.0),
        initial=ue.subs(t, 0),
        sparse=True,
    )
    assert integrator._transient_source
    u_hat = integrator.solve(dt=0.02, steps=50, progress=False)
    got = T.backward(u_hat)
    want = lambdify((x, y), ue.subs(t, 1.0))(*T.mesh())
    error = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
    assert error < 1e-5, error


def test_a_steady_source_keeps_exactly_the_old_forcing() -> None:
    """Nothing is deferred when nothing moves, and the old object is returned."""
    R1 = R(1)
    (x,) = R1.base_scalars()
    t = R1.base_time()
    V = FunctionSpace(
        16,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name="steadysrc",
        system=R1,
    )
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    integrator = IMEXRungeKutta(
        v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)) - sp.sin(sp.pi * x)),
        tableau=ARK4_3_6L2SA,
        time=(0.0, 1.0),
        initial=sp.Integer(0),
        sparse=True,
    )
    assert not integrator._transient_source
    assert not integrator._transient_forcing
    assert integrator._source is None
    assert integrator.forcing_at(0.7) is integrator.linear_forcing


def test_etdrk4_rejects_a_source_that_moves() -> None:
    """Its phi-propagators fold the forcing in at setup; a moving one is invalid."""
    V, ue, f, _x, t = _manufactured_1d(24, "etd")
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    with pytest.raises(NotImplementedError, match="time-dependent source"):
        ETDRK4(
            v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)) - f),
            time=(0.0, 1.0),
            initial=ue.subs(t, 0),
            sparse=True,
        )


def test_a_time_dependent_operator_coefficient_is_refused() -> None:
    """Operators are assembled and factorized once, at construction."""
    R1 = R(1)
    (x,) = R1.base_scalars()
    t = R1.base_time()
    V = FunctionSpace(
        16,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name="opcoeff",
        system=R1,
    )
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    with pytest.raises(ValueError, match="bilinear term"):
        IMEXRungeKutta(
            v * (u.diff(t) - sp.exp(-t) * Div(Grad(u))),
            tableau=ARK4_3_6L2SA,
            time=(0.0, 1.0),
            initial=sp.sin(sp.pi * x),
            sparse=True,
        )


def test_a_time_dependent_nonlinear_term_is_refused() -> None:
    """Pointwise evaluation lambdifies against the coordinates alone."""
    R1 = R(1)
    (x,) = R1.base_scalars()
    t = R1.base_time()
    V = FunctionSpace(
        16,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        name="nlsrc",
        system=R1,
    )
    v = TestFunction(V)
    u = TrialFunction(V, transient=True)
    with pytest.raises(NotImplementedError, match="Nonlinear term depends on"):
        IMEXRungeKutta(
            v * (u.diff(t) - Div(Grad(u)) - sp.exp(-t) * u**2),
            tableau=ARK4_3_6L2SA,
            time=(0.0, 1.0),
            initial=sp.sin(sp.pi * x),
            sparse=True,
        )
