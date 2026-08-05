import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as Cheb
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import (
    ARK2_1_3L2SA,
    ARK3_2_4L2SA,
    ARK4_3_6L2SA,
    ARK5_4_8L2SA,
    ARS222,
    ARS443,
    ETDRK4,
    IMEX_EULER,
    IMEX_SSP2_222,
    BackwardEuler,
    IMEXRungeKutta,
)
from jaxfun.integrators.tableau import IMEXTableau
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify, n

pytestmark = pytest.mark.integration


def test_imex_euler_matches_backward_euler() -> None:
    """A single-stage IMEX-Euler tableau must reduce exactly to BackwardEuler."""
    N = 32
    c = Constant("c", 1.0)
    T = 0.2
    steps = 200
    dt = T / steps

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(x)
    weak_form = v * (u.diff(t) + c * u.diff(x))

    be = BackwardEuler(
        V, weak_form, time=(0.0, T), initial=u0, sparse=True, sparse_tol=1000
    )
    be_result = be.solve(dt=dt, steps=steps, progress=False)

    imex = IMEXRungeKutta(
        V,
        weak_form,
        tableau=IMEX_EULER,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    imex_result = imex.solve(dt=dt, steps=steps, progress=False)

    # Mathematically identical update, but a different summation order over
    # 200 complex64 steps accumulates float32 rounding differences.
    rel_error = jnp.linalg.norm(imex_result - be_result) / jnp.linalg.norm(be_result)
    assert float(rel_error) < 1e-4


def _chebyshev_diffusion_error(
    tableau: IMEXTableau, *, M: int, nu_val: float, T: float, steps: int
) -> float:
    nu = Constant("nu", nu_val)
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    V = FunctionSpace(M, Cheb, bcs=bcs, name="V", fun_str="psi", scaling=n + 1)

    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(sp.pi * (x + 1) / 2)
    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))

    integrator = IMEXRungeKutta(
        V,
        weak_form,
        tableau=tableau,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    dt = T / steps
    uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    u_num = V.backward(uhat_t).real
    k = sp.pi / 2
    u_ex = lambdify(x, sp.sin(k * (x + 1)) * sp.exp(-nu_val * float(k**2) * T))(xj)
    return float(jnp.linalg.norm(u_num - u_ex))


@pytest.mark.parametrize(
    "tableau,order,base_steps,T",
    [
        (ARK2_1_3L2SA, 2, 4, 0.4),
        (ARS222, 2, 4, 0.4),
        (ARK3_2_4L2SA, 3, 3, 0.4),
        (ARK4_3_6L2SA, 4, 1, 0.4),
        (ARS443, 3, 2, 0.4),
        (ARK5_4_8L2SA, 5, 2, 1.2),
    ],
    ids=[
        "ark2_1_3l2sa",
        "ars222",
        "ark3_2_4l2sa",
        "ark4_3_6l2sa",
        "ars443",
        "ark5_4_8l2sa",
    ],
)
def test_imex_rk_empirical_convergence_order(
    tableau: IMEXTableau, order: int, base_steps: int, T: float
) -> None:
    """Halving dt on a purely linear (implicit-only) problem should reduce the
    error by roughly 2**order, empirically validating both the stage-loop
    machinery and the transcribed tableau coefficients.

    ARK2(1)3L[2]SA/ARS222/ARK3(2)4L[2]SA/ARK4(3)6L[2]SA/ARS443/ARK5(4)8L[2]SA's
    DIRK companions are themselves full-order SDIRK methods, so this
    purely-linear (no explicit/nonlinear contribution) problem is a valid
    standalone check for them specifically. This is *not* generally true for
    IMEX pairs (e.g. IMEX-SSP2(2,2,2)'s DIRK companion is only 1st order
    alone -- its 2nd order claim only holds for the joint explicit+implicit
    problem), so this test is not parametrized over IMEX_SSP2_222; see
    `test_imex_ssp2_222_kdv_soliton_is_accurate` instead. ARS222/ARS443 are
    additionally globally stiffly accurate (`tableau.is_stiffly_accurate` is
    True), so this also exercises `step`'s last-stage shortcut path
    (`IMEX_EULER` is the other GSA scheme, already covered by
    `test_imex_euler_matches_backward_euler`). `T` is widened for
    `ARK5_4_8L2SA` (5th order) since its temporal error hits the float32
    noise floor almost immediately at the default `T`, well before two
    dt-halvings show clean asymptotic scaling.
    """
    M = 16
    nu_val = 0.5

    err1 = _chebyshev_diffusion_error(
        tableau, M=M, nu_val=nu_val, T=T, steps=base_steps
    )
    err2 = _chebyshev_diffusion_error(
        tableau, M=M, nu_val=nu_val, T=T, steps=2 * base_steps
    )
    observed_order = float(jnp.log2(err1 / err2))
    assert observed_order > order - 0.75


def test_imex_ssp2_222_kdv_soliton_is_accurate() -> None:
    """Smoke/accuracy check for IMEX-SSP2(2,2,2) on a genuine stiff-linear +
    nonstiff-nonlinear problem (its DIRK companion alone is only 1st order, so
    a pure-linear empirical-order check like the ARK3/ARK4 test above is not
    meaningful for this scheme -- see the note there)."""
    N = 64
    L = 20.0
    domain = Domain(-L, L)
    mu_val = 0.4
    mu = Constant("mu", mu_val)
    wave_speed = 0.5
    x0 = -5.0
    steps = 40
    dt = 1.25e-4
    T = steps * dt

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E", domain=domain)
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = 3 * wave_speed * sp.sech(0.5 * sp.sqrt(wave_speed) / mu_val * (x - x0)) ** 2
    weak_form = v * (u.diff(t) + u * u.diff(x) + mu**2 * u.diff(x, 3))

    ssp2 = IMEXRungeKutta(
        V,
        weak_form,
        tableau=IMEX_SSP2_222,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    uhat_t = ssp2.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    u_num = V.backward(uhat_t).real
    u_exact = (
        3
        * wave_speed
        * jnp.cosh(0.5 * jnp.sqrt(wave_speed) / mu_val * (xj - wave_speed * T - x0))
        ** -2
    )
    rel_error = jnp.linalg.norm(u_num - u_exact) / jnp.linalg.norm(u_exact)
    assert float(rel_error) < 1e-3


def test_ark4_kdv_soliton_matches_etdrk4_reference() -> None:
    """Cross-check IMEX-RK against the existing ETDRK4 reference on the same
    stiff-linear/nonstiff-nonlinear KdV soliton problem."""
    N = 64
    L = 20.0
    domain = Domain(-L, L)
    mu_val = 0.4
    mu = Constant("mu", mu_val)
    wave_speed = 0.5
    x0 = -5.0
    steps = 20
    dt = 2.5e-4
    T = steps * dt

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E", domain=domain)
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = 3 * wave_speed * sp.sech(0.5 * sp.sqrt(wave_speed) / mu_val * (x - x0)) ** 2
    weak_form = v * (u.diff(t) + u * u.diff(x) + mu**2 * u.diff(x, 3))

    ark = IMEXRungeKutta(
        V,
        weak_form,
        tableau=ARK4_3_6L2SA,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    uhat_ark = ark.solve(dt=dt, steps=steps, progress=False)

    etdrk4 = ETDRK4(
        V,
        weak_form,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    uhat_etdrk4 = etdrk4.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    u_ark = V.backward(uhat_ark).real
    u_etdrk4 = V.backward(uhat_etdrk4).real
    rel_error = jnp.linalg.norm(u_ark - u_etdrk4) / jnp.linalg.norm(u_etdrk4)
    assert float(rel_error) < 5e-3

    u_exact = (
        3
        * wave_speed
        * jnp.cosh(0.5 * jnp.sqrt(wave_speed) / mu_val * (xj - wave_speed * T - x0))
        ** -2
    )
    rel_error_exact = jnp.linalg.norm(u_ark - u_exact) / jnp.linalg.norm(u_exact)
    assert float(rel_error_exact) < 5e-3
