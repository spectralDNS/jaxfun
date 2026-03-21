import warnings

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProductSpace
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ETDRK4
from jaxfun.integrators.etdrk4 import _etdrk4_diag_coeffs
from jaxfun.operators import Constant
from jaxfun.utils.common import lambdify


def _count_primitive(jpr, primitive: str) -> int:
    total = 0
    for eqn in jpr.eqns:
        if str(eqn.primitive) == primitive:
            total += 1
        for val in eqn.params.values():
            if hasattr(val, "jaxpr") and hasattr(val, "consts"):
                total += _count_primitive(val.jaxpr, primitive)
            elif isinstance(val, tuple | list):
                for item in val:
                    if hasattr(item, "jaxpr") and hasattr(item, "consts"):
                        total += _count_primitive(item.jaxpr, primitive)
    return total


def test_etdrk4_zero_linear_coefficients_match_rk4_limit() -> None:
    Ldiag = jnp.zeros(8)
    E, E2, Q, f1, f2, f3 = _etdrk4_diag_coeffs(Ldiag, dt=0.1)
    ones = jnp.ones_like(Ldiag)
    sixth = ones / 6

    assert jnp.allclose(E, ones)
    assert jnp.allclose(E2, ones)
    assert jnp.allclose(Q, 0.5 * ones)
    assert jnp.allclose(f1, sixth)
    assert jnp.allclose(f2, sixth)
    assert jnp.allclose(f3, sixth)


def test_etdrk4_setup_stage_coefficient_matches_coeff_builder() -> None:
    N = 16
    c = Constant("c", 1.0)
    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    weak_form = v * (u.diff(t) + c * u.diff(x))
    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, 0.1),
        initial=sp.cos(x),
        sparse=True,
        sparse_tol=1000,
    )
    dt = 0.1
    integrator.setup(dt=dt)

    assert bool(integrator.is_diag)
    assert integrator.mass_diag is not None
    assert integrator.linear_diag is not None

    Ldiag = integrator.linear_diag / integrator.mass_diag
    _, _, expected_Q, _, _, _ = _etdrk4_diag_coeffs(Ldiag, dt)
    assert jnp.allclose(jnp.asarray(integrator.Q), expected_Q)


def test_etdrk4_solve_and_frames_interface() -> None:
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

    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    u_num = V.backward(uhat_t).real
    u_ex = lambdify(x, sp.sin(x - c.val * T))(xj)
    rel_error = jnp.linalg.norm(u_num - u_ex) / jnp.linalg.norm(u_ex)
    assert float(rel_error) < 1e-4


@pytest.mark.parametrize(
    "u0f, domain",
    [(lambda x: sp.cos(sp.pi * x), Domain(-1, 1)), (lambda x: sp.cos(x), None)],
)
def test_etdrk4_kdv_short_run_is_finite(u0f, domain) -> None:
    N = 32
    mu = Constant("mu", sp.Rational(11, 500))
    steps = 8
    dt = 2.5e-4

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E", domain=domain)
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = u0f(x)
    # u0 = sp.cos(sp.pi * x)
    weak_form = v * (u.diff(t) + (u * u).diff(x) / 2 + mu**2 * u.diff(x, 3))

    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, dt * steps),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    states = integrator.solve(
        dt=dt,
        steps=steps,
        n_batches=4,
        return_each_step=True,
        progress=False,
    )

    assert states.shape == (5, V.num_dofs)

    u0_phys = V.backward(states[0]).real
    uT_phys = V.backward(states[-1]).real
    assert bool(jnp.isfinite(uT_phys).all())
    assert float(jnp.linalg.norm(uT_phys - u0_phys)) > 1e-8

    assert integrator.has_nonlinear
    assert integrator.linear_operator is not None


def test_etdrk4_kdv_soliton_tracks_exact_short_time() -> None:
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
    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)

    warning_messages = [str(w.message) for w in caught]
    assert not any(
        "scatter inputs have incompatible types" in msg for msg in warning_messages
    )
    assert not any(
        "Casting complex values to real discards the imaginary part" in msg
        for msg in warning_messages
    )

    xj = V.mesh()
    u_num = V.backward(uhat_t).real
    u_exact = (
        3
        * wave_speed
        * jnp.cosh(0.5 * jnp.sqrt(wave_speed) / mu_val * (xj - wave_speed * T - x0))
        ** -2
    )
    rel_error = jnp.linalg.norm(u_num - u_exact) / jnp.linalg.norm(u_exact)
    assert float(rel_error) < 5e-3


def test_etdrk4_kdv_nonlinear_jaxpr_uses_fft_path() -> None:
    N = 64
    mu = Constant("mu", sp.Rational(11, 500))
    V = FourierSpace(N, Domain(-1, 1))
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    weak_form = v * (u.diff(t) + (u * u).diff(x) / 2 + mu**2 * u.diff(x, 3))
    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, 0.01),
        initial=sp.cos(sp.pi * x),
        sparse=True,
        sparse_tol=1000,
    )
    uhat0 = integrator.initial_coefficients()
    jaxpr = jax.make_jaxpr(integrator._N)(uhat0).jaxpr

    # Fast Fourier path should stay spectral (FFT), not dense basis matmuls.
    assert _count_primitive(jaxpr, "fft") >= 3
    assert _count_primitive(jaxpr, "dot_general") == 0


def test_etdrk4_zk_step_jaxpr_uses_diagonal_fft_path() -> None:
    N = 24
    T = 0.01
    steps = 24
    dt = T / steps
    mu = Constant("mu", sp.Rational(3, 20))

    F = FourierSpace(N, Domain(-1, 1))
    V = TensorProductSpace((F, F), name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    x, _y = V.system.base_scalars()
    t = V.system.base_time()

    weak_form = v * (u.diff(t) + (u * u).diff(x) / 2 + mu**2 * Div(Grad(u)).diff(x))
    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, T),
        initial=sp.cos(sp.pi * x),
        sparse=True,
        sparse_tol=1000,
    )
    integrator.setup(dt=dt)

    assert bool(integrator.is_diag)
    assert integrator.mass_diag is not None
    assert integrator.linear_diag is not None

    uhat0 = integrator.initial_coefficients()
    jaxpr = jax.make_jaxpr(integrator.step)(uhat0, dt).jaxpr

    # The 2D ZK timestep should stay on the diagonal ETD + FFT pseudospectral path.
    assert _count_primitive(jaxpr, "fft") >= 24
    assert _count_primitive(jaxpr, "dot_general") == 0
