import jax.numpy as jnp
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ETDRK4


def test_etdrk4_focusing_nls_soliton_tracks_exact_short_time() -> None:
    N = 32
    L = 20.0
    omega = 1.0
    T = 0.01
    steps = 16
    dt = T / steps

    V = FunctionSpace(
        N,
        FourierSpace,
        domain=Domain(-L, L),
        name="V",
        fun_str="E",
    )
    v = TestFunction(V, name="v")
    psi = TrialFunction(V, name="psi", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    psi0 = sp.sqrt(2 * omega) * sp.sech(sp.sqrt(omega) * x)
    weak_form = v * (
        psi.diff(t) - sp.I * psi.diff(x, 2) - sp.I * sp.Abs(psi) ** 2 * psi
    )

    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, T),
        initial=psi0,
        sparse=True,
        sparse_tol=1000,
    )
    psi_hat_t = integrator.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    psi_init = V.backward(integrator.initial_coefficients())
    psi_num = V.backward(psi_hat_t)
    psi_exact = (
        jnp.sqrt(2 * omega) / jnp.cosh(jnp.sqrt(omega) * xj) * jnp.exp(1j * omega * T)
    )

    rel_error = jnp.linalg.norm(psi_num - psi_exact) / jnp.linalg.norm(psi_exact)
    mass_init = jnp.trapezoid(jnp.abs(psi_init) ** 2, xj)
    mass_final = jnp.trapezoid(jnp.abs(psi_num) ** 2, xj)

    assert integrator.has_nonlinear
    # rel error doesn't scale with precision
    assert float(rel_error) < 0.002
    # abs error is 0 with float32, around 1e-9 with float64
    # Although, it is apparently 1e-6 when run in CI with float32...
    assert float(jnp.abs(mass_final - mass_init)) < 1e-6
