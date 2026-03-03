import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ETDRK4
from jaxfun.operators import Constant
from jaxfun.utils.common import lambdify


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
