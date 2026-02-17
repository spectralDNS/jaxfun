import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as Cheb
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import RK4, BackwardEuler
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify, n


def test_backward_euler_linear_advection_fourier() -> None:
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

    integrator = BackwardEuler(
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
    assert float(rel_error) < 3e-2


def test_rk4_solve_uses_initial_and_time_api() -> None:
    M = 16
    nu = Constant("nu", 0.1)
    T = 0.1
    steps = 200
    dt = T / steps

    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    V = FunctionSpace(M, Cheb, bcs=bcs, name="V", fun_str="psi", scaling=n + 1)

    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(sp.pi * (x + 1) / 2)
    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))

    integrator = RK4(
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
    k = sp.pi / 2
    u_ex = lambdify(x, sp.sin(k * (x + 1)) * sp.exp(-nu.val * float(k**2) * T))(xj)
    error = jnp.linalg.norm(u_num - u_ex)
    assert float(error) < 2e-1


def test_solve_n_batches_and_return_each_step() -> None:
    N = 16
    dt = 0.01
    steps = 23

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(x)
    weak_form = v * (u.diff(t) + u.diff(x))
    integrator = RK4(V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True)

    states = integrator.solve(
        dt=dt,
        steps=steps,
        n_batches=5,
        return_each_step=True,
        progress=False,
    )
    integrator_final = RK4(
        V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True
    )
    final = integrator_final.solve(dt=dt, steps=steps, n_batches=5, progress=False)

    # 23 steps in 5 batches => 1 initial + 5 snapshots + 1 remainder snapshot.
    assert states.shape == (7, V.num_dofs)
    assert jnp.allclose(states[-1], final)

    with pytest.raises(ValueError, match="n_batches must be a positive integer"):
        _ = integrator.solve(dt=dt, steps=steps, n_batches=0, progress=False)
