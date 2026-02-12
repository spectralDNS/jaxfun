import jax.numpy as jnp
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
