import jax.numpy as jnp
import sympy as sp

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
