# Solve the 1D linear advection equation with Backward Euler in time
#
#   u_t + c u_x = 0,  x in [0, 2π] periodic
#
# Spatial discretization: Fourier Galerkin (spectral)
# Time discretization: Backward Euler (implicit)
# ruff: noqa: E402
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier as space
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import BackwardEuler
from jaxfun.operators import Constant
from jaxfun.utils.common import lambdify

N = 64

c = Constant("c", 1.0)
T = 5
steps = 20000
dt = T / steps

V = FunctionSpace(N, space, name="V", fun_str="E")
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
uhat_T = integrator.solve(dt=dt, steps=steps)

xj = V.mesh()
u_num = V.backward(uhat_T).real
u_ex = sp.sin(x - c.val * T)
u_ex_j = lambdify(x, u_ex)(xj)

rel_error = jnp.linalg.norm(u_num - u_ex_j) / jnp.linalg.norm(u_ex_j)
if "PYTEST" in os.environ:
    # Backward Euler is diffusive for pure advection; keep tolerance loose.
    assert rel_error < 0.35, rel_error
    sys.exit(1)

print("Relative L2 error =", float(rel_error))
plt.plot(xj, lambdify(x, u0)(xj), "--k", label="initial")
plt.plot(xj, u_ex_j, "r", label="exact")
plt.plot(xj, u_num, "b", label="backward euler")
plt.legend()
plt.show()
