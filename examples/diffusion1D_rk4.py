# Solve the 1D diffusion equation with RK4 in time
#
#   u_t = nu * u_xx,  x in [-1, 1]
#   u(-1,t) = u(1,t) = 0
#
# Spatial discretization: Chebyshev Galerkin with Dirichlet BCs
# Time discretization: RK4 (explicit)
# ruff: noqa: E402, I001

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.galerkin.Chebyshev import Chebyshev as space
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import RK4
from jaxfun.operators import Div, Grad, Constant
from jaxfun.utils.common import lambdify, n


M = 24
nu = Constant("nu", 0.1)
T = 1
steps = 2000
dt = T / steps

bcs = {"left": {"D": 0}, "right": {"D": 0}}
V = FunctionSpace(M, space, bcs=bcs, name="V", fun_str="psi", scaling=n + 1)
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

uhat_T = integrator.solve(dt=dt, steps=steps)

xj = V.mesh()
u_num = V.backward(uhat_T).real
k = sp.pi / 2
u_ex = sp.sin(k * (x + 1)) * sp.exp(-nu.val * float(k**2) * T)
u_ex_j = lambdify(x, u_ex)(xj)

error = jnp.linalg.norm(u_num - u_ex_j)
if "PYTEST" in os.environ:
    assert error < 5e-2, error
    sys.exit(1)

print("Relative L2 error =", float(error))
plt.plot(xj, lambdify(x, u0)(xj), "--k", label="initial")
plt.plot(xj, u_ex_j, "r", label="exact")
plt.plot(xj, u_num, "b", label="rk4")
plt.legend()
plt.show()
