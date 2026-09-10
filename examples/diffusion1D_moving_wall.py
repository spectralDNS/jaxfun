# Solve the 1D diffusion equation with a wall value that varies in time
#
#   u_t = nu * u_xx,  x in [-1, 1]
#   u(-1,t) = g(-1,t),  u(1,t) = g(1,t)
#
# The exact solution g = exp(-4*nu*t) * sin(2x) solves the equation, so no
# source term is needed -- but it is nonzero at both ends and decays, so the
# boundary data moves and the lifting has to move with it.
#
# An inhomogeneous space splits the solution into a free part and a boundary
# lifting B, u = u_h + B. Under the time derivative that contributes
# `-<v, dB/dt>`, which vanishes only when B is steady; here it is the whole
# boundary effect, because a two-point Dirichlet lifting is linear in x and so
# has no Laplacian for the stiffness term to see.
#
# Spatial discretization: Legendre Galerkin with time-dependent Dirichlet BCs
# Time discretization: IMEX Runge-Kutta
# ruff: noqa: E402, I001

import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.coordinates import R
from jaxfun.galerkin.Legendre import Legendre as space
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ARK4_3_6L2SA, IMEXRungeKutta
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify, ulp

M = 32
nu = Constant("nu", 0.5)
T = 1.0
steps = 100
dt = T / steps

C = R(1)
x, t = C.x, C.base_time()
ue = sp.exp(-4 * nu.val * t) * sp.sin(2 * x)

bcs = {"left": {"D": ue.subs(x, -1)}, "right": {"D": ue.subs(x, 1)}}
V = FunctionSpace(M, space, bcs=bcs, system=C, name="V", fun_str="psi")
# Nonzero boundary values, so the space is a direct sum carrying a lifting --
# which is what `evaluate(..., t=)` below needs.
assert isinstance(V, DirectSum)
v = TestFunction(V, name="v")
u = TrialFunction(V, name="u", transient=True)

weak_form = v * (u.diff(t) - nu * Div(Grad(u)))

integrator = IMEXRungeKutta(
    weak_form,
    tableau=ARK4_3_6L2SA,
    time=(0.0, T),
    initial=sp.sin(2 * x),
    sparse=True,
    sparse_tol=1000,
)

uhat_T = integrator.solve(dt=dt, steps=steps, progress="PYTEST" not in os.environ)

# The state holds the homogeneous coefficients, so reconstructing it needs the
# lifting at the time the run finished -- not the one the space was built with.
t_end = integrator.end_time(dt, steps)

xj = V.mesh(kind="uniform", N=200)
u_num = V.evaluate(xj, uhat_T, t=t_end).real
u_ex_j = lambdify(x, ue.subs(t, T))(xj)

error = jnp.linalg.norm(u_num - u_ex_j) / jnp.linalg.norm(u_ex_j)
if "PYTEST" in os.environ:
    assert error < jnp.sqrt(ulp(1)), error
    sys.exit(0)

print("Relative L2 error =", float(error))
plt.plot(xj, lambdify(x, sp.sin(2 * x))(xj), "--k", label="initial")
plt.plot(xj, u_ex_j, "r", label=f"exact, t={T}")
plt.plot(xj, u_num, "b:", label="IMEX RK4")
plt.legend()
plt.xlabel("x")
plt.title("Diffusion with time-dependent Dirichlet data")
plt.show()
