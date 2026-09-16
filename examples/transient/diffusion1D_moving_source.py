# Solve the 1D diffusion equation driven by a source that varies in time
#
#   u_t = nu * u_xx + f(x,t),  x in [-1, 1]
#   u(-1,t) = u(1,t) = 0
#
# The walls are held at zero, so nothing about the boundary moves -- everything
# time-dependent here is the source. `f` is manufactured from a chosen exact
# solution, a standing mode driven at a frequency of its own:
#
#   ue = sin(pi x) * (1 + a*sin(omega*t))
#
# which vanishes at both ends for every t. The source follows from
# f = ue_t - nu*ue_xx.
#
# A source is supported when its time dependence factors out of the quadrature,
# `f = sum_k g_k(t) * h_k(x)`. Each `h_k` is then assembled once into a load
# vector and a stage costs one scalar evaluation and one axpy -- there is no
# re-integration per step. A source written so that time rides inside a
# coordinate factor, sqrt(x + t) say, is refused rather than mishandled.
#
# The forcing is evaluated at every Runge-Kutta stage time, which is what keeps
# the scheme at its design order: held at the step's own t it would drop to
# first order however good the tableau.
#
# Spatial discretization: Legendre Galerkin with homogeneous Dirichlet BCs
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
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ARK4_3_6L2SA, IMEXRungeKutta
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify

M = 32
nu = Constant("nu", 0.1)
T = 2.0
steps = 200
dt = T / steps

C = R(1)
x, t = C.x, C.base_time()
# Driven at omega, which the diffusion alone would never produce -- so a frozen
# source is visible in the answer rather than a small correction to it.
omega = 4.0
ue = sp.sin(sp.pi * x) * (1 + sp.Rational(1, 2) * sp.sin(omega * t))
f = sp.diff(ue, t) - nu * Div(Grad(ue))

bcs = {"left": {"D": 0}, "right": {"D": 0}}
V = FunctionSpace(M, space, bcs=bcs, system=C, name="V", fun_str="psi")
v = TestFunction(V, name="v")
u = TrialFunction(V, name="u", transient=True)

weak_form = v * (u.diff(t) - nu * Div(Grad(u)) - f)

integrator = IMEXRungeKutta(
    weak_form,
    tableau=ARK4_3_6L2SA,
    time=(0.0, T),
    initial=ue.subs(t, 0),
    sparse=True,
)
# The source moves; the walls do not.
assert integrator._transient_source  # noqa: SLF001
assert not integrator._transient_boundary  # noqa: SLF001

uhat_T = integrator.solve(dt=dt, steps=steps, progress="PYTEST" not in os.environ)

xj = V.mesh(kind="uniform", N=200)
u_num = V.evaluate(xj, uhat_T).real
u_ex_j = lambdify(x, ue.subs(t, T))(xj)

error = jnp.linalg.norm(u_num - u_ex_j) / jnp.linalg.norm(u_ex_j)
if "PYTEST" in os.environ:
    assert error < 1e-5, error
    sys.exit(0)

print("Relative L2 error =", float(error))
plt.plot(xj, lambdify(x, ue.subs(t, 0))(xj), "--k", label="initial")
plt.plot(xj, u_ex_j, "r", label=f"exact, t={T}")
plt.plot(xj, u_num, "b:", label="IMEX ARK4_3_6L2SA")
plt.legend()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Diffusion driven by a time-dependent source")
plt.show()
