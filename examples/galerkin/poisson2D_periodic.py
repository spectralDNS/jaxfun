# Solve Poisson's equation in 2D with periodic boundary conditions in x and
# Dirichlet boundary conditions in y.
#
# Note that this solver may be run in parallel using SPMD, by setting the environment
# variable `JAX_NUM_CPU_DEVICES` to the number of devices to use. Or specify the number
# of devices in the code with `jax.config.update("jax_num_cpu_devices", 2)` before any
# imports from jaxfun.
#
# The matrix A is using the TPMatricesWavenumberSolver, which is a *communication-free*
# parallel solver for tensor product matrices with wavenumber decomposition. That is,
# with a Fourier basis along the first (and even second for 3D) axis. The solver works
# by sharding the LU-decomposition of the matrix and by running only over the local
# wavenumbers on each device.
#
# In order to make the solver work also on distributed devices, one also needs to
# initialize a distributed JAX environment. See, e.g., the `RayleighBenard.py` example.


import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from jaxfun.coordinates import R
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

R2 = R(2)
x, y = R2.base_scalars()

ue = (1 - y**2) * (sp.cos(2 * x)) * sp.exp(sp.cos(sp.pi * y))

M, N = 80, 20
bcs = {"left": {"D": ue.subs(y, -1)}, "right": {"D": ue.subs(y, 1)}}
D = FunctionSpace(M, Legendre, bcs, scaling=n + 1, name="D", fun_str="psi")
F = FunctionSpace(N, Fourier, name="F", fun_str="E")
T = TensorProduct(F, D, name="T", real=True)
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# A, b = inner(-Dot(Grad(u), Grad(v)) - v * Div(Grad(ue)), sparse=True)
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True, kind="system")

uh = A.solve(b)

N = 100
uj = T.backward(uh, N=(2 * N, 2 * M))
xj = T.mesh(N=(2 * N, 2 * M), broadcast=True)
uej = lambdify((x, y), ue)(*xj)

error = jnp.linalg.norm(uj - uej) / N
if "PYTEST" in os.environ:
    assert error < ulp(100), error
    sys.exit(0)

print("Error =", error)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
xj = T.mesh(N=(2 * N, 2 * M), broadcast=False)
ax1.contourf(xj[1], xj[0], uj.real)
ax2.contourf(xj[1], xj[0], uej.real)
ax2.set_autoscalex_on(False)
c3 = ax3.contourf(xj[1], xj[0], (uej - uj).real)
axins = inset_axes(
    ax3,
    width="5%",  # width = 10% of parent_bbox width
    height="100%",  # height : 50%
    loc=6,
    bbox_to_anchor=(1.05, 0.0, 1, 1),
    bbox_transform=ax3.transAxes,
    borderpad=0,
)
cbar = plt.colorbar(c3, cax=axins)
ax1.set_title("Jaxfun")
ax2.set_title("Exact")
ax3.set_title("Error")
plt.show()
