# Solve Poisson's equation in 2D
# ruff: noqa: E402
import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.cartesianproductspace import CartesianProduct
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.la import BlockArray, BlockMatrix
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import lambdify, ulp

M = 28
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, Legendre, bcs, name="D", fun_str="psi")
T = TensorProduct(D, D, name="T")
V = CartesianProduct(T, T, rank=1, name="V")

v = TestFunction(V, name="v")
u = TrialFunction(V, name="u")

# Method of manufactured solution
x, y = T.system.base_scalars()
i, j = T.system.base_vectors()
ue = (1 - x**2) * (1 - y**2) * i + (1 - x**2) * (1 - y**2) * sp.sin(2 * sp.pi * x) * j

A = inner(-Grad(u), Grad(v), sparse=False)
b = inner(Div(Grad(ue)), v)

assert isinstance(A, BlockMatrix)
assert isinstance(b, BlockArray)
uh = A.solve(b, method="lu")

N = 100
uj = V.evaluate_mesh(uh.array, kind="uniform", N=(N, N))
xj = T.mesh(kind="uniform", N=(N, N), broadcast=True)
uej = jnp.stack(
    [
        lambdify((x, y), Dot(ue, i).doit())(*xj),
        lambdify((x, y), Dot(ue, j).doit())(*xj),
    ]
)

error = jnp.linalg.norm(uj - uej) / (2 * N**2)
if "PYTEST" in os.environ:
    assert error < ulp(1000), error
    sys.exit(0)

print("Error =", error)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
xj = T.mesh(kind="uniform", N=(N, N), broadcast=False)
ax1.contourf(xj[0], xj[1], uj[1])
ax2.contourf(xj[0], xj[1], uej[1])
ax2.set_autoscalex_on(False)
c3 = ax3.contourf(xj[0], xj[1], uej[1] - uj[1])
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
