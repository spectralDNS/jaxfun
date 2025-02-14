# Solve Poisson's equation in 2D
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse as scipy_sparse

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.Basespace import n
from jaxfun.Chebyshev import Chebyshev
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.operators import Div, Grad
from jaxfun.tensorproductspace import TensorProduct, tpmats_to_scipy_kron
from jaxfun.utils.common import lambdify, ulp

M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, Chebyshev, bcs, scaling=n + 1, name="D", fun_str="psi")
T = TensorProduct((D, D), name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
x, y = T.system.base_scalars()
ue = (1 - x**2) * (1 - y**2)  # * sp.exp(sp.cos(sp.pi * x)) * sp.exp(sp.sin(sp.pi * y))

# A, b = inner(-Dot(Grad(u), Grad(v)) - v * Div(Grad(ue)), sparse=False)
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)

# jax can only do kron for dense matrices
#C = jnp.kron(*A[0].mats) + jnp.kron(*A[1].mats)
#uh = jnp.linalg.solve(C, b.flatten()).reshape(b.shape)

# Alternative scipy sparse implementation
A0 = tpmats_to_scipy_kron(A)
uh = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

N = 100
uj = T.backward(uh, kind="uniform", N=(N, N))
xj = T.mesh(kind="uniform", N=(N, N), broadcast=True)
uej = lambdify((x, y), ue)(*xj)

error = jnp.linalg.norm(uj - uej) / N
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
xj = T.mesh(kind="uniform", N=(N, N), broadcast=False)
ax1.contourf(xj[0], xj[1], uj)
ax2.contourf(xj[0], xj[1], uej)
ax2.set_autoscalex_on(False)
c3 = ax3.contourf(xj[0], xj[1], uej - uj)
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
