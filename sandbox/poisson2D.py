# Solve Poisson's equation in 2D
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct, TPMatrices, tpmats_to_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

M = 16
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, Legendre, bcs)
F = FunctionSpace(M, Fourier)
T = TensorProduct(F, D, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
x, y = T.system.base_scalars()
ue = sp.cos(2 * x) * (1 - y**2)  # * sp.exp(sp.cos(sp.pi * x)) * sp.exp(sp.sin(sp.pi * y))

# A, b = inner(-Dot(Grad(u), Grad(v)) - v * Div(Grad(ue)), sparse=False)
A, b = inner(v * Div(Grad(u)) +u*v - v * Div(Grad(ue)) - ue*v, sparse=True)

#C = tpmats_to_kron(A)
#uh = C.solve(b.flatten()).reshape(b.shape)
C = TPMatrices(A)
uh = C.solve(b)

N = 100
uj = T.backward(uh, N=(N, N))
xj = T.mesh(N=(N, N), broadcast=True)
uej = lambdify((x, y), ue)(*xj)

error = jnp.linalg.norm(uj - uej) / N
if "PYTEST" in os.environ:
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
