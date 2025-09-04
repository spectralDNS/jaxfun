# Solve Poisson's equation in polar coordinates on parts of an annulus
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse as scipy_sparse

from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_scipy_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
r, theta = sp.symbols("r,theta", real=True, positive=True)
C = get_CoordSys("C", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))
D0 = FunctionSpace(
    M, Legendre, bcs, scaling=n + 1, domain=(sp.S.Half, 1), name="D0", fun_str="psi"
)
D1 = FunctionSpace(
    M, Legendre, bcs, scaling=n + 1, domain=(0, sp.pi / 2), name="D1", fun_str="phi"
)
T = TensorProduct(D0, D1, system=C, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
r, theta = C.base_scalars()
ue = (1 - r) * (sp.S.Half - r) * theta * (sp.pi / 2 - theta)

# Assemble linear system of equations
# A, b = inner(-Dot(Grad(u), Grad(v)) + v * Div(Grad(ue)), sparse=False)
A, b = inner((v * Div(Grad(u)) - v * Div(Grad(ue))), sparse=False)

# jax can only do kron for dense matrices
H = A[0].mat + A[1].mat + A[2].mat
uh = jnp.linalg.solve(H, b.flatten()).reshape(b.shape)

# Alternative scipy sparse implementation
A0 = tpmats_to_scipy_kron(A)
un = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

assert jnp.linalg.norm(uh - un) < ulp(1000)

rj, tj = T.mesh(kind="uniform", N=(100, 100))
xc, yc = T.cartesian_mesh(kind="uniform", N=(100, 100))
uj = T.backward(uh, kind="uniform", N=(100, 100))
uej = lambdify((r, theta), ue)(rj, tj)

error = jnp.linalg.norm(uj - uej)
if "PYTEST" in os.environ:
    assert error < ulp(1000)
    sys.exit(1)

print("Error =", error)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
ax1.contourf(xc, yc, uj)
ax2.contourf(xc, yc, uej)
ax2.set_autoscalex_on(False)
c3 = ax3.contourf(xc, yc, uej - uj)
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
