# Solve Helmholtz' equation
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse as scipy_sparse

from jaxfun.coordinates import x, y
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as space
from jaxfun.galerkin.functionspace import FunctionSpace

# from jaxfun.Jacobi import Jacobi as space
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_scipy_kron

# from jaxfun.galerkin.Legendre import Legendre as space
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

M = 50
ue = sp.exp(-(x**2 + y**2))

bcsx = {"left": {"D": ue.subs(x, 0)}, "right": {"D": ue.subs(x, 1)}}
bcsy = {"left": {"D": ue.subs(y, 0)}, "right": {"D": ue.subs(y, 1)}}
Dx = FunctionSpace(
    M, space, bcs=bcsx, name="Dx", fun_str="psi", scaling=n + 1, domain=(0, 1)
)
Dy = FunctionSpace(
    M, space, bcs=bcsy, name="Dy", fun_str="phi", scaling=n + 1, domain=(0, 1)
)
T = TensorProduct(Dx, Dy, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
ue = T.system.expr_psi_to_base_scalar(ue)

A, L = inner(
    v * (Div(Grad(u)) + u) - v * (Div(Grad(ue)) + ue), sparse=True, sparse_tol=1000
)

A0 = tpmats_to_scipy_kron(A)
un = jnp.array(scipy_sparse.linalg.spsolve(A0, L.flatten()).reshape(L.shape))

N = 100
uj = T.backward(un, kind="uniform", N=(N, N))
xj = T.mesh(kind="uniform", N=(N, N))
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
