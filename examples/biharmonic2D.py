# Solve biharmonic equation in 2D
import sys
import os

# import jax
# jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse as scipy_sparse
import sympy as sp
import jax.numpy as jnp
from jaxfun.utils.common import lambdify, ulp
from jaxfun.Legendre import Legendre
from jaxfun.Chebyshev import Chebyshev
from jaxfun.inner import inner
from jaxfun.arguments import TestFunction, TrialFunction, x, y
from jaxfun.operators import Grad, Div, Dot
from jaxfun.Basespace import n
from jaxfun.functionspace import FunctionSpace
from jaxfun.tensorproductspace import TensorProduct, tpmats_to_scipy_sparse_list


# Method of manufactured solution
#ue = sp.exp(sp.cos(2 * sp.pi * (x - sp.S.Half / 2))) * sp.exp(
#    sp.sin(2 * (y - sp.S.Half))
#)
ue = (x-x**2)**2*(x-y**2)**2
M = 20

bcsx = {
    "left": {"D": ue.subs(x, -1), "N": ue.diff(x, 1).subs(x, -1)},
    "right": {"D": ue.subs(x, 1), "N": ue.diff(x, 1).subs(x, 1)},
}
bcsy = {
    "left": {"D": ue.subs(y, -1), "N": ue.diff(y, 1).subs(y, -1)},
    "right": {"D": ue.subs(y, 1), "N": ue.diff(y, 1).subs(y, 1)},
}

Dx = FunctionSpace(M, Chebyshev, scaling=n+1, bcs=bcsx, name="Dx", fun_str="psi")
Dy = FunctionSpace(M, Chebyshev, scaling=n+1, bcs=bcsy, name="Dy", fun_str="phi")
T = TensorProduct((Dx, Dy), name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")
ue = T.system.expr_psi_to_base_scalar(ue)

A, b = inner(Div(Grad(Div(Grad(u))))*v - Div(Grad(Div(Grad(ue))))*v, sparse=False)

# jax can only do kron for dense matrices
C = (
    jnp.kron(*A[0].mats)
    + jnp.kron(*A[1].mats)
    + jnp.kron(*A[2].mats)
)
uh = jnp.linalg.solve(C, b.flatten()).reshape(b.shape)

## Alternative scipy sparse implementation
#a = tpmats_to_scipy_sparse_list(A, tol=1000)
#A0 = (
#    scipy_sparse.kron(a[0], a[1], 'csc')
#    + scipy_sparse.kron(a[2], a[3], 'csc')
#    + scipy_sparse.kron(a[4], a[5], 'csc')
#)
#un = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

N = 100
xj = T.mesh(kind="uniform", N=(N, N))
uj = T.backward(uh, kind="uniform", N=(N, N))
uej = lambdify((x, y), ue)(*xj)

error = jnp.linalg.norm(uj - uej) / N
if "pytest" in os.environ:
    assert error < ulp(C.max()), error
    sys.exit(1)

print("Error =", error)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
xj = T.mesh(kind="uniform", N=(100, 100), broadcast=False)
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
