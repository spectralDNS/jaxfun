# Solve Poisson's equation
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sympy as sp
from scipy import sparse as scipy_sparse
import jax.numpy as jnp
from jaxfun.utils.common import lambdify, ulp
from jaxfun.Legendre import Legendre
from jaxfun.Chebyshev import Chebyshev
from jaxfun.composite import Composite
from jaxfun.inner import inner
from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.operators import Grad, Div, Dot
from jaxfun.Basespace import n
from jaxfun.tensorproductspace import TensorProductSpace, tpmats_to_scipy_sparse_list
from jaxfun.coordinates import get_CoordSys


M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
r, theta = sp.symbols('r,theta', real=True, positive=True)
C = get_CoordSys('C', sp.Lambda((r, theta), (r*sp.cos(theta), r*sp.sin(theta))))
D0 = Composite(Legendre, M, bcs, scaling=n + 1, domain=(sp.S.Half, 1), name="D", fun_str="psi")
D1 = Composite(Legendre, M, bcs, scaling=n + 1, domain=(0, sp.pi/2), name="D", fun_str="psi")
T = TensorProductSpace((D0, D1), coordinates=C, name="T")
v = TestFunction(T)
u = TrialFunction(T)

# Method of manufactured solution
rb, tb = T.system.base_scalars()
ue = (1-rb)*(sp.S.Half-rb) * tb * (sp.pi/2 - tb) #* sp.exp(sp.cos(sp.pi * x)) * sp.exp(sp.sin(sp.pi * y))

# A, b = inner(-Dot(Grad(u), Grad(v)) + v * Div(Grad(ue)), sparse=False)
A, b = inner((v * Div(Grad(u)) + v * Div(Grad(ue)))*rb, sparse=False)

# jax can only do kron for dense matrices
H = jnp.kron(*A[0].mats) + jnp.kron(*A[1].mats) + jnp.kron(*A[2].mats)
uh = jnp.linalg.solve(H, b.flatten()).reshape(b.shape)

# Alternative scipy sparse implementation
a = tpmats_to_scipy_sparse_list(A)
A0 = scipy_sparse.kron(a[0], a[1]) + scipy_sparse.kron(a[2], a[3]) + scipy_sparse.kron(a[4], a[5])
un = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

assert jnp.linalg.norm(uh-un) < 10*ulp(1)

rj, tj = T.mesh(kind='uniform', N=100)
xc, yc = T.cartesian_mesh(kind='uniform', N=100)
uj = T.evaluate(uh, kind='uniform', N=100)
uej = lambdify((rb, tb), ue)(rj, tj)

error = jnp.linalg.norm(uj - uej)
if 'pytest' in os.environ:
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
