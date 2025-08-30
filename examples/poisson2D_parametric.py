# Solve Poisson's equation in 2D in a generic quadrilateral
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

# Some quadrilateral
nodes = ((0, 0), (4, 0), (3, 2), (0, 3))
#nodes = ((0, 0), (1, 0), (2, 1), (0, 1))
#x0, x1, x2, x3  = sp.symbols('x:4', real=True)
#y0, y1, y2, y3  = sp.symbols('y:4', real=True)
#nodes =((x0, y0), (x1, y1), (x2, y2), (x3, y3))

xi, eta = sp.symbols("xi,eta", real=True)
phi = (
    (1 - xi) * (1 - eta) / 4,
    (1 + xi) * (1 - eta) / 4,
    (1 + xi) * (1 + eta) / 4,
    (1 - xi) * (1 + eta) / 4
)
C = get_CoordSys(
    "C",
sp.Lambda(
    (xi, eta),
    (
        sum([phi[i] * nodes[i][0] for i in range(4)]),
        sum([phi[i] * nodes[i][1] for i in range(4)]),
    ),
),
    assumptions=sp.Q.positive(xi+1)&sp.Q.positive(eta+1)&sp.Q.positive(1-eta)&sp.Q.positive(1-xi),
    #replace=((x0, 0), (y0, 0), (x1, 4), (y1, 0), (x2, 3), (y2, 2), (x3, 0), (y3, 3))
)

M = 20
ue = (1 + xi**2) * (1 + eta**2)
bcsx = {"left": {"D": ue.subs(xi, -1)}, "right": {"D": ue.subs(xi, 1)}}
bcsy = {"left": {"D": ue.subs(eta, -1)}, "right": {"D": ue.subs(eta, 1)}}
D0 = FunctionSpace(M, Chebyshev, bcsx, name="D0", fun_str="phi")
D1 = FunctionSpace(M, Chebyshev, bcsy, name="D1", fun_str="psi")
T = TensorProduct(D0, D1, system=C, name="T")

v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
xi, eta = T.system.base_scalars()
ue = T.system.expr_psi_to_base_scalar(ue)

# A, b = inner(-Dot(Grad(u), Grad(v)) - v * Div(Grad(ue)), sparse=False)
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=False)

H = jnp.sum(jnp.array([a.mat for a in A]), axis=0)
uh = jnp.linalg.solve(H, b.flatten()).reshape(b.shape)

N = 100
xi, eta = C.base_scalars()
xij, etaj = T.mesh(kind="uniform", N=(N, N))
xc, yc = T.cartesian_mesh(kind="uniform", N=(N, N))
uj = T.backward(uh, kind="uniform", N=(N, N))
uej = lambdify((xi, eta), ue)(xij, etaj)

error = jnp.linalg.norm(uj - uej) / N
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