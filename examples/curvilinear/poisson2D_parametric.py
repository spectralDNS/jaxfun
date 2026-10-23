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

xi, eta = sp.symbols("xi,eta", real=True)


def bilinear_map(nodes, xi, eta):
    """Map the reference square [-1, 1]^2 onto the quadrilateral with given nodes."""
    nodes = [tuple(sp.nsimplify(c, rational=True) for c in p) for p in nodes]
    phi = (
        (1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4,
    )
    return sp.Lambda(
        (xi, eta), tuple(sum(phi[i] * nodes[i][k] for i in range(4)) for k in range(2))
    )


# Some quadrilateral
nodes = ((0, 0), (4, 0), (3, 2), (0, 3))

C = get_CoordSys(
    "C",
    bilinear_map(nodes, xi, eta),
    assumptions=sp.Q.positive(xi + 1)
    & sp.Q.positive(eta + 1)
    & sp.Q.positive(1 - eta)
    & sp.Q.positive(1 - xi),
)

# Method of manufactured solution
xi, eta = C.base_scalars()
ue = (1 + xi**2) * (1 + eta**2)

M = 20
bcsx = {"left": {"D": ue.subs(xi, -1)}, "right": {"D": ue.subs(xi, 1)}}
bcsy = {"left": {"D": ue.subs(eta, -1)}, "right": {"D": ue.subs(eta, 1)}}
D0 = FunctionSpace(M, Chebyshev, bcsx, name="D0", fun_str="phi")
D1 = FunctionSpace(M, Chebyshev, bcsy, name="D1", fun_str="psi")
T = TensorProduct(D0, D1, system=C, name="T")
P = T.get_testspace("GR")

v = TestFunction(P, name="v")
u = TrialFunction(T, name="u")

A, b = inner(
    (v * Div(Grad(u)) - v * Div(Grad(ue))) * C.sg**2, sparse=True, kind="system"
)

uh = A.solve(b, method="kron", kron_method="rcm")

N = 100
xij, etaj = T.mesh(kind="uniform", N=(N, N))
xc, yc = T.cartesian_mesh(kind="uniform", N=(N, N))
uj = T.evaluate_mesh(uh, kind="uniform", N=(N, N))
uej = lambdify((xi, eta), ue)(xij, etaj)

error = jnp.linalg.norm(uj - uej) / N
if "PYTEST" in os.environ:
    assert error < ulp(1000)
    sys.exit(0)

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
