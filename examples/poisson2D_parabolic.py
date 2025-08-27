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

M = 40
bcs = {"left": {"D": 0}, "right": {"D": 0}}

# Define parabolic coordinates
tau, sigma = sp.symbols("tau, sigma", real=True)

C = get_CoordSys(
    "C",
    sp.Lambda((tau, sigma), (tau * sigma, (tau**2 - sigma**2) / 2)),
    assumptions=sp.Q.positive(tau)&sp.Q.positive(sigma+1)
)
D0 = FunctionSpace(
    M, Legendre, bcs, scaling=n + 1, domain=(0, 1), name="D0", fun_str="tau"
)
D1 = FunctionSpace(M, Legendre, bcs, scaling=n + 1, name="D1", fun_str="sigma")
T = TensorProduct((D0, D1), system=C, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
ue = (tau * (1 - tau)) ** 2 * (1 - sigma**2) ** 1 * sp.sin(4 * sp.pi * sigma)
ue = C.expr_psi_to_base_scalar(ue)

# Assemble linear system of equations
A, b = inner((v * Div(Grad(u)) - v * Div(Grad(ue)))*C.sg, sparse=False)

# Alternative scipy sparse implementation
A0 = tpmats_to_scipy_kron(A)
un = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

N = 100
tau, sigma = C.base_scalars()
rj, tj = T.mesh(kind="uniform", N=(N, N))
xc, yc = T.cartesian_mesh(kind="uniform", N=(N, N))
uj = T.backward(un, kind="uniform", N=(N, N))
uej = lambdify((tau, sigma), ue)(rj, tj)

error = jnp.linalg.norm(uj - uej) / N
if "pytest" in os.environ:
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
