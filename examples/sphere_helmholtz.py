# Solve Poisson's equation in polar coordinates on parts of an annulus
import os
import sys

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy import sparse as scipy_sparse

from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_scipy_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

N, M = 50, 50

# Define parabolic coordinates
r = 1
theta, phi = sp.symbols("theta, phi", real=True, positive=True)

C = get_CoordSys(
    "C",
    sp.Lambda(
        (theta, phi),
        (
            r * sp.sin(theta) * sp.cos(phi),
            r * sp.sin(theta) * sp.sin(phi),
            r * sp.cos(theta),
        ),
    ),
    assumptions=sp.Q.positive(theta)
    & sp.Q.positive(phi)
    & sp.Q.positive(sp.sin(theta)),
)
D0 = FunctionSpace(N, Chebyshev, domain=(0, np.pi), name="D0", fun_str="theta")
D1 = FunctionSpace(M, Fourier, name="D1", fun_str="phi")
T = TensorProduct(D0, D1, system=C, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
sph = sp.functions.special.spherical_harmonics.Ynm
ue = sph(6, 3, theta, phi)
ue = C.expr_psi_to_base_scalar(ue)

# Assemble linear system of equations
A, b = inner(
    (v * (2 * u - Div(Grad(u))) - v * (2 * ue - Div(Grad(ue)))),
    sparse=True,
    sparse_tol=1000,
)

# Alternative scipy sparse implementation
A0 = tpmats_to_scipy_kron(A)
un = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

theta, phi = C.base_scalars()
rj, tj = T.mesh(N=(100, 00))
xc, yc, zc = T.cartesian_mesh(N=(100, 00))
uj = T.backward(un, N=(100, 00))
uej = lambdify((theta, phi), ue)(rj, tj)

error = jnp.linalg.norm(uj - uej) / N
if "PYTEST" in os.environ:
    assert error < ulp(1000)
    sys.exit(1)

print("Error =", error)

zc = jnp.broadcast_to(zc, xc.shape)
s = go.Surface(x=xc, y=yc, z=zc, surfacecolor=uj.real)
fig = go.Figure(s)
d = {"visible": False, "showgrid": False, "zeroline": False}
fig.update_layout(scene={"xaxis": d, "yaxis": d, "zaxis": d})
fig.show()
