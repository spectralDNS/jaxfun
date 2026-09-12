# Solve Poisson's equation in spherical coordinates
import os
import sys

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import sympy as sp

from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

N, M = 50, 50

# Define spherical coordinates
r = 1
theta, phi = sp.symbols("theta, phi", real=True, positive=True)

C = get_CoordSys(
    "C",
    sp.Lambda(
        (phi, theta),
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
L = FunctionSpace(N, Legendre, domain=(0, np.pi), name="L", fun_str="theta")
F = FunctionSpace(M, Fourier, name="F", fun_str="phi")
T = TensorProduct(
    F, L, system=C, name="T"
)  # Fourier first for efficient wavenumber solver
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
phi, theta = C.base_scalars()
sph = sp.functions.special.spherical_harmonics.Ynm  # ty:ignore[possibly-missing-submodule]
ue = sph(6, 3, theta, phi)

# Assemble linear system of equations
A, b = inner(
    (v * (2 * u - Div(Grad(u))) - v * (2 * ue - Div(Grad(ue)))),
    sparse=True,
    kind="system",
)

un = A.solve(b)

tj, rj = T.mesh(N=(100, 100))
xc, yc, zc = T.cartesian_mesh(N=(100, 100))
uj = T.backward(un, N=(100, 100))
uej = lambdify((phi, theta), ue)(tj, rj)

error = jnp.linalg.norm(uj - uej) / 100
if "PYTEST" in os.environ:
    assert error < ulp(1000), error
    sys.exit(0)

print("Error =", error)

zc = jnp.broadcast_to(zc, xc.shape)
s = go.Surface(x=xc, y=yc, z=zc, surfacecolor=uj.real)
fig = go.Figure(s)
d = {"visible": False, "showgrid": False, "zeroline": False}
fig.update_layout(scene={"xaxis": d, "yaxis": d, "zaxis": d})
fig.show()
