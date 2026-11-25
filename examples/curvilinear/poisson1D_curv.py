r"""
Solve Poisson's equation on a curved line in 2D or 3D space.

Define a position vector, `rv(t)`, as::

    rv(t) = x(t)i + y(t)j + z(t)k,

where i, j, k are the Cartesian unit vectors and t is found
in some suitable interval, e.g., [0, 1] or [0, 2\pi]. Note that
the position vector describes a curved domain with only one
parameter, so we can use a 1D function space with a curvilinear
coordinate system.

Solve::

    -div(grad(u(t))) = f(t), for t in interval

using curvilinear coordinates.

"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre as space, LGComposite
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

t = sp.Symbol("t", real=True)
rv = (sp.sin(2 * sp.pi * t), sp.cos(2 * sp.pi * t), 2 * t)
C = get_CoordSys("C", sp.Lambda((t,), rv))

N = 28
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(N, space, bcs, system=C, name="D", fun_str="phi")
v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
t = C.t  # use BaseScalar from the coordinate system, not the original symbol
ue = sp.sin(4 * sp.pi * t)

A, b = inner(-v * (Div(Grad(u)) - Div(Grad(ue))), sparse=True, kind="system")

u_hat = A.solve(b)

uj = D.backward(u_hat)
uq = lambdify(t, ue)(D.mesh())
error = np.linalg.norm(uj - uq)
if "PYTEST" in os.environ:
    assert error < ulp(1000), error
    sys.exit(0)

print(f"poisson1D_curv l2 error = {error:2.6e}")

# Plot solution
assert isinstance(D, LGComposite)
xf, yf, zf = D.cartesian_mesh("uniform", 400)
tf = D.mesh("uniform", 400)
uf = lambdify(t, ue)(tf)  # Exact solution on a fine uniform grid

# Computed solution sits at the (quadrature) mesh points
xm, ym, zm = D.cartesian_mesh()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(xf, yf, zf, "k", lw=0.5, label="curve")
ax.plot(xf, yf, zf + uf, "C0", label="exact")
ax.plot(xm, ym, zm + uj, "C1o", ms=4, label="computed")
ax.legend()
plt.show()
