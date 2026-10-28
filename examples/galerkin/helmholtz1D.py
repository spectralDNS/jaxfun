# Solve Helmholtz' equation in 1D with Dirichlet boundary conditions
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.coordinates import R
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.operators import Div, Grad
from jaxfun.typing import TestSpaceKind
from jaxfun.utils.common import lambdify, ulp

N = 60
R1 = R(1)
x = R1.x
ue = sp.exp(sp.cos(2 * sp.pi * x))  # Manufactured solution

domain = (0, 1)
bcs = {
    "left": {"D": ue.subs(x, domain[0])},
    "right": {"D": ue.subs(x, domain[1])},
}
D = FunctionSpace(N, Chebyshev, bcs=bcs, name="D", fun_str="psi", domain=domain)
P = D.get_testspace(kind=TestSpaceKind.GR)
v = TestFunction(P, name="v")
u = TrialFunction(D, name="u")
f = Div(Grad(ue)) + ue

A, b = inner(v, Div(Grad(u)) + u - f, sparse=True, kind="system")

uh = A.solve(b)

xj = D.mesh(kind="uniform", N=100)
uj = D.evaluate(xj, uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "PYTEST" in os.environ:
    assert error < jnp.sqrt(ulp(10)), error
    sys.exit(0)

print("Error =", error)
plt.plot(xj, uej, "r")
plt.plot(xj, uj, "b")
plt.show()
