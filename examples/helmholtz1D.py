# Solve Helmholtz' equation
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.operators import Div, Grad
from jaxfun.typing import TestSpaceKind
from jaxfun.utils.common import lambdify, ulp

x = sp.Symbol("x", real=True)
N = 80
# Method of manufactured solution
ue = sp.exp(sp.cos(2 * sp.pi * x))

bcs = {"left": {"D": float(ue.subs(x, -1))}, "right": {"D": float(ue.subs(x, 1))}}
D = FunctionSpace(N, Chebyshev, bcs=bcs, name="D", fun_str="psi")
T = D.get_testspace(kind=TestSpaceKind.PG)
v = TestFunction(T, name="v")
u = TrialFunction(D, name="u")
ue = D.system.expr_psi_to_base_scalar(ue)

A, L = inner(v * (Div(Grad(u)) + u) - v * (Div(Grad(ue)) + ue), sparse=True)

xj = D.mesh(kind="uniform", N=100)
uh = A.solve(L)
uj = D.evaluate(xj, uh)
uej = lambdify(D.system.x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "PYTEST" in os.environ:
    assert error < jnp.sqrt(ulp(10)), error
    sys.exit(1)

print("Error =", error)
plt.plot(xj, uej, "r")
plt.plot(xj, uj, "b")
plt.show()
