# Solve Helmholtz' equation
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.Basespace import n

# from jaxfun.Jacobi import Jacobi as space
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.Legendre import Legendre as space
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

x = sp.Symbol("x", real=True)
M = 50
ue = sp.exp(sp.cos(2 * sp.pi * x))

bcs = {"left": {"D": float(ue.subs(x, -1))}, "right": {"D": float(ue.subs(x, 1))}}
D = FunctionSpace(M, space, bcs=bcs, name="D", fun_str="psi", scaling=n + 1)
v = TestFunction(D, name="v")
u = TrialFunction(D, name="u")

# Method of manufactured solution
x = D.system.x  # use the same coordinate as u and v
ue = D.system.expr_psi_to_base_scalar(ue)

A, L = inner(
    v * (Div(Grad(u)) + u) - v * (Div(Grad(ue)) + ue),
    sparse=True,
    sparse_tol=1000,
    return_all_items=False,
)

xj = D.mesh(kind="uniform", N=100)
uh = jnp.linalg.solve(A.todense(), L)
uj = D.evaluate(xj, uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)
plt.plot(xj, uej, "r")
plt.plot(xj, uj, "b")
plt.show()
