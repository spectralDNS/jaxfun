# Solve Poisson's equation in 1D with Dirichlet boundary conditions
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as space
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

M = 32
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, space, bcs=bcs, name="D", fun_str="psi", scaling=n + 1)
v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
x = D.system.x  # use the same coordinate as u and v
ue = (1 - x**2) * sp.exp(sp.cos(sp.pi * x))

A, b = inner(
    v * Div(Grad(u)) - v * sp.Derivative(ue, x, 2),
    sparse=True,
    sparse_tol=1000,
    kind="system",
)

xj = D.mesh()
uh = A.solve(b)
uj = D.backward(uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "PYTEST" in os.environ:
    assert error < jnp.sqrt(ulp(10)), error
    sys.exit(0)

print("Error =", error)
plt.plot(xj, uej, "r")
plt.plot(xj, uj, "b")
plt.plot(D.mesh(N=100), D.backward(uh, N=100).real, "g")
plt.show()
