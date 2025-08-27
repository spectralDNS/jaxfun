# Solve Poisson's equation
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.basespace import n

# from jaxfun.Legendre import Legendre as space
from jaxfun.Chebyshev import Chebyshev as space
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner

# from jaxfun.Jacobi import Jacobi as space
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

M = 30
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, space, bcs=bcs, name="D", fun_str="psi", scaling=n + 1)
v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
x = D.system.x  # use the same coordinate as u and v
ue = 1 - x**2  # * sp.exp(sp.cos(2 * sp.pi * x))

# A = inner(v*sp.Derivative(u, x, 2), sparse=True)
# A = inner(-Dot(Grad(v), Grad(u)), sparse=True)
# A = inner(v*Div(Grad(u)), sparse=True)
# b = inner(v*sp.Derivative(ue, x, 2))
A, b = inner(
    v * Div(Grad(u)) - v * sp.Derivative(ue, x, 2), sparse=True, sparse_tol=1000
)

xj = D.mesh()
uh = jnp.linalg.solve(A.todense(), b)
uj = D.backward(uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)
plt.plot(xj, uej, "r")
plt.plot(xj, uj, "b")
plt.plot(D.mesh(N=100), D.backward(uh, N=100).real, "g")
plt.show()
