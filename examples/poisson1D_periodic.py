# Solve Poisson's equation
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.Fourier import Fourier
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

M = 10
D = FunctionSpace(M, Fourier, name="D", fun_str="psi", domain=(-2 * sp.pi, 2 * sp.pi))
v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
x = D.system.x  # use the same coordinate as u and v
ue = sp.cos(2 * x) + sp.I * sp.sin(1 * x)

A, b = inner(v * Div(Grad(u)) + v * Div(Grad(ue)), sparse=True)

uh = jnp.hstack((jnp.array([0.0]), b[1:] / A.todense().diagonal()[1:]))

uj = D.backward(uh)
xj = D.mesh()
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)
plt.plot(xj, uej.real, "r")
plt.plot(xj, uj.real, "b")
plt.plot(D.mesh(N=100), D.backward(uh, N=100).real, "g")

plt.show()
