# Solve Helmholtz' equation
import os
import sys
import jax 

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.basespace import n

# from jaxfun.Jacobi import Jacobi as space
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.Legendre import Legendre as space
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

x = sp.Symbol("x", real=True)
M = 20
# ue = sp.exp(sp.cos(2 * sp.pi * x))

omega = 20
alpha = sp.Rational(1, 10)
phi = jnp.pi / 2
A = 1
ue = A * sp.exp(-alpha * omega * x) * sp.sin(sp.sqrt(1 - alpha**2) * omega * x + phi)

domain = (0, 0.5)

bcs = {
    "left": {"D": float(ue.subs(x, domain[0]))},
    "right": {"D": float(ue.subs(x, domain[1]))},
}
D = FunctionSpace(
    M, space, domain=domain, bcs=bcs, name="D", fun_str="psi"
)
v = TestFunction(D, name="v")
u = TrialFunction(D, name="u")

# Method of manufactured solution
x = D.system.x  # use the same coordinate as u and v
ue = D.system.expr_psi_to_base_scalar(ue)

A, L = inner(
    v * (Div(Grad(u)) + 1 * u) - v * (Div(Grad(ue)) + 1 * ue),
    #(Div(Grad(v)) + 1 * v) * (Div(Grad(u)) + 1 * u) - (Div(Grad(v)) + 1 * v) * (Div(Grad(ue)) + 1 * ue),
    sparse=True,
    sparse_tol=1000,
    return_all_items=False,
)

xj = D.mesh()
uh = jnp.linalg.solve(A.todense(), L)
uj = D.backward(uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej)
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)
plt.plot(xj, uej, "r")
plt.plot(xj, uj, "b:")
plt.show()
