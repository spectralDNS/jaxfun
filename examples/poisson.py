# Solve Poisson's equation
import matplotlib.pyplot as plt
import sympy as sp
import jax.numpy as jnp 
from jaxfun.Legendre import Legendre as space
from jaxfun.composite import Composite
from jaxfun.inner import inner
from jaxfun.arguments import TestFunction, TrialFunction, x


# Method of manufactured solution
ue = (1 - x**2) * sp.exp(sp.cos(2 * sp.pi * x))
f = ue.diff(x, 2)
N = 100
bcs = {'left': {'D': 0}, 'right': {'D': 0}}
C = Composite(space, N, bcs)
v = TestFunction(x, C)
u = TrialFunction(x, C)
A = inner(v*sp.diff(u, x, 2), sparse=True)
b = inner(v*f)
uj = jnp.linalg.solve(A.todense(), b)
xj = jnp.linspace(-1, 1, 100)
plt.plot(xj, sp.lambdify(x, ue)(xj), "r")
plt.plot(xj, C.evaluate(xj, uj), "b")
plt.show()
