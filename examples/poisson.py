# Solve Poisson's equation
import matplotlib.pyplot as plt
import sympy as sp
import jax.numpy as jnp 
from jaxfun import Chebyshev
from jaxfun.composite import Composite
from jaxfun.inner import inner

s = sp.Symbol("s")

# Method of manufactured solution
ue = (1 - s**2) * sp.exp(sp.cos(2 * sp.pi * s))
f = ue.diff(s, 2)
N = 50
bcs = {'left': {'D': 0}, 'right': {'D': 0}}
C = Composite(Chebyshev, N, bcs)
v = (C, 0)
u = (C, 2)
A = inner(v, u, sparse=True)
b = inner(v, f)
u = jnp.linalg.solve(A.todense(), b)
x = jnp.linspace(-1, 1, 100)
plt.plot(x, sp.lambdify(s, ue)(x), "r")
plt.plot(x, Chebyshev.evaluate(x, u @ C.S), "b") # u @ C.S return coefficients in the orthogonal basis
plt.show()
