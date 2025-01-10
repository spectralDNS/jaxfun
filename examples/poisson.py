# Solve Poisson's equation
# ruff: noqa: E402
import jax 
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt
import sympy as sp
import jax.numpy as jnp 
from jaxfun.utils.common import lambdify
from jaxfun.Legendre import Legendre as space
from jaxfun.composite import Composite
from jaxfun.inner import inner
from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.operators import Grad, Div, Dot
from jaxfun.Basespace import n


M = 20
bcs = {'left': {'D': 0}, 'right': {'D': 0}}
D = Composite(space, M, bcs, scaling=n+1, name='D', fun_str='psi')
v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
x = D.system.x # use the same coordinate as u and v
ue = (1 - x**2) * sp.exp(sp.cos(2 * sp.pi * x))

#A = inner(v*sp.Derivative(u, x, 2), sparse=True)
#A = inner(-Dot(Grad(v), Grad(u)), sparse=True)
A = inner(v*Div(Grad(u)), sparse=True)
b = inner(v*sp.Derivative(ue, x, 2))

uj = jnp.linalg.solve(A.todense(), b)
xj = jnp.linspace(-1, 1, 100)
plt.plot(xj, lambdify(x, ue)(xj), "r")
plt.plot(xj, D.evaluate(xj, uj), "b")
plt.show()
