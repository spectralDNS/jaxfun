import jax.numpy as jnp
import sympy as sp
from sympy.functions import legendre as Leg

from jaxfun.galerkin import Legendre, TestFunction, TrialFunction, inner
from jaxfun.utils import ulp

N = 6
L = Legendre.Legendre(N, name="L")
u = TrialFunction(L)
v = TestFunction(L)
x = L.system.x

A = inner(x**2 * v * u, num_quad_points=7)

t = sp.integrate(x**2 * Leg(N-1, x) * Leg(N-1, x), (x, -1, 1)).n()

assert jnp.allclose(A[N-1, N-1], float(t), atol=ulp(1000))