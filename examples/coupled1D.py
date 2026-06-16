# ruff: noqa: E402
# Solve Poisson's equation using mixed formulation in 1D
import jax.numpy as jnp
import sympy as sp

from jaxfun.coordinates import BaseScalar, x
from jaxfun.galerkin import CartesianProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction

# from jaxfun.galerkin.Chebyshev import Chebyshev as space
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre as space
from jaxfun.utils.common import lambdify, ulp

M = 60
ue = sp.exp(sp.cos(2 * sp.pi * x))

bcs = {"left": {"D": ue.subs(x, -1)}, "right": {"D": ue.subs(x, 1)}}

D = FunctionSpace(M, space, bcs=bcs, name="D", fun_str="psi")
S = FunctionSpace(M, space, name="S", fun_str="phi")
C = CartesianProduct(D, S, name="C")

v, q = TestFunction(C, name="vq")
u, s = TrialFunction(C, name="us")

# Method of manufactured solution
x: BaseScalar = C.system.x
ue = C.system.expr_psi_to_base_scalar(ue)

A, a = inner(s.diff(x) * v - ue.diff(x, 2) * v, kind="system", sparse=True)
B, b = inner(u.diff(x) * q - s * q, kind="system", sparse=True)

H = A + B
h = a + b  # ty:ignore[unsupported-operator]

uh = H.solve(h, method="banded", pivot=True)

z = H @ uh
assert jnp.linalg.norm(z.flatten() - h.flatten()) < ulp(100)

xj = D.mesh()
uj = C.backward(uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj[0] - uej)
print("Error =", error)
