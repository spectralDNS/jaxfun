# ruff: noqa: E402
import os
import sys

import jax
import matplotlib.pyplot as plt

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxfun.coordinates import x
from jaxfun.galerkin import (
    CartesianProduct,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.la.blocktpmatrix import BlockArray
from jaxfun.operators import Constant, Div, Dot, Grad

f = (1 - x) ** 2 * (1 + x) ** 2
N = 24
bcsx = {"left": {"D": 0}, "right": {"D": 0}}
bcsy = {"left": {"D": 0}, "right": {"D": f}}
nu = Constant("nu", 0.01)
D0 = FunctionSpace(N, Legendre.Legendre, bcs=bcsx, name="D0")
D1 = FunctionSpace(N, Legendre.Legendre, bcs=bcsy, name="D1")
P0 = FunctionSpace(
    N - 2, Legendre.Legendre, name="P0"
)  # Need to use N - 2 to escape several nullspaces and excessive pinning  # noqa: E501
T0 = TensorProduct(D0, D1, name="T0")
T1 = TensorProduct(D0, D0, name="T1")
Q = TensorProduct(P0, P0, name="Q")
V = CartesianProduct(T0, T1, name="V", rank=1)  # Vector
W = CartesianProduct(V, Q, name="W")

u, p = TrialFunction(W, name="up")
v, q = TestFunction(W, name="vq")

A, a = inner(
    Dot(nu * Div(Grad(u)), v), sparse=True, kind="system", num_quad_points=(N, N)
)
B, b = inner(q * Div(u), sparse=True, kind="system", num_quad_points=(N, N))
D = inner(p * Div(v), sparse=True, kind="bilinear", num_quad_points=(N, N))

C = A + B + D
c = a + b  # ty:ignore[unsupported-operator]

# pin pressure dof 0
C_pin = C.tosparse().pin({2 * (N - 2) ** 2: 0})
d = C_pin.lu_solve(c.flatten(), method="rcm", pivot=True)

D = BlockArray(W, flat_array=d)
up_ = W.backward(D.array, N=(None, None, (N, N)))

if "PYTEST" in os.environ:
    for i in range(3):
        assert jnp.all(jnp.isfinite(up_[i]))
    sys.exit(0)

xj = W.mesh()
shape = up_[0].shape
x0, y0 = jnp.broadcast_to(xj[0], shape), jnp.broadcast_to(xj[1], shape)
plt.figure()
plt.spy(C_pin.todense())
plt.figure()
plt.contourf(x0, y0, jnp.sqrt(up_[0] ** 2 + up_[1] ** 2))
plt.quiver(x0, y0, up_[0], up_[1])
plt.show()
