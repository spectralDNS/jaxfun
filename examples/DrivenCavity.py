# ruff: noqa: E402
import os

import jax
import matplotlib.pyplot as plt
from flax import nnx

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxfun.coordinates import x
from jaxfun.galerkin import CartesianProduct, FunctionSpace, Legendre, TensorProduct
from jaxfun.operators import Constant, Div, Dot, Grad
from jaxfun.pinns import FlaxFunction, Loss, Trainer, adam, lbfgs
from jaxfun.pinns.mesh import Rectangle

f = (1 - x) ** 2 * (1 + x) ** 2
N = 24
bcsx = {"left": {"D": 0}, "right": {"D": 0}}
bcsy = {"left": {"D": 0}, "right": {"D": f}}
Re = 10.0
rho = 1.0
nu = Constant("nu", 2.0 / Re)

D0 = FunctionSpace(N, Legendre.Legendre, bcs=bcsx, name="D0")
D1 = FunctionSpace(N, Legendre.Legendre, bcs=bcsy, name="D1")
P0 = FunctionSpace(N - 2, Legendre.Legendre, name="P0")
T0 = TensorProduct(D0, D1, name="T0")
T1 = TensorProduct(D0, D0, name="T1")
Q = TensorProduct(P0, P0, name="Q")
V = CartesianProduct(T0, T1, name="V", rank=1)
W = CartesianProduct(Q, V, name="W")

pu = FlaxFunction(W, name="pu", rngs=nnx.Rngs(2001))
p, u = pu

x, y = V.system.base_scalars()

eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
eq2 = Div(u)

N = 40
mesh = Rectangle(-1, 1, -1, 1)
kind = "legendre"
xyi = mesh.get_points(N, N, domain="inside", kind=kind)
xyp = jnp.array([[0.0, 0.0]])
wi = mesh.get_weights(N, N, domain="inside", kind=kind)
# Each item is (equation, points, target, optional weights)
loss_fn = Loss(
    (eq1, xyi, 0, wi),  # momentum vector equation
    (eq2, xyi, 0, wi),  # Divergence constraint
    (p, xyp, 0, 10),  # Pressure pin-point
)

opt_adam = adam(pu)
opt_lbfgs = lbfgs(pu, memory_size=50, max_linesearch_steps=5)

trainer = Trainer(loss_fn)

trainer.train(opt_adam, 5000, epoch_print=1000)

trainer.train(opt_lbfgs, 5000, epoch_print=1000, abs_limit_change=0)

yj = jnp.linspace(-1, 1, 50)
xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
z = jnp.column_stack((xx.ravel(), yy.ravel()))
uvp = pu(z)
plt.contourf(xx, yy, uvp[:, 0].reshape(xx.shape), 100)
plt.figure()
plt.contourf(xx, yy, uvp[:, 1].reshape(xx.shape), 100)
plt.figure()
plt.contourf(xx, yy, uvp[:, 2].reshape(xx.shape), 100)
plt.colorbar()
