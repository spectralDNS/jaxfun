# ruff: noqa: E402
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from jaxfun import Div, Grad
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.pinns import FlaxFunction, Loss, Trainer
from jaxfun.pinns.mesh import Rectangle
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs
from jaxfun.utils import lambdify
from jaxfun.utils.common import ulp

print("JAX running on", jax.devices()[0].platform.upper())

# V = PirateSpace(
#    [20], dims=2, rank=0, name="V", act_fun=nnx.tanh, act_fun_hidden=nnx.swish
# )
C = FunctionSpace(10, Chebyshev, domain=(-1, 1), name="C")
V = TensorProduct(C, C, name="V")
w = FlaxFunction(V, name="w", rngs=nnx.Rngs(1001))

N = 32
mesh = Rectangle(-1, 1, -1, 1)
points = C.__class__.__name__.lower()
xyi = mesh.get_points(N, N, domain="inside", kind=points)
xyb = mesh.get_points(N, N, domain="boundary", kind=points)
wi = mesh.get_weights(N, N, domain="inside", kind=points)
Nb = xyb.shape[0]

x, y = V.system.base_scalars()
ue = (1 - x**2) * (1 - y**2)  # manufactured solution

f = Div(Grad(w)) + x * w * w.diff(x) - (Div(Grad(ue)) + x * ue * ue.diff(x))

loss_fn = Loss((f, xyi, 0, wi), (w, xyb, 0, 10))
trainer = Trainer(loss_fn)

t0 = time.time()

opt_adam = adam(w, learning_rate=1e-3)
trainer.train(opt_adam, 1000, epoch_print=100)

opt_lbfgs = lbfgs(w, memory_size=20)
trainer.train(opt_lbfgs, 1000, epoch_print=100, update_global_weights=100)

opt_hess = GaussNewton(w, use_lstsq=True, use_GN=True)

trainer.train(opt_hess, 10, epoch_print=1)

print("time", time.time() - t0)

uj = lambdify((x, y), ue)(*xyi.T)
error = jnp.linalg.norm(w.module(xyi)[:, 0] - uj) / jnp.sqrt(len(xyi))
print("Error", error)

if "PYTEST" in os.environ:
    assert error < ulp(10000), error
    sys.exit(1)


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
M = N - 2
ax0.contourf(
    xyi[:, 0].reshape((M, M)), xyi[:, 1].reshape((M, M)), w(xyi).reshape((M, M))
)
ax1.contourf(xyi[:, 0].reshape((M, M)), xyi[:, 1].reshape((M, M)), uj.reshape((M, M)))
# plt.colorbar()
