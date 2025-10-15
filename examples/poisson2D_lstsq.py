# ruff: noqa: E402
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxfun import Div, Grad
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.pinns import LSQR, FlaxFunction, Trainer
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
w = FlaxFunction(V, name="w")

N = 32
mesh = Rectangle(N, N, -1, 1, -1, 1)
xyi = mesh.get_points_inside_domain(C.__class__.__name__.lower())
xyb = mesh.get_points_on_domain(C.__class__.__name__.lower())
wi = mesh.get_weights_inside_domain(C.__class__.__name__.lower())
wb = mesh.get_weights_on_domain(C.__class__.__name__.lower())

x, y = V.system.base_scalars()
ue = (1 - x**2) * (1 - y**2)  # manufactured solution

f = Div(Grad(w)) + x * w * w.diff(x) - (Div(Grad(ue)) + x * ue * ue.diff(x))

loss_fn = LSQR((f, xyi, 0, wi), (w, xyb, 0, wb))
trainer = Trainer(loss_fn)

t0 = time.time()

opt_adam = adam(w, learning_rate=1e-3)
trainer.train(opt_adam, 1000, epoch_print=100)

opt_lbfgs = lbfgs(w, memory_size=20)
trainer.train(opt_lbfgs, 1000, epoch_print=100, update_global_weights=100)

opt_hess = GaussNewton(w, use_lstsq=True)
trainer.train(opt_hess, 10, epoch_print=1, abs_limit_change=0)

print("time", time.time() - t0)

uj = lambdify((x, y), ue)(*xyi.T)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.contourf(
    xyi[:, 0].reshape((N, N)), xyi[:, 1].reshape((N, N)), w(xyi).reshape((N, N))
)
ax1.contourf(xyi[:, 0].reshape((N, N)), xyi[:, 1].reshape((N, N)), uj.reshape((N, N)))
# plt.colorbar()

error = jnp.linalg.norm(w.module(xyi)[:, 0] - uj) / jnp.sqrt(len(xyi))
print("Error", error)

if "PYTEST" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)
