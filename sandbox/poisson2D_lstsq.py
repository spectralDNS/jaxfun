# ruff: noqa: E402
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from soap_jax import soap

from jaxfun import Div, Grad
from jaxfun.pinns import (
    LSQR,
    FlaxFunction,
    MLPSpace,
    PirateSpace,
    run_optimizer,
)
from jaxfun.pinns.mesh import Rectangle
from jaxfun.utils import lambdify
from jaxfun.utils.common import ulp

print("JAX running on", jax.devices()[0].platform.upper())

N = 16
mesh = Rectangle(N, N, -1, 1, -1, 1)
xyi = mesh.get_points_inside_domain()
xyb = mesh.get_points_on_domain()

V = PirateSpace(
    [20], dims=2, rank=0, name="V", act_fun=nnx.tanh, act_fun_hidden=nnx.swish
)
w = FlaxFunction(V, name="w")

x, y = V.system.base_scalars()
ue = (1 - x**2) * (1 - y**2)  # manufactured solution

f = (
    Div(Grad(w))
    + 8 * x * w * w.diff(x) * w.diff(y)
    - (Div(Grad(ue)) + 8 * x * ue * ue.diff(x) * ue.diff(y))
)

loss_fn = LSQR((f, xyi), (w, xyb))

t0 = time.time()

opt_adam = nnx.Optimizer(w.module, optax.adam(learning_rate=1e-3))
run_optimizer(loss_fn, w.module, opt_adam, 1000, "Adam", 100, update_global_weights=10)
# opt_soap = nnx.Optimizer(w.module, soap(learning_rate=1e-3))
# run_optimizer(loss_fn, w.module, opt_soap, 1000, "Soap", 100, abs_limit_change=0)

opt_lbfgs = nnx.Optimizer(
    w.module,
    optax.lbfgs(
        memory_size=20,
        linesearch=optax.scale_by_zoom_linesearch(25, max_learning_rate=1.0),
    ),
)
run_optimizer(
    loss_fn, w.module, opt_lbfgs, 10000, "LBFGS", 1000, update_global_weights=10
)
print("time", time.time() - t0)

uj = lambdify((x, y), ue)(*xyi.T)
plt.contourf(
    xyi[:, 0].reshape((N, N)),
    xyi[:, 1].reshape((N, N)),
    w(xyi).reshape((N, N)) - uj.reshape((N, N)),
)
plt.colorbar()

error = jnp.linalg.norm(w.module(xyi)[:, 0] - uj) / jnp.sqrt(len(xyi))
print("Error", error)

if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)
