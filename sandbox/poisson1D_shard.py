# Solve Poisson's equation
# ruff: noqa: E402
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_num_cpu_devices', 2)

import jax.flatten_util
import jax.numpy as jnp
import optax
import sympy as sp
from flax import nnx
from jax import random
from jax.sharding import PartitionSpec

from jaxfun.operators import Div, Grad
from jaxfun.pinns import (
    Loss,
    FlaxFunction,
    MLPSpace,
    run_optimizer,
)
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import adam, lbfgs
from jaxfun.utils.common import Domain, lambdify, ulp

print(jax.devices())

auto_mesh = jax.make_mesh((len(jax.devices()),), ('data',))

V = MLPSpace([8], dims=1, rank=0, name="VN")
w = FlaxFunction(V, name="w")
x = V.system.x
domain = Domain(-sp.pi, sp.pi)
ue = sp.sin(x) + x**2

# collocation points
mesh = Line(10000, float(domain.lower), float(domain.upper))
xj = mesh.get_points_inside_domain("uniform")
xb = mesh.get_points_on_domain()

@nnx.pmap(axis_name='data')
def train_step(model: nnx.Module, optimizer: nnx.Optimizer) -> float:
    gd, state = nnx.split(model, nnx.Param)
    unravel = jax.flatten_util.ravel_pytree(state)[1]
    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    loss_fn_split = lambda state: loss_fn(nnx.merge(gd, state))
    H_loss_fn = lambda flat_weights: loss_fn(nnx.merge(gd, unravel(flat_weights)))
    optimizer.update(
        gradients,
        grad=gradients,
        value_fn=loss_fn_split,
        value=loss,
        H_loss_fn=H_loss_fn,
    )
    return loss

with jax.set_mesh(auto_mesh):
    xp = jax.device_put(xj, PartitionSpec('data', None))
    jax.debug.visualize_array_sharding(xp)

    # Equations to solve
    f = Div(Grad(w)) - w - (Div(Grad(ue)) - ue)
    loss_fn = Loss((f, xp), (w, xb, lambdify(x, ue)(xb)))

    opt_adam = adam(w.module, learning_rate=1e-3)
    #opt_repl = jax.device_put_replicated(opt_adam, jax.devices())
    module_repl = jax.device_put_replicated(w.module, jax.devices())
    #opt_lbfgs = lbfgs(w.module, memory_size=20)
    t0 = time.time()
    train_step(module_repl, opt_adam)
    #run_optimizer(loss_fn, opt_adam, 100, epoch_print=10, update_global_weights=10)
    print("Time", time.time() - t0)

uej = lambdify(x, ue)(xj)
print(
    jnp.linalg.norm(w.module(xj) - uej) / jnp.sqrt(len(xj)),
)
