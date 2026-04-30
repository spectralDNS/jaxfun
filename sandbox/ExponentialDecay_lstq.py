# ruff: noqa: E402
from time import time as _time

import jax
from packaging.utils import _

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx

from jaxfun.operators import Constant
from jaxfun.pinns.loss import Loss, Residual, evaluate
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace
from jaxfun.pinns.optimizer import GaussNewton, Trainer, adam, lbfgs

V = MLPSpace(
    [6], dims=1, transient=False, rank=0, name="V", act_fun=[nnx.sigmoid, lambda x: x**2]
)
x = V.system.x

u = FlaxFunction(V, name="u")

Nx = u.dim
mesh = Line(0, 8)
xa = mesh.get_points(Nx, domain="all", kind="uniform")
xi = xa[1:]
xb = xa[0:1]
u0 = 1.0
a = Constant("a", 2)

loss_fn = Loss((u, xb, u0), (u.diff(x, 1) + a * u, xi, 0, 1.0 / xi.shape[0]))

trainer = Trainer(loss_fn)

opt_adam = adam(u)
opt_lbfgs = lbfgs(u)
opt_hess = GaussNewton(u, use_lstsq=False, cg_max_iter=50, initial_guess_strategy="one")

# Initialize
trainer.train(opt_adam, 1000, epoch_print=100, abs_limit_loss=1e-6)
#trainer.train(opt_lbfgs, 100, epoch_print=1, abs_limit_loss=1e-10, print_final_loss=True)
trainer.train(opt_hess, 100, epoch_print=1, abs_limit_loss=1e-12, print_final_loss=True)

xj = mesh.get_points(500, domain="all", kind="uniform")
plt.semilogy(xj, u(xj), 'b', xj, jnp.exp(-a.val * xj), 'r--', xa, u(xa), 'g')
plt.show()
