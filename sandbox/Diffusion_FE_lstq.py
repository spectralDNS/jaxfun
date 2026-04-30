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
from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

Nx = 1000
t0 = 0
dt = 0.0098
left = -1
right = 1
mesh = Line(left, right)
xa = mesh.get_points(Nx, domain="all", kind="random")
xi = mesh.get_points(Nx, domain="inside", kind="random")
xb = mesh.get_points(Nx, domain="boundary", kind="random")

V = MLPSpace(
    32, dims=1, transient=False, rank=0, name="V", act_fun=[nnx.tanh, lambda x: x**2]
)
x = V.system.x

u = FlaxFunction(V, name="u")

u0 = sp.exp(-(x**2) / 0.01)

nu = Constant("nu", sp.Rational(1, 10))
eq = nu * u.diff(x, 2)

loss_fn = Loss((u, xb), (u - u0, xi, 0, 1.0 / xi.shape[0]))

trainer = Trainer(loss_fn)

opt_adam = adam(u)
opt_lbfgs = lbfgs(u)

losses = []
# Initialize
trainer.train(opt_adam, 5000, epoch_print=1000, abs_limit_loss=1e-6)
losses.append(jnp.array(trainer.losses))
trainer.train(opt_lbfgs, 5000, epoch_print=1000, abs_limit_loss=1e-7, print_final_loss=True)
losses.append(jnp.array(trainer.losses))

FE_res = Residual(u + nu * dt * u.diff(x, 2), xi)
loss_fn.residuals[1].target_expr = sp.S.Zero

x0 = mesh.get_points(200, domain="all", kind="uniform")
results = [u(x0)]
Nsteps = 10
t0 = _time()
for step in range(Nsteps):
    print(f"Time step {step+1}/{Nsteps}")
    loss_fn.residuals[1].target = FE_res.evaluate(u.module)
    #opt_lbfgs.opt_state = optax.tree.zeros_like(opt_lbfgs.opt_state)
    trainer.train(opt_lbfgs, 10000, epoch_print=1000, abs_limit_loss=1e-7, print_final_loss=True)
    results.append(u(x0))
    losses.append(jnp.array(trainer.losses))

print(f"Total time for {Nsteps} steps: {_time() - t0:.2f} seconds")
for i, res in enumerate(results):
    plt.plot(x0, res, label=f"t={i*dt:.2f}")
plt.legend()
plt.title("1D Diffusion Equation solved with PINNs")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.show()