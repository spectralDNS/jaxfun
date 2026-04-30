# ruff: noqa: E402
from time import time as _time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx

from jaxfun.operators import Constant
from jaxfun.pinns.loss import Loss, Residual
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace, PirateSpace
from jaxfun.pinns.optimizer import DiscreteTimeTrainer as Trainer, adam, lbfgs
from jaxfun.utils import lambdify

Nx = 200
t0 = 0
dt = jnp.pi / 100
left = 0
right = sp.pi
mesh = Line(left, right)
xa = mesh.get_points(Nx, domain="all", kind="random")
xi = mesh.get_points(Nx, domain="inside", kind="random")
xb = mesh.get_points(Nx, domain="boundary", kind="random")

# Bra: softmax, glu, swish, tanh, sigmoid
V = MLPSpace(64, dims=1, transient=False, rank=0, name="V", act_fun=nnx.softmax)
# V = PirateSpace(
#    16,
#    dims=1,
#    transient=False,
#    rank=0,
#    name="V",
#    act_fun=nnx.tanh,
#    act_fun_hidden=nnx.swish,
# )

x = V.system.x
t = sp.Symbol("t", real=True)

u = FlaxFunction(V, name="u")

ue = sp.sin(x) * (sp.cos(t) + sp.sin(t))
u0 = sp.sin(x)

c = Constant("c", sp.S.One)

loss_fn = Loss((u, xb), (u - u0, xi, 0, 1.0 / Nx))
trainer = Trainer(loss_fn)

opt_adam = adam(u)
opt_lbfgs = lbfgs(u, memory_size=10)
losses = []

# Initialize t=0
trainer.train(opt_adam, 5000, epoch_print=1000, abs_limit_loss=1e-6)
losses.append(jnp.array(trainer.losses))
trainer.train(
    opt_lbfgs, 5000, epoch_print=1000, abs_limit_loss=1e-8, print_final_loss=True
)
losses.append(jnp.array(trainer.losses))
trainer.step(u.module)
x0 = mesh.get_points(200, domain="all", kind="uniform")
results = [u(x0)]
unm1 = u(xi)

# Initialize timestep 1 at t=dt using du/dt(t=0) = sin(x)
FE_res = Residual(c**2 * dt**2 * u.diff(x, 2), xi)
loss_fn.residuals[1].target_expr = sp.S.Zero
loss_fn.residuals[1].target0 = 0
loss_fn.residuals[1].target = unm1 * (1 + dt) + 0.5 * FE_res.evaluate(u.module)

trainer.train(
    opt_lbfgs, 5000, epoch_print=1000, abs_limit_loss=1e-8, print_final_loss=True
)
losses.append(jnp.array(trainer.losses))
results.append(u(x0))
un = u(xi)
trainer.step(u.module)

Nsteps = 40
t0 = _time()
for step in range(1, Nsteps):
    print(f"Time step {step + 1}/{Nsteps}")
    loss_fn.residuals[1].target = 2 * un - unm1 + FE_res.evaluate(u.module)
    opt_lbfgs.opt_state = optax.tree.zeros_like(opt_lbfgs.opt_state)
    trainer.train(
        opt_lbfgs, 5000, epoch_print=1000, abs_limit_loss=1e-8, print_final_loss=True
    )
    results.append(u(x0))
    unm1 = un
    un = u(xi)
    trainer.step(u.module)
    losses.append(jnp.array(trainer.losses))

uej = lambdify(x, ue.subs(t, dt * Nsteps))(x0[:, 0])

print("Training time:", _time() - t0)
print(
    "Error at final time step:",
    jnp.linalg.norm(results[-1] - uej) / jnp.sqrt(uej.shape[0]),
)


for i, res in enumerate(results):
    plt.plot(x0, res, label=f"t={i * dt:.2f}")
plt.legend()
plt.title("1D Wave Equation solved with PINNs")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.show()
