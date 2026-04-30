# ruff: noqa: E402
from time import time as _time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from flax import nnx

from jaxfun.operators import Constant
from jaxfun.pinns.loss import Residual, TimeMarchingLoss as Loss
from jaxfun.pinns.mesh import Line, TimeMarchingMesh
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace
from jaxfun.pinns.optimizer import TimeMarchingTrainer as Trainer, adam, lbfgs

Nt = 3
Nx = 200
t0 = 0
dt = 0.02
left = -1
right = 1
line = Line(left, right)
time = Line(0, dt)
mesh = TimeMarchingMesh(line, time)
xi = mesh.get_points(Nx, Nt, domain="inside", kind=["random", "uniform"])
xba = mesh.get_points(Nx, Nt, domain="boundary", kind=["random", "uniform"])
xt0 = mesh.get_points(Nx, Nt, domain="initial-time", kind=["random", "uniform"])

xt0 = xt0[1:-1]  # Remove boundary points
xb = xba[(abs(xba[:, 0] - left) < 1e-8) | (abs(xba[:, 0] - right) < 1e-8)]

V = MLPSpace(
    [16],
    dims=1,
    transient=True,
    rank=0,
    name="V",
    act_fun=[nnx.sigmoid, lambda x: x**2],
    weight_factorization=False,
)
x, t = V.base_variables()

u = FlaxFunction(
    V, name="u", kernel_init=nnx.initializers.xavier_normal(), rngs=nnx.Rngs(111)
)

u0 = sp.exp(-(x**2) / 0.05)

nu = Constant("nu", sp.Rational(1, 10))
eq = u.diff(t, 1) - nu * u.diff(x, 2)

loss_fn = Loss(
    (eq, xi, 0, 1.0 / xi.shape[0]),
    (u, xb, 0, 1.0 / xb.shape[0]),
    (u - u0, xt0, 0, 1.0 / xt0.shape[0]),
    initial_conditions=1,
)
trainer = Trainer(loss_fn, mesh)

opt_adam = adam(u)
opt_lbfgs = lbfgs(u, memory_size=10)

# Initialize
trainer.train(opt_adam, 5000, epoch_print=1000, abs_limit_loss=1e-6)

results = []
Nsteps = 4
t0 = _time()
for step in range(Nsteps):
    print(f"Time step {step + 1}/{Nsteps}")
    trainer.train(
        opt_lbfgs,
        20000,
        epoch_print=1000,
        abs_limit_loss=1e-7,
        print_final_loss=True,
        update_global_weights=-1,
    )
    results.append(trainer.evaluate_at_time(200, u=u.module, t=time.left))
    if step < (Nsteps - 1):
        trainer.update_time(u.module)

results.append(trainer.evaluate_at_time(200, u=u.module, t=time.right))

print(f"Total time for {Nsteps} steps: {_time() - t0:.2f} seconds")

for i, (x0, u0) in enumerate(results):
    plt.plot(x0[:, 0], u0, label=f"t={i * dt:.2f}")

plt.legend()
plt.title("1D Diffusion Equation solved with PINNs")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.show()
