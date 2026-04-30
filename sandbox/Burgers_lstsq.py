# ruff: noqa: E402
import jax
from time import time as _time

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.operators import Constant
from jaxfun.pinns.loss import TimeMarchingLoss as Loss
from jaxfun.pinns.mesh import Line, TimeMarchingMesh
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import KANMLPSpace, MLPSpace, PirateSpace
from jaxfun.pinns.optimizer import TimeMarchingTrainer as Trainer, adam, lbfgs, soap

Nt = 8
Nx = 200
t0 = 0
tmax = 0.1
left = -8
right = 8
line = Line(left, right)
time = Line(t0, tmax)
mesh = TimeMarchingMesh(line, time)
xi = mesh.get_points(Nx, Nt, domain="inside", kind="uniform")
xba = mesh.get_points(Nx, Nt, domain="boundary", kind="uniform")
xt0 = mesh.get_points(Nx, Nt, domain="initial-time", kind="uniform")
xt0 = xt0[1:-1]  # Remove boundary points
xb = xba[((xba[:, 0] <= left) | (xba[:, 0] >= right))]  # Boundary points

V = MLPSpace(
    64, dims=1, transient=True, rank=0, name="V", act_fun=[jnp.tanh, lambda x: x**2]
)
#V = PirateSpace(
#    16, dims=1, rank=0, transient=True, name="V", nonlinearity=0.5, act_fun_final=lambda x: x**2
#)
# V = KANMLPSpace(
#    3,
#    8,
#    dims=1,
#    transient=True,
#    rank=0,
#    name="V",
#    domains=[(left, right), (t0, tmax)],
#    weight_factorization=True
# )
x, t = V.base_variables()

u = FlaxFunction(V, name="u")
# u = (8 - x) * (8 + x) * w

u0 = sp.exp(-(x**2) / 2)

nu = Constant("nu", sp.Rational(1, 10))
eq = u.diff(t) + u * u.diff(x) - nu * u.diff(x, 2)

loss_fn = Loss((eq, xi), (u, xb), (u - u0, xt0))

trainer = Trainer(loss_fn, mesh)

t0 = _time()
results = []
trainer.train(adam(u.module), 5000, epoch_print=1000, abs_limit_loss=1e-6)

Nsteps = 2
for step in range(Nsteps):
    trainer.train(lbfgs(u.module), 5000, epoch_print=1000, abs_limit_loss=5e-7)
    results.append(trainer.evaluate_at_time(100, u=u, t=time.left))
    if step < Nsteps - 1:
        trainer.update_time(u.module)
print("Total training time:", _time() - t0)
results.append(trainer.evaluate_at_time(100, u=u, t=time.right))

plt.figure()
for step, (x0, u0) in enumerate(results):
    plt.plot(x0[:, 0], u0, label=f"t={step}")
plt.legend()
plt.show()
