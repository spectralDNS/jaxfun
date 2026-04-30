# ruff: noqa: E402
from time import time as _time

import jax

from jaxfun import lambdify

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx

from jaxfun.galerkin import Chebyshev
from jaxfun.operators import Constant
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import TimeMarchingLoss as Loss
from jaxfun.pinns.mesh import Line, TimeMarchingMesh
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import KANMLPSpace, MLPSpace, sPIKANSpace
from jaxfun.pinns.optimizer import TimeMarchingTrainer as Trainer, adam, lbfgs, soap
from jaxfun.utils import lambdify

Nt = 160
Nx = 800
tmin = 0
tmax = jnp.pi / 10
left = 0
right = jnp.pi
line = Line(left, right)
time = Line(tmin, tmax)
mesh = TimeMarchingMesh(line, time)
xi = mesh.get_points(Nx, Nt, domain="inside", kind="random")
xba = mesh.get_points(Nx, Nt, domain="boundary", kind="random")
xt0 = mesh.get_points(Nx, Nt, domain="initial-time", kind="random")
xt1 = mesh.get_points(Nx, Nt, domain="end-time", kind="random")

xi = jnp.vstack((xi, xt1[1:-1]))
xt0 = xt0[1:-1]  # Remove boundary points
xb = xba[(abs(xba[:, 0] - left) < 1e-8) | (abs(xba[:, 0] - right) < 1e-8)]

V = MLPSpace(
    [12, 12],
    dims=1,
    transient=True,
    rank=0,
    name="V",
    act_fun=nnx.sigmoid,
    weight_factorization=False,
)
# V = sPIKANSpace(
#    4,
#    [6],
#    dims=1,
#    transient=True,
#    rank=0,
#    name="V",
#    domains=[(left, right), (tmin, tmax)],
# )
# V = KANMLPSpace(
#    4,
#    8,
#    dims=1,
#    transient=True,
#    rank=0,
#    name="V",
#    domains=[(left, right), (tmin, tmax)],
#    weight_factorization=True
# )

x, t = V.base_variables()

ue = sp.sin(x) * (sp.sin(t) + sp.cos(t))

u = FlaxFunction(V, name="u")

# Dirichlet boundary conditions
u0 = ue.subs(t, 0)

# Neumann condition on t=tmin
du0 = ue.diff(t).subs(t, tmin)

c = Constant("c", sp.S(1))
eq = u.diff(t, 2) - c**2 * u.diff(x, 2)

loss_fn = Loss(
    (eq, xi, 0, 1.0 / xi.shape[0]),
    (u, xb, 0, 1.0 / xb.shape[0]),
    (u - u0, xt0, 0, 1.0 / xt0.shape[0]),
    (u.diff(t) - du0, xt0, 0, 1.0 / xt0.shape[0]),
    initial_conditions=2,
)

trainer = Trainer(loss_fn, mesh)
trainer.train(adam(u), 5000, epoch_print=1000, abs_limit_loss=1e-6)
losses = [jnp.array(trainer.losses)]

opt_lbfgs = lbfgs(u, memory_size=20)
results = []
Nsteps = 4
t0 = _time()
for step in range(Nsteps):
    # opt_lbfgs.opt_state = optax.tree.zeros_like(opt_lbfgs.opt_state)
    trainer.train(
        opt_lbfgs, 5000, epoch_print=1000, abs_limit_loss=1e-6, print_final_loss=True
    )
    results.append(trainer.evaluate_at_time(200, u=u.module, t=time.left))
    if step < Nsteps - 1:
        trainer.update_time(u.module)
    losses.append(jnp.array(trainer.losses))

results.append(trainer.evaluate_at_time(200, u=u.module, t=time.right))

print(f"Total time for {Nsteps} steps: {_time() - t0:.2f} seconds")
uej = lambdify(x, ue.subs(t, mesh.deltat * Nsteps))(results[Nsteps][0][:, 0])

print(
    "Error at final time step:",
    jnp.linalg.norm(results[Nsteps][1][:, 0] - uej) / jnp.sqrt(uej.shape[0]),
)

plt.figure()
for step, (x0, u0) in enumerate(results):
    plt.plot(x0[:, 0], u0, label=f"t={step}")
plt.legend()
plt.show()
