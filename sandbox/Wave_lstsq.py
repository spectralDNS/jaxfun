# ruff: noqa: E402
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.galerkin import Chebyshev
from jaxfun.operators import Constant
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import Loss
from jaxfun.pinns.mesh import Rectangle, points_along_axis
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import KANMLPSpace, MLPSpace, sPIKANSpace
from jaxfun.pinns.optimizer import (
    Trainer,
    adam,
    lbfgs,
)
from jaxfun.utils import lambdify

Nt = 40
Nx = 20
tmin = 0
tmax = 2 * jnp.pi
left = 0
right = jnp.pi
mesh = Rectangle(left, right, tmin, tmax)
xi = mesh.get_points(Nx, Nt, domain="inside")
xa = mesh.get_points(Nx, Nt, domain="boundary")

# Boundary conditions on three sides of the rectangle only
xb = xa[(abs(xa[:, 0] - left) < 1e-8) | (abs(xa[:, 0] - right) < 1e-8)]

# Points along t=0 for Neumann condition
xt0 = xa[(abs(xa[:, 1] - tmin) < 1e-8)]
xt0 = xt0[1:-1] # Remove boundary points at x=0 and x=pi

#V = MLPSpace([20], dims=1, transient=True, rank=0, name="V")
V = sPIKANSpace(
   4,
   [6, 6],
   dims=1,
   transient=True,
   rank=0,
   name="V",
   domains=[(left, right), (tmin, tmax)],
)
x, t = V.base_variables()

ue = sp.sin(x) * (sp.sin(t) + sp.cos(t))

u = FlaxFunction(V, name="u")

# Initial condition
u0 = ue.subs(t, 0)
# Neumann condition on t=tmin
du0 = ue.diff(t).subs(t, tmin)

c = Constant("c", sp.S.One)
eq = u.diff(t, 2) - c**2 * u.diff(x, 2)

loss_fn = Loss((eq, xi), (u, xb), (u - u0, xt0), (u.diff(t) - du0, xt0))

t0 = time.time()
opt_adam = adam(u)
trainer = Trainer(loss_fn)
trainer.train(opt_adam, 5000, epoch_print=1000)

opt_lbfgs = lbfgs(u, memory_size=20)
trainer.train(opt_lbfgs, 5000, epoch_print=1000)
print("Training time:", time.time() - t0)

xj = jnp.linspace(left, right, 50)
tj = jnp.linspace(tmin, tmax, 50)
xx, tt = jnp.meshgrid(xj, tj, sparse=False, indexing="ij")
z = jnp.column_stack((xx.ravel(), tt.ravel()))
uj = u.module(z)
uej = lambdify((x, t), ue)(*z.T)
print("L2 error:", jnp.linalg.norm(uj[:, 0] - uej) / jnp.sqrt(len(uj)))
plt.figure(figsize=(3, 6))
plt.contourf(xx, tt, uj[:, 0].reshape(xx.shape), 100)
plt.figure(figsize=(3, 6))
plt.contourf(xx, tt, uj[:, 0].reshape(xx.shape) - uej.reshape(xx.shape), 100)
plt.colorbar()
plt.title("Error u - u_exact")

plt.figure()
x0 = points_along_axis(jnp.linspace(left, right, 100), tmin)
x1 = points_along_axis(jnp.linspace(left, right, 100), tmin + jnp.pi / 4)
x2 = points_along_axis(jnp.linspace(left, right, 100), tmin + 2 * jnp.pi / 4)
x3 = points_along_axis(jnp.linspace(left, right, 100), tmin + 3 * jnp.pi / 4)
x4 = points_along_axis(jnp.linspace(left, right, 100), tmin + 4 * jnp.pi / 4)
plt.plot(x0[:, 0], u(x0), "b-", label="t=0")
plt.plot(x1[:, 0], u(x1), "r-", label="t=π/4")
plt.plot(x2[:, 0], u(x2), "g-", label="t=π/2")
plt.plot(x3[:, 0], u(x3), "m-", label="t=3π/4")
plt.plot(x4[:, 0], u(x4), "k-", label="t=π")
plt.legend()
plt.show()
