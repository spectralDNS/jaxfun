# ruff: noqa: E402
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.operators import Constant
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import Loss
from jaxfun.pinns.mesh import Rectangle, points_along_axis
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace
from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

Nt = 40
Nx = 40
t0 = 0
tmax = 5
left = -8
right = 8
mesh = Rectangle(left, right, t0, tmax)
xi = mesh.get_points(Nx * Nt, 4 * Nt, domain="inside", kind="random")
xba = mesh.get_points(Nx * Nt, 4 * Nt, domain="boundary", kind="random")

# Boundary points on three sides of the rectangle:
xb = xba[((xba[:, 0] <= left) | (xba[:, 0] >= right)) | (xba[:, 1] <= t0)]

V = MLPSpace([20], dims=1, transient=True, rank=0, name="V")
x, t = V.base_variables()

u = FlaxFunction(V, name="u")

ub = DirichletBC(
    u, xb, sp.Piecewise((0, x <= left), (0, x >= right), (sp.exp(-(x**2) / 2), t <= t0))
)

nu = Constant("nu", sp.Rational(1, 10))
eq = u.diff(t) + u * u.diff(x) - nu * u.diff(x, 2)

loss_fn = Loss((eq, xi), (u, xb, ub))

trainer = Trainer(loss_fn)

trainer.train(adam(u), 1000, abs_limit_change=0)
trainer.train(lbfgs(u), 1000, print_final_loss=True)

if "PYTEST" in os.environ:
    sys.exit(1)

xj = jnp.linspace(left, right, 50)
tj = jnp.linspace(t0, tmax, 50)
xx, tt = jnp.meshgrid(xj, tj, sparse=False, indexing="ij")
z = jnp.column_stack((xx.ravel(), tt.ravel()))
uj = u.module(z)
plt.contourf(xx, tt, uj[:, 0].reshape(xx.shape), 100)

plt.figure()
x0 = points_along_axis(jnp.linspace(left, right, 100), 0)
x2 = points_along_axis(jnp.linspace(left, right, 100), 2)
x4 = points_along_axis(jnp.linspace(left, right, 100), 4)
plt.plot(x0[:, 0], u(x0), "b-", label="t=0")
plt.plot(x2[:, 0], u(x2), "r-", label="t=2")
plt.plot(x4[:, 0], u(x4), "g-", label="t=4")
plt.show()
