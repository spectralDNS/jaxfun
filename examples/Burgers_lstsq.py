# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx
from soap_jax import soap

from jaxfun.operators import Constant
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import LSQR
from jaxfun.pinns.mesh import Rectangle, points_along_axis
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace, PirateSpace
from jaxfun.pinns.optimizer import run_optimizer

Nt = 40
Nx = 40
t0 = 0
tmax = 5
left = -8
right = 8
mesh = Rectangle(Nx, Nt, left, right, t0, tmax)
xi = mesh.get_points_inside_domain()

# Boundary points on three sides of the rectangle:
xb = jnp.vstack(
    [
        points_along_axis(left, jnp.linspace(t0, tmax, Nt)),
        points_along_axis(right, jnp.linspace(t0, tmax, Nt)),
        points_along_axis(jnp.linspace(left, right, Nx)[1:-1], t0),
    ]
)

V = MLPSpace([20], dims=1, transient=True, rank=0, name="V")
x, t = V.base_variables()

u = FlaxFunction(V, name="u")

ub = DirichletBC(
    u, xb, sp.Piecewise((0, x <= -8), (0, x >= 8), (sp.exp(-(x**2) / 2), t <= 0))
)

nu = Constant("nu", sp.Rational(1, 10))
eq = u.diff(t) + u * u.diff(x) - nu * u.diff(x, 2)

loss_fn = LSQR((eq, xi), (u, xb, ub))

opt_soap = nnx.Optimizer(u.module, soap(optax.linear_schedule(1e-3, 1e-4, 10000)))
run_optimizer(loss_fn, u.module, opt_soap, 1000, "Soap", abs_limit_change=0)
#opt_adam = nnx.Optimizer(u.module, optax.adam(optax.linear_schedule(1e-3, 1e-4, 10000)))
#run_optimizer(loss_fn, u.module, opt_adam, 1000, "Adam")

optlbfgs = optax.lbfgs(
    memory_size=100,
    linesearch=optax.scale_by_zoom_linesearch(20, max_learning_rate=1.0),
)
opt_lbfgs = nnx.Optimizer(u.module, optlbfgs)
run_optimizer(
    loss_fn, u.module, opt_lbfgs, 1000, "LBFGS", 100, update_global_weights=10
)

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
