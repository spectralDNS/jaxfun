# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.operators import Constant, Div, Grad
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import Loss
from jaxfun.pinns.mesh import CartesianProductMesh, Line, Triangle
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace
from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

N = 5000
Nt = 10
T = 0.1
triangle = Triangle(boundary_factor=0.1)
line = Line(0, T)
mesh = CartesianProductMesh(triangle, line)

xyti = mesh.get_points(N, Nt, domain="inside")
xytb = mesh.get_points(N, Nt, domain="boundary")

V = MLPSpace([20], dims=2, transient=True, rank=0, name="V")
x, y, t = V.base_variables()

u = FlaxFunction(V, name="u")

c = Constant("c", 0.1)
eq = u.diff(t, 1) - c * Div(Grad(u))

xytb0 = xytb[
    (xytb[:, 2] <= 0.0)
    | (xytb[:, 0] <= 0.0)
    | (xytb[:, 1] <= 0.0)
    | (xytb[:, 0] + xytb[:, 1] >= 1.0)
]
# u0 = DirichletBC(u, xytb, sp.Piecewise((1, x <= 0.0), (0, t <= 0), (0, y <= 1.0)))
u0 = DirichletBC(u, xytb0, (x * y * (1 - x - y)) ** 2)

loss_fn = Loss((eq, xyti), (u, xytb0, u0))

trainer = Trainer(loss_fn)

trainer.train(adam(u), 1000, epoch_print=100, abs_limit_change=0)
trainer.train(lbfgs(u), 1000, print_final_loss=True, update_global_weights=100)

xy = mesh.submeshes[0].get_points(N, domain="all")

xyt0 = jnp.hstack((xy, jnp.zeros(N)[:, None]))
xyt1 = jnp.hstack((xy, T * jnp.ones(N)[:, None]))

mesh.submeshes[0].plot_solution(xyt0, u(xyt0))
mesh.submeshes[0].plot_solution(xyt1, u(xyt1))
