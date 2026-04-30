# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxfun.operators import Div, Grad
from jaxfun.pinns.loss import Loss
from jaxfun.pinns.mesh import Triangle
from jaxfun.pinns.module import FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace
from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

N = 5000
mesh = Triangle()
xyi = mesh.get_points(N, domain="inside")
xyb = mesh.get_points(N, domain="boundary")

V = MLPSpace([20], dims=2, rank=0, name="V")
x, y = V.base_variables()

u = FlaxFunction(V, name="u")

eq = Div(Grad(u)) - 1

loss_fn = Loss((eq, xyi), (u, xyb))

trainer = Trainer(loss_fn)

trainer.train(adam(u), 1000, abs_limit_change=0)
trainer.train(lbfgs(u), 1000, print_final_loss=True)

xa = jnp.vstack((xyi, xyb))
mesh.plot_solution(xa, u(xa), xb=xyb)
plt.show()