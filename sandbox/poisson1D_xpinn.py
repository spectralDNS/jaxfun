# Solve Poisson's equation
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun import Div, Domain, Grad, lambdify
from jaxfun.pinns import FlaxFunction, Line, Loss, MLPSpace, Trainer, UnionSpace
from jaxfun.pinns.loss import get_Loss
from jaxfun.pinns.mesh import UnionMesh
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs, sgd

domains = (Domain(-1, -0.5), Domain(-0.5, 0), Domain(0, 0.5), Domain(0.5, 1))
V = UnionSpace(MLPSpace(12, name=f"V_{i}") for i in range(len(domains)))
w = FlaxFunction(V, name="w")
x = V.system.x
ue = sp.sin(x * sp.pi)
mesh = UnionMesh(tuple(Line(domain.lower, domain.upper) for domain in domains))

N = (500,)*4
xj = mesh.get_points_inside_domain(N, kind="uniform")
xb = mesh.get_points_on_domain()
xc = jnp.vstack(mesh.get_points_on_intersection())

f = Div(Grad(w)) - Div(Grad(ue))

loss_fn = get_Loss((f, xj), (w - ue, xb), (w, xc))

opt_sgd = sgd(
    w, learning_rate=1e-3, end_learning_rate=1e-4, decay_steps=2000, nesterov=True
)
opt_adam = adam(
    w, learning_rate=1e-3, end_learning_rate=1e-4, decay_steps=2000, nesterov=True
)
opt_lbfgs = lbfgs(w, memory_size=100)
# opt_hess = GaussNewton(w, use_lstsq=False, cg_max_iter=50)

trainer = Trainer(loss_fn)

# trainer.train(opt_sgd, 50000, epoch_print=1000)
trainer.train(opt_adam, 1000, epoch_print=1000)
trainer.train(opt_lbfgs, 1000, epoch_print=100)
# trainer.train(opt_hess, 2000, epoch_print=100, abs_limit_change=0)

M = len(domains) * 200
y = jnp.linspace(-1, 1, M).reshape((len(domains), M // len(domains), 1))

z0 = w.module(y, at_interfaces=False)
z1 = lambdify(x, ue)(y.reshape((M, 1)))

print(
    "L2 error:",
    jnp.linalg.norm(z0 - z1) / jnp.sqrt(y.shape[0]),
)

yy = y.reshape((M, 1))
plt.plot(yy, z0, label="PINN")
plt.plot(yy, z1, label="Exact", linestyle="dashed")
plt.legend()
plt.show()