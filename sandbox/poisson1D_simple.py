# Solve Poisson's equation
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sympy as sp

from jaxfun import Div, Domain, Grad, lambdify
from jaxfun.pinns import Loss, FlaxFunction, Line, MLPSpace, Trainer, sPIKANSpace
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs, sgd

domain = Domain(0, 1)
#V = MLPSpace([6, 6, 6], dims=1, rank=0, name="V")
V = sPIKANSpace(9, [2], dims=1, rank=0, name="S", domains=(domain,))
w = FlaxFunction(V, name="w")
x = V.system.x
ue = sp.sin(x * sp.pi)
mesh = Line(4*w.dim-2, domain.lower, domain.upper)
xj = mesh.get_points_inside_domain("random")
xb = mesh.get_points_on_domain()
ub = lambdify(x, ue)(xb)

f = Div(Grad(w)) - Div(Grad(ue))

loss_fn = Loss((f, xj), (w, xb, ub))
opt_sgd = sgd(w, learning_rate=1e-3, end_learning_rate=1e-4, decay_steps=2000, nesterov=True)
opt_adam = adam(w, learning_rate=1e-3, end_learning_rate=1e-4, decay_steps=2000, nesterov=True)
opt_lbfgs = lbfgs(w, memory_size=100, initial_guess_strategy="keep")
opt_hess = GaussNewton(w, use_lstsq=False, cg_max_iter=50)

trainer = Trainer(loss_fn)

#trainer.train(opt_sgd, 50000, epoch_print=1000)
trainer.train(opt_adam, 2000, epoch_print=1000)
trainer.train(opt_lbfgs, 2000, epoch_print=100)
#trainer.train(opt_hess, 2000, epoch_print=100, abs_limit_change=0)

y = jnp.linspace(0, 1, 1000)[:, None]

print("L2 error:", jnp.linalg.norm(w.module(y) - lambdify(x, ue)(y)) / jnp.sqrt(y.shape[0]))