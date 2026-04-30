# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import sympy as sp
from flax import nnx

from jaxfun import Div, Grad, get_CoordSys
from jaxfun.operators import Constant
from jaxfun.pinns import Loss, Trainer
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.mesh import AnnulusPolar
from jaxfun.pinns.module import FlaxFunction, MLPSpace
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs

r, theta = sp.symbols("r,theta", real=True, positive=True)
P = get_CoordSys("P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))

Q = MLPSpace([8], dims=2, rank=0, system=P, name="Q")

u = FlaxFunction(Q, name="u", rngs=nnx.Rngs(202))

r, theta = Q.system.base_scalars()

alpha = Constant("alpha", 2.0)
eq = Div(Grad(u)) + alpha * r**2 * u

N, M = 20, 10

mesh = AnnulusPolar(0.5, 1)
xi = mesh.get_points_inside_domain(N, M, "uniform")
xb = mesh.get_points_on_domain(N, M, "uniform")

ub = DirichletBC(u, xb, sp.Piecewise((1, r <= mesh.radius_inner), (0, True)))

loss_fn = Loss((eq, xi), (u, xb, ub))

trainer = Trainer(loss_fn)
opt_adam = adam(u.module, 1e-3)
trainer.train(opt_adam, 5000, epoch_print=1000, update_global_weights=10)

opt_lbfgs = lbfgs(u.module, memory_size=100)
trainer.train(opt_lbfgs, 5000, epoch_print=1000, update_global_weights=10)

opt_hess = GaussNewton(u.module, use_lstsq=True, use_GN=True)
trainer.train(opt_hess, 100, epoch_print=10, update_global_weights=10)

xy = u.cartesian_mesh(xi)
# plt.scatter(xy[:, 0], xy[:, 1], c=u(xi))
plt.contourf(
    xy[:, 0].reshape((N, M)), xy[:, 1].reshape((N, M)), u(xi).reshape((N, M)), 100
)
plt.show()
