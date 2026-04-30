# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx

from jaxfun import Cross, Curl, Div, Dot, Grad, get_CoordSys
from jaxfun.operators import Constant, Source
from jaxfun.pinns import Loss, run_optimizer
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.mesh import AnnulusPolar
from jaxfun.pinns.module import (
    Comp,
    FlaxFunction,
    MLPSpace,
)
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs

r, theta = sp.symbols("r,theta", real=True, positive=True)
P = get_CoordSys("P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))

V = MLPSpace([8], dims=2, rank=1, system=P, name="V")  # Vector space for velocity
Q = MLPSpace([8], dims=2, rank=0, system=P, name="Q")  # Scalar space for pressure

u = FlaxFunction(V, name="u", rngs=nnx.Rngs(2002))
p = FlaxFunction(Q, name="p", rngs=nnx.Rngs(202))

module = Comp(u, p)

r, theta = V.system.base_scalars()
b0, b1 = V.system.base_vectors()

Re = 10.0  # Define Reynolds number
nu = Constant("nu", 2.0 / Re)
R1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
R2 = Div(u)

N = 20
mesh = AnnulusPolar(N, N, 0.5, 1)
xi = mesh.get_points_inside_domain("uniform")
xb = mesh.get_points_on_domain("uniform")
xp = jnp.array([[0.6, 0.0]])

ub = DirichletBC(u, xb, 0, sp.Piecewise((1, r <= mesh.radius_inner), (0, True)))

loss_fn = Loss((R1, xi), (R2, xi), (u, xb, ub, 10), (p, xp, 0, 10))

opt_adam = adam(module, 1e-3)
run_optimizer(loss_fn, opt_adam, 5000, 1000)

opt_lbfgs = lbfgs(module, memory_size=100)
run_optimizer(loss_fn, opt_lbfgs, 5000, 1000)

xy = u.cartesian_mesh(xi)
plt.scatter(xy[:, 0], xy[:, 1], c=u(xi)[:, 1])
plt.show()
