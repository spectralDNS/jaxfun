# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx

from jaxfun import Cross, Curl, Div, Dot, Grad
from jaxfun.arguments import Constant
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.mesh import Annulus
from jaxfun.pinns.module import Loss, Comp, FlaxFunction, MLPSpace, run_optimizer

V = MLPSpace([8], dims=2, rank=1, name="V")  # Vector space for velocity
Q = MLPSpace([8], dims=2, rank=0, name="Q")  # Scalar space for pressure

u = FlaxFunction(
    V,
    "u",
    rngs=nnx.Rngs(2002),
    kernel_init=nnx.initializers.xavier_normal(dtype=float),
)
p = FlaxFunction(
    Q,
    "p",
    rngs=nnx.Rngs(2001),
    kernel_init=nnx.initializers.xavier_normal(dtype=float),
)

module = Comp(u, p)

Re = 100.0  # Define Reynolds number
nu = Constant("nu", 2.0 / Re)
R1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
R2 = Div(u)

N = 30
mesh = Annulus(N, N, 0.5, 1)
xi = mesh.get_points_inside_domain("random")
xb = mesh.get_points_on_domain("random")
xp = jnp.array([[0.55, 0.0]])

x, y = V.system.base_scalars()
ub = DirichletBC(
    u,
    xb,
    sp.Piecewise(
        (sp.sin(sp.atan2(y, x)), sp.sqrt(x**2 + y**2) <= mesh.radius_inner),
        (0, True),
    ),
    sp.Piecewise(
        (-sp.cos(sp.atan2(y, x)), sp.sqrt(x**2 + y**2) <= mesh.radius_inner),
        (0, True),
    ),
)

loss_fn = Loss((R1, xi), (R2, xi), (u, xb, ub, 2), (p, xp, 0, 10))

opt = optax.adam(1e-3)
opt_adam = nnx.Optimizer(module, opt)
run_optimizer(loss_fn, module, opt_adam, 10000, "Adam", 1000, update_global_weights=10)

optlbfgs = optax.lbfgs(
    memory_size=100,
    linesearch=optax.scale_by_zoom_linesearch(20, max_learning_rate=1.0),
)
opt_lbfgs = nnx.Optimizer(module, optlbfgs)
run_optimizer(loss_fn, module, opt_lbfgs, 10000, "LBFGS", 1000, update_global_weights=10)
