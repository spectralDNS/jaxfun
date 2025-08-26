# ruff: noqa: E402
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pyvista
import sympy as sp
from flax import nnx
from soap_jax import soap

from jaxfun import Div, Dot, Grad, Outer
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.freeze import freeze_layer, unfreeze_layer
from jaxfun.pinns.hessoptimizer import hess
from jaxfun.pinns.mesh import Rectangle
from jaxfun.pinns.module import (
    LSQR,
    Comp,
    CompositeMLP,
    CompositeNetwork,
    FlaxFunction,
    MLPSpace,
    PirateSpace,
    run_optimizer,
)
from jaxfun.utils.common import ulp

print("JAX running on", jax.devices()[0].platform.upper())

Re = 10.0
rho = 1.0
nu = 2.0 / Re

N = 50

mesh = Rectangle(N, N, -1, 1, -1, 1)
xyi = mesh.get_points_inside_domain()
xyb = mesh.get_points_on_domain()
xyp = jnp.array([[0.0, 0.0]])
wqi = mesh.get_weights_inside_domain()
wqb = mesh.get_weights_on_domain()

V = PirateSpace([20], dims=2, rank=1, name="V")     # Vector space for velocity
Q = PirateSpace([20], dims=2, rank=0, name="Q")     # Scalar space for pressure

u = FlaxFunction(V, "u", rngs=nnx.Rngs(2002), bias_init=nnx.initializers.normal())
p = FlaxFunction(Q, "p", rngs=nnx.Rngs(2002), bias_init=nnx.initializers.normal())

module = Comp([u, p])

x, y = V.system.base_scalars()
i, j = V.system.base_vectors()

eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
# eq1 = Div(Outer(u, u)) - nu * Div(Grad(u)) + Grad(p)

eq2 = Div(u)

ub = DirichletBC(
    u, xyb, sp.Piecewise((0, y < 1), ((1 - x) ** 2 * (1 + x) ** 2, True)), 0
) # No-slip on walls, u=(1-x)**2*(1+x)**2 on lid

# Each item is (equation, points, target, optional weights)
loss_fn = LSQR(
    (eq1, xyi),    # momentum vector equation
    (eq2, xyi),    # Divergence constraint
    (u, xyb, ub),  # Boundary conditions on u
    (p, xyp, 0),   # Pressure pin-point
    alpha=0.8,     # Global weights update parameter
)

opt = optax.adam(optax.linear_schedule(1e-3, 1e-4, 10000))
optlbfgs = optax.lbfgs(
    memory_size=100,
    linesearch=optax.scale_by_zoom_linesearch(20, max_learning_rate=1.0),
)
opthess = hess(
    use_lstsq=False,
    cg_max_iter=100,
    linesearch=optax.scale_by_zoom_linesearch(25, max_learning_rate=1.0),
)
opt_adam = nnx.Optimizer(module, opt)
opt_soap = nnx.Optimizer(module, soap(optax.linear_schedule(1e-3, 1e-4, 10000)))

# tm1 = time.time()
# run_optimizer(loss_fn, module, opt_soap, 1000, "SOAP", 100, abs_limit_change=0)
# print("Time Soap", time.time() - tm1)

t0 = time.time()
run_optimizer(loss_fn, module, opt_adam, 1000, "Adam", 100, update_global_weights=10)
print("Time Adam", time.time() - t0)

opt_lbfgs = nnx.Optimizer(module, optlbfgs)
t1 = time.time()
run_optimizer(
    loss_fn, module, opt_lbfgs, 1000, "LBFGS", 100, update_global_weights=10
)
print("Time LBFGS", time.time() - t1)

# opt_hess = nnx.Optimizer(module, opthess)
# t2 = time.time()
# run_optimizer(loss_fn, module, opt_hess, 10, "Hess", 1, abs_limit_change=ulp(1))
# print("Time Hess", time.time() - t2)

yj = jnp.linspace(-1, 1, 50)
xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
z = jnp.column_stack((xx.ravel(), yy.ravel()))
uvp = module(z)
plt.contourf(xx, yy, uvp[:, 0].reshape(xx.shape), 100)
plt.figure()
plt.contourf(xx, yy, uvp[:, 1].reshape(xx.shape), 100)
plt.figure()
plt.contourf(xx, yy, uvp[:, 2].reshape(xx.shape), 100)
plt.colorbar()

m = pyvista.read(
    "/Users/mikaelmortensen/MySoftware/OpenFOAM/cavity/cavity/VTK/cavity_5000.vtk"
)
cen = m.cell_centers()
pts = cen.points
xO = jnp.array(pts[:, 0], dtype=float)
yO = jnp.array(pts[:, 1], dtype=float)
zO = jnp.column_stack((xO, yO))
U0 = module(zO)
U1 = jnp.array(m.cell_data["U"], dtype=float)
UC = jnp.array(
    np.load(
        "/Users/mikaelmortensen/MySoftware/shenfun/demo/drivencavity_Re10_N100.npy",
        allow_pickle=True,
    )
)
print("U0", jnp.linalg.norm(UC.T - U0[:, :2].reshape((100, 100, 2))))
print("U1", jnp.linalg.norm(UC.T - U1[:, :2].reshape((100, 100, 2))))

# plt.show()
