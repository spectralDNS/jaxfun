# ruff: noqa: E402
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista
except ImportError:
    print("pyvista not found, skipping some parts of the example")
    pyvista = None
import sympy as sp
from flax import nnx

from jaxfun.operators import Div, Grad, Outer
from jaxfun.pinns.bcs import DirichletBC
from jaxfun.pinns.loss import LSQR
from jaxfun.pinns.mesh import Rectangle
from jaxfun.pinns.module import Comp, FlaxFunction
from jaxfun.pinns.nnspaces import MLPSpace
from jaxfun.pinns.optimizer import Trainer, adam, lbfgs

print("JAX running on", jax.devices()[0].platform.upper())

Re = 10.0
rho = 1.0
nu = 2.0 / Re

N = 50

mesh = Rectangle(N, N, -1, 1, -1, 1)
xyi = mesh.get_points_inside_domain("legendre")
xyb = mesh.get_points_on_domain("legendre")
xyp = jnp.array([[0.0, 0.0]])
wi = mesh.get_weights_inside_domain("legendre")
wb = mesh.get_weights_on_domain("legendre")

V = MLPSpace([30], dims=2, rank=1, name="V")  # Vector space for velocity
Q = MLPSpace([20], dims=2, rank=0, name="Q")  # Scalar space for pressure

u = FlaxFunction(V, "u", rngs=nnx.Rngs(2002), bias_init=nnx.initializers.normal())
p = FlaxFunction(Q, "p", rngs=nnx.Rngs(2001), bias_init=nnx.initializers.normal())

# eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
eq1 = Div(Outer(u, u)) - nu * Div(Grad(u)) + Grad(p)  # Alternative form

eq2 = Div(u)

module = Comp(u, p)
x, y = V.system.base_scalars()

ub = DirichletBC(
    u, xyb, sp.Piecewise((0, y < 1), ((1 - x) ** 2 * (1 + x) ** 2, True)), 0
)  # No-slip on walls, u=(1-x)**2*(1+x)**2 on lid

# Each item is (equation, points, target, optional weights)
loss_fn = LSQR(
    (eq1, xyi, 0, wi),  # momentum vector equation
    (eq2, xyi, 0, wi),  # Divergence constraint
    (u, xyb, ub, 2),  # Boundary conditions on u
    (p, xyp, 0, 10),  # Pressure pin-point
)

opt_adam = adam(module)
opt_lbfgs = lbfgs(module, memory_size=100)

trainer = Trainer(loss_fn)

t0 = time.time()
trainer.train(opt_adam, 5000, epoch_print=1000)
print("Time Adam", time.time() - t0)

t1 = time.time()
trainer.train(opt_lbfgs, 10000, epoch_print=1000, abs_limit_change=0)
print("Time LBFGS", time.time() - t1)

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
# plt.show()

if pyvista is not None:
    m = pyvista.read("./data/cavity_5000.vtk")
    cen = m.cell_centers()
    pts = cen.points
    xO = jnp.array(pts[:, 0], dtype=float)
    yO = jnp.array(pts[:, 1], dtype=float)
    zO = jnp.column_stack((xO, yO))
    U0 = module(zO)
    U1 = jnp.array(m.cell_data["U"], dtype=float)
    UC = jnp.array(
        np.load(
            "./data/drivencavity_Re10_N100.npy",
            allow_pickle=True,
        )
    )
    print("U0", jnp.linalg.norm(UC.T - U0[:, :2].reshape((100, 100, 2))))
    print("U1", jnp.linalg.norm(UC.T - U1[:, :2].reshape((100, 100, 2))))
