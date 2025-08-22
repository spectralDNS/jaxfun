import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# import pyvista
import sympy as sp
from flax import nnx
from soap_jax import soap

from jaxfun import Div, Dot, Grad
from jaxfun.pinns.freeze import freeze_layer, unfreeze_layer
from jaxfun.pinns.hessoptimizer import hess
from jaxfun.pinns.module import (
    LSQR,
    CompositeMLP,
    CompositeNetwork,
    FlaxFunction,
    MLPSpace,
    PirateSpace,
    run_optimizer,
    train,
)
from jaxfun.utils import lambdify

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

print("JAX running on", jax.devices()[0].platform.upper())

Re = 10.0
rho = 1.0
nu = 2.0 / Re
N = 32
xj = jnp.linspace(-1, 1, N + 1)
c, wq = np.polynomial.legendre.leggauss(N + 1)
xyi = jnp.array(jnp.meshgrid(c, c, indexing="ij")).reshape((2, (N + 1) ** 2)).T
wq2 = jnp.outer(wq, wq).flatten() * xyi.shape[0]
xyb = np.array(jnp.vstack((jnp.hstack((c, c, c, c)),) * 2)).T

xyb[: N + 1, 1] = -1
xyb[N + 1 : 2 * (N + 1), 1] = 1
xyb[2 * (N + 1) : 3 * (N + 1), 0] = -1
xyb[3 * (N + 1) :, 0] = 1
xyb = jnp.array(xyb)
xyp = jnp.array([[0.0, 0.0]])

ub = np.zeros(xyb.shape)
ub[N + 1 : 2 * (N + 1), 0] = (1 - xyb[N + 1 : 2 * (N + 1), 0]) ** 2 * (
    1 + xyb[N + 1 : 2 * (N + 1), 0]
) ** 2
ub = jnp.array(ub)

V = PirateSpace(
    [64, 64], dims=2, rank=1, fourier_emb={"embed_scale": 1.0, "embed_dim": 64}
)  # Vector space for velocity
Q = PirateSpace(
    [64, 64], dims=2, rank=0, fourier_emb={"embed_scale": 1.0, "embed_dim": 64}
)  # Scalar space for pressure
VQ = CompositeNetwork((V, Q))  # Coupled space V x Q

# V = MLPSpace([20, 20], dims=2, rank=1)  # Vector space for velocity
# Q = MLPSpace([20, 20], dims=2, rank=0)  # Scalar space for pressure
# VQ = CompositeMLP((V, Q))  # Coupled space V x Q

up = FlaxFunction(
    VQ,
    "up",
    rngs=nnx.Rngs(2002),
    kernel_init=nnx.initializers.xavier_normal(dtype=float),
)
u, p = up

x, y = VQ.system.base_scalars()
i, j = VQ.system.base_vectors()

eq1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
eq2 = Div(u)

# Each item is (equation, points, target, optional weights)
eqs = LSQR(
    (
        (eq1, xyi, 0),  # momentum vector equation
        (eq2, xyi, 0),  # Divergence constraint
        (u, xyb, ub, 10),  # Boundary conditions on u
        (p, xyp, 0, 10),  # Pressure pin-point
    )
)

# Alternatively
# eqs = LSQR((
#    (Dot(eq1, i), xyi), # momentum eq in x-direction
#    (Dot(eq1, j), xyi), # momentum eq in y-direction
#    (eq2, xyi),         # Divergence constraint
#    (Dot(u, i), xyb, ub[:, 0]), # Boundary condition on u_x
#    (Dot(u, j), xyb),           # Boundary condition on u_y
#    (p, xyp)            # Pressure pin-point
# ))


opt = soap(
    learning_rate=3e-3,
    b1=0.95,
    b2=0.95,
    weight_decay=0.01,
    precondition_frequency=5,
)

opt_soap = nnx.Optimizer(up.module, opt)
train_step = train(eqs)
tm1 = time.time()
run_optimizer(train_step, up.module, opt_soap, 3000, "SOAP", 100)
print("Time SOAP", time.time() - tm1)

# opt = optax.adam(optax.linear_schedule(1e-3, 1e-4, 10000))
# optlbfgs = optax.lbfgs(
#     memory_size=100,
#     linesearch=optax.scale_by_zoom_linesearch(20, max_learning_rate=1.0),
# )
# opthess = hess(
#     use_lstsq=False,
#     cg_max_iter=100,
#     linesearch=optax.scale_by_zoom_linesearch(25, max_learning_rate=1.0),
# )
# opt_adam = nnx.Optimizer(up.module, opt)

# train_step = train(eqs)
# t0 = time.time()
# print(up.module)
# run_optimizer(train_step, up.module, opt_adam, 1000, "Adam", 100)
# print("Time Adam", time.time() - t0)

# # up.module = freeze_layer(up.module, "hidden", 0)
# # up.module = freeze_layer(up.module, "hidden", 1)

# opt_lbfgs = nnx.Optimizer(up.module, optlbfgs)
# t1 = time.time()
# run_optimizer(train_step, up.module, opt_lbfgs, 1000, "LBFGS", 100)
# print("Time LBFGS", time.time() - t1)

# # up.module = unfreeze_layer(up.module, "hidden", 0)

# opt_hess = nnx.Optimizer(up.module, opthess)
# t2 = time.time()
# run_optimizer(train_step, up.module, opt_hess, 10, "Hess", 1)
# print("Time Hess", time.time() - t2)

gd, st = nnx.split(up.module)
pyt, ret = jax.flatten_util.ravel_pytree(st)

yj = jnp.linspace(-1, 1, 50)
xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
z = jnp.column_stack((xx.ravel(), yy.ravel()))
uvp = up.module(z)
plt.contourf(xx, yy, uvp[:, 0].reshape(xx.shape), 100)
plt.figure()
plt.contourf(xx, yy, uvp[:, 1].reshape(xx.shape), 100)
plt.figure()
plt.contourf(xx, yy, uvp[:, 2].reshape(xx.shape), 100)
plt.colorbar()

# m = pyvista.read(
#     "/Users/mikaelmortensen/MySoftware/OpenFOAM/cavity/cavity/VTK/cavity_5000.vtk"
# )
# cen = m.cell_centers()
# pts = cen.points
# xO = jnp.array(pts[:, 0], dtype=float)
# yO = jnp.array(pts[:, 1], dtype=float)
# zO = jnp.column_stack((xO, yO))
# U0 = up.module(zO)
# U1 = jnp.array(m.cell_data["U"], dtype=float)
# UC = jnp.array(
#     np.load(
#         "/Users/mikaelmortensen/MySoftware/shenfun/demo/drivencavity_Re10_N100.npy",
#         allow_pickle=True,
#     )
# )
# print("U0", jnp.linalg.norm(UC.T - U0[:, :2].reshape((100, 100, 2))))
# print("U1", jnp.linalg.norm(UC.T - U1[:, :2].reshape((100, 100, 2))))

# # plt.show()
