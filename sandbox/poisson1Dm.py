# Solve Poisson's equation
# ruff: noqa: E402
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import sympy as sp
from flax import nnx
from jax import random

from jaxfun.basespace import n
from jaxfun.galerkin import Chebyshev, Legendre
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.composite import BCGeneric, Composite, DirectSum
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.sincos import Cosine, SinCos, SinCosHalf, Sine
from jaxfun.operators import Div, Grad
from jaxfun.pinns import (
    Loss,
    FlaxFunction,
    MLPSpace,
    PirateSpace,
    Trainer,
)
from jaxfun.pinns.freeze import freeze_layer, unfreeze_layer
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import adam, GaussNewton, lbfgs
from jaxfun.utils.common import Domain, lambdify, ulp

M = 20
space = Chebyshev.Chebyshev
# space = Legendre.Legendre

x = sp.symbols("x", real=True)
domain = Domain(-sp.pi, sp.pi)
ue = sp.sin(x) + x**2
bcs = {
    "left": {"D": float(ue.subs(x, domain.lower))},
    "right": {"D": float(ue.subs(x, domain.upper))},
}
# bcs = {"left": {"D": 0}, "right": {"D": 0}}

C = Composite(
    M + 5,
    SinCosHalf,
    stencil={0: 1, 4: -1},
    bcs=bcs,
    name="C",
    fun_str="psi",
    domain=domain,
)
B = BCGeneric(
    1,
    Chebyshev.Chebyshev,
    bcs=bcs,
    domain=domain,
    num_quad_points=M + 5,
    name="B",
    fun_str="psi_b",
)
D = DirectSum(C, B)

# D = FunctionSpace(M, space, bcs=bcs, name="D", fun_str="psi", scaling=n + 1, domain=domain)

v = TestFunction(D)
u = TrialFunction(D)

# Method of manufactured solution
x = D.system.x  # use the same coordinate as u and v
ue = D.system.expr_psi_to_base_scalar(ue)

# A = inner(v*sp.Derivative(u, x, 2), sparse=True)
# A = inner(-Dot(Grad(v), Grad(u)), sparse=True)
# A = inner(v*Div(Grad(u)), sparse=True)
# b = inner(v*sp.Derivative(ue, x, 2))
# A, b = inner(
#    v * (Div(Grad(u)) + u) - v * (Div(Grad(ue)) + ue), sparse=True, sparse_tol=1000
# )

# lstsq
A, b = inner(
    # (Div(Grad(v)) - v) * (Div(Grad(u)) - u) - (Div(Grad(v)) - v) * (Div(Grad(ue)) - ue),
    v * (Div(Grad(u)) - u) - v * (Div(Grad(ue)) - ue),
    sparse=True,
    sparse_tol=1000,
)

xj = D.mesh()
uh = jnp.linalg.solve(A.todense(), b)
uj = D.backward(uh)
uej = lambdify(x, ue)(xj)
error = jnp.linalg.norm(uj - uej) / jnp.sqrt(len(xj))
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)
xx = jnp.linspace(float(domain.lower), float(domain.upper), 100)
XX = D.map_reference_domain(xx)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(xx, lambdify(x, ue)(xx), "b*", xx, D.evaluate(XX, uh), "ro")
ax2.plot(xx, lambdify(x, ue)(xx) - D.evaluate(XX, uh), "b*")

#
# plt.plot(xj, uej, "r")
# plt.plot(xj, uj, "b")
# plt.plot(D.mesh(N=100), D.backward(uh, N=100).real, "g")
# plt.show()

# VN = MLPSpace([8, 8], dims=1, rank=0, name="VN")
# VN = PirateSpace([32], dims=1, rank=0, name="VN")
# VN = Legendre.Legendre(20, domain=domain)
# VN = Chebyshev.Chebyshev(20, domain=domain)
VN = SinCosHalf(21, name="VN", domain=domain)
# VN = Chebyshev.Chebyshev(60)
w = FlaxFunction(VN, name="w")

# collocation points
mesh = Line(5 * w.dim, float(domain.lower), float(domain.upper))
xj = mesh.get_points_inside_domain("uniform")
xb = mesh.get_points_on_domain()

# Equations to solve
f = Div(Grad(w)) - w - (Div(Grad(ue)) - ue)
loss_fn = Loss((f, xj), (w, xb, lambdify(x, ue)(xb)))

opt_adam = adam(w.module, learning_rate=1e-3)
opt_lbfgs = lbfgs(w.module, memory_size=20)
opt_hess = GaussNewton(w.module, use_lstsq=True)

trainer = Trainer(loss_fn)

t0 = time.time()
trainer.train(opt_adam, 1000, epoch_print=100, update_global_weights=10)
trainer.train(opt_lbfgs, 1000, epoch_print=100, update_global_weights=10)

# v.module = unfreeze_layer(w.module, "hidden", 0)
print("Time", time.time() - t0)

uej = lambdify(x, ue)(xj)
print(
    jnp.linalg.norm(uej - D.evaluate(xj, uh)) / jnp.sqrt(len(xj)),
    jnp.linalg.norm(w.module(xj) - uej) / jnp.sqrt(len(xj)),
)

xx = jnp.linspace(float(domain.lower), float(domain.upper), 100)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(xx, lambdify(x, ue)(xx), "b*", xx, w.module(xx), "ro")
ax2.plot(xx, lambdify(x, ue)(xx) - w.module(xx), "b*")
plt.show()

raise RuntimeError

import timeit

e0 = timeit.timeit(
    stmt="VN.evaluate(VN.map_reference_domain(xj), w.module.kernel.value[0])",
    globals=globals(),
    number=1000,
)

e1 = timeit.timeit(
    stmt="VN.evaluate2(VN.map_reference_domain(xj), w.module.kernel.value[0])",
    globals=globals(),
    number=1000,
)

e2 = timeit.timeit(
    stmt="VN.evaluate3(VN.map_reference_domain(xj), w.module.kernel.value[0])",
    globals=globals(),
    number=1000,
)
dv = jax.vmap(VN.evaluate, in_axes=(0, None))
a = dv(VN.map_reference_domain(xj), w.module.kernel.value[0])
e3 = timeit.timeit(
    stmt="dv(VN.map_reference_domain(xj), w.module.kernel.value[0])",
    globals=globals(),
    number=1000,
)
print(e0, e1, e2, e3)
