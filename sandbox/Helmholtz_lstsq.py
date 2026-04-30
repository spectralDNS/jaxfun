# ruff: noqa: E402
import os
import socket

import jax

jax.config.update("jax_enable_x64", True)

import time
from functools import partial

import jax.numpy as jnp
import optax
import sympy as sp
from flax import nnx

from jaxfun.galerkin import Chebyshev, FunctionSpace, Legendre, TestFunction
from jaxfun.operators import Div, Dot, Grad
from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, Trainer, sPIKANSpace, PirateSpace
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs, soap, sgd
from jaxfun.utils.common import Domain, jacn, lambdify

domain = Domain(-1, 1)
# V = Chebyshev.Chebyshev(109, dims=1, rank=0, name="V", domain=domain)
V = FunctionSpace(
    32,
    Legendre.Legendre,
    bcs={"left": {"D": 0}, "right": {"D": 0}},
    name="V",
    domain=domain,
)
#W = MLPSpace(
#    [32], dims=1, rank=0, name="V", act_fun=jnp.tanh, weight_factorization=False
#)
# W = Legendre.Legendre(36, dims=1, rank=0, name="V", domain=domain)
# V = sPIKANSpace(8, [4], dims=1, rank=0, name="V", domains=[domain])
W = PirateSpace(24, dims=1, rank=0, name="W", nonlinearity=0.5, act_fun_hidden=nnx.swish)

w = FlaxFunction(W, "w", rngs=nnx.Rngs(1000))
# v = TestFunction(V, name="v")

x = W.system.x
ue = (1 - x**2) * sp.cos(2 * sp.pi * x)

# Each process gets N // world points, total global shape is (N, 1)
N = 1000
global_shape = (N, 1)

mesh = Line(domain.lower, domain.upper, key=nnx.Rngs(1000)())

xj = mesh.get_points_inside_domain(N, "random")
wj = mesh.get_weights_inside_domain(N, "random")
xb = mesh.get_points_on_domain(N)

ut = lambdify(x, (Div(Grad(ue)) + ue).doit())(xj)

f = Div(Grad(w)) + w - (Div(Grad(ue)) + ue)
loss_fn = Loss((f, xj), (w, xb, 0, 50))
# fv = -Dot(Grad(w), Grad(v)) + w * v - (-Dot(Grad(ue), Grad(v)) + ue * v)
# fv = f * v
# loss_fn = Loss((fv, xj, 0, wj), (w, xb, 0, 100))
trainer = Trainer(loss_fn)

opt_adam = adam(w, learning_rate=1e-3)
opt_sgd = sgd(w, learning_rate=1e-3)
t0 = time.time()

trainer.train(
    opt_adam,
    5000,
    epoch_print=1000,
    update_global_weights=-1,
    abs_limit_change=0,
)


print(f"Time Adam {time.time() - t0:.1f}s")

opt_lbfgs = lbfgs(w, memory_size=50, max_linesearch_steps=5)

opt_soap = soap(w, precondition_frequency=5, learning_rate=1e-3)

t0 = time.time()
trainer.train(opt_lbfgs, 10000, epoch_print=1000, update_global_weights=100)

print(f"Time LBFGS {time.time() - t0:.1f}s")
t0 = time.time()
opt_hess = GaussNewton(w, use_lstsq=True, use_GN=False)
trainer.train(opt_hess, 4, epoch_print=1, abs_limit_change=0)
print(f"Time Hessian {time.time() - t0:.1f}s")

df = lambda mod, x, k: jacn(mod, k)(x).reshape((-1, 1))

uej = lambdify(x, ue)
duej = lambdify(x, sp.diff(ue, x))
d2uej = lambdify(x, sp.diff(ue, x, 2))


def print_error(t0):
    print(
        "Accuracy f(x)=",
        jnp.linalg.norm((w.module(t0) - uej(t0)) / len(t0))
        / jnp.linalg.norm((uej(t0)) / len(t0)),
        "f'(x)=",
        jnp.linalg.norm((df(w.module, t0, 1) - duej(t0)) / len(t0))
        / jnp.linalg.norm((duej(t0)) / len(t0)),
        "f''(x)=",
        jnp.linalg.norm((df(w.module, t0, 2) - d2uej(t0)) / len(t0))
        / jnp.linalg.norm((d2uej(t0)) / len(t0)),
    )


t0 = jnp.linspace(-1, 1, 1000)[:, None]
print_error(t0)
