# ruff: noqa: E402
import os
import sys
import time

import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.galerkin import FunctionSpace, Legendre, TestFunction
from jaxfun.operators import Dot, Grad
from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, Trainer
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import adam, lbfgs
from jaxfun.utils.common import Domain, jacn, lambdify

domain = Domain(-1, 1)
V = FunctionSpace(
    32,
    Legendre.Legendre,
    bcs={"left": {"D": 0}, "right": {"D": 0}},
    name="V",
    domain=domain,
)
W = MLPSpace(64, dims=1, rank=0, name="V")
w = FlaxFunction(W, "w", rngs=nnx.Rngs(1000))
v = TestFunction(V, name="v")

# Manufactured solution
x = V.system.x
ue = (1 - x**2) * sp.cos(2 * sp.pi * x)

N = 1000
mesh = Line(domain.lower, domain.upper, key=nnx.Rngs(1000)())

xj = mesh.get_points_inside_domain(N, "legendre")
wj = mesh.get_weights_inside_domain(N, "legendre")
xb = mesh.get_points_on_domain()

# fv = (Div(Grad(w)) + w - (Div(Grad(ue)) + ue)) * v
fv = -Dot(Grad(w), Grad(v)) + w * v - (-Dot(Grad(ue), Grad(v)) + ue * v)
loss_fn = Loss((fv, xj, 0, wj), (w, xb, 0, 100))
trainer = Trainer(loss_fn)

opt_adam = adam(w, learning_rate=1e-3)

t0 = time.time()

trainer.train(opt_adam, 5000, epoch_print=1000)

print(f"Time Adam {time.time() - t0:.1f}s")

opt_lbfgs = lbfgs(w, memory_size=50, max_linesearch_steps=5)

t0 = time.time()
trainer.train(opt_lbfgs, 5000, epoch_print=1000, update_global_weights=1000)

print(f"Time LBFGS {time.time() - t0:.1f}s")

df = lambda mod, x, k: jacn(mod, k)(x).reshape((-1, 1))
uej = lambdify(x, ue)
duej = lambdify(x, sp.diff(ue, x))
d2uej = lambdify(x, sp.diff(ue, x, 2))


def print_error(t0):
    print(
        "Accuracy f(x)=",
        jnp.linalg.norm((w.module(t0) - uej(t0)) / len(t0)),
        "f'(x)=",
        jnp.linalg.norm((df(w.module, t0, 1) - duej(t0)) / len(t0)),
        "f''(x)=",
        jnp.linalg.norm((df(w.module, t0, 2) - d2uej(t0)) / len(t0)),
    )


t0 = jnp.linspace(-1, 1, 1000)[:, None]
print_error(t0)

if "PYTEST" in os.environ:
    error = jnp.linalg.norm((w.module(t0) - uej(t0)) / len(t0))
    assert error < 1e-4, error
    sys.exit(1)
