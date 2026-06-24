# ruff: noqa: E402
# Solve nonlinear equation using mixed formulation in 1D
import os
import sys
import time

import jax.numpy as jnp
import sympy as sp

from jaxfun.coordinates import BaseScalar, x
from jaxfun.galerkin import CartesianProduct
from jaxfun.galerkin.arguments import TestFunction

# from jaxfun.galerkin.Chebyshev import Chebyshev as space
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.Legendre import Legendre as space
from jaxfun.pinns import FlaxFunction, Line, Loss, Trainer, adam, lbfgs
from jaxfun.utils.common import jacn, lambdify, ulp

M = 60
ue = sp.exp(sp.cos(2 * sp.pi * x))

bcs = {"left": {"D": ue.subs(x, -1)}, "right": {"D": ue.subs(x, 1)}}

D = FunctionSpace(M, space, bcs=bcs, name="D", fun_str="psi")
S = FunctionSpace(M, space, name="S", fun_str="phi")
C = CartesianProduct(D, S, name="C")

v, q = TestFunction(C, name="vq")

us = FlaxFunction(C, name="us")
u, s = us

x: BaseScalar = C.system.x
ue = C.system.expr_psi_to_base_scalar(ue)

eq1 = (s.diff(x) + u**2) - (ue.diff(x, 2) + ue**2)
eq2 = u.diff(x) - s

N = 1000
mesh = Line(-1, 1)
xj = mesh.get_points(N, domain="inside", kind="legendre")
wj = mesh.get_weights(N, domain="inside", kind="legendre")
loss_fn = Loss((eq1, xj, 0, wj), (eq2, xj, 0, wj))

opt_adam = adam(us)
opt_lbfgs = lbfgs(us, memory_size=50, max_linesearch_steps=10)

trainer = Trainer(loss_fn)

t0 = time.time()
trainer.train(opt_adam, 2000, epoch_print=1000)
print("Time Adam", time.time() - t0)

t1 = time.time()
trainer.train(opt_lbfgs, 500, epoch_print=100, abs_limit_change=0)
print("Time LBFGS", time.time() - t1)

df = lambda mod, x, k: jacn(mod, k)(x).reshape((-1, 1))
uej = lambdify(x, ue)
duej = lambdify(x, sp.diff(ue, x))
d2uej = lambdify(x, sp.diff(ue, x, 2))


def print_error(t0):
    print(
        "Accuracy f(x)=",
        jnp.linalg.norm((u(t0) - uej(t0)[:, 0]) / len(t0)),
        "f'(x)=",
        jnp.linalg.norm((df(u.module, t0, 1)[:, 0] - duej(t0)[:, 0]) / len(t0)),
        "f''(x)=",
        jnp.linalg.norm((df(u.module, t0, 2)[:, 0] - d2uej(t0)[:, 0]) / len(t0)),
    )


t0 = jnp.linspace(-1, 1, 1000)[:, None]
print_error(t0)

if "PYTEST" in os.environ:
    error = jnp.linalg.norm((u(t0) - uej(t0)[:, 0]) / len(t0))
    assert error < jnp.sqrt(ulp(1)), error
    sys.exit(0)
