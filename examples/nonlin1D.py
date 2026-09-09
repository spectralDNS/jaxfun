# Solve Helmholtz' equation
import os
import sys

import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.operators import Div, Grad
from jaxfun.pinns import FlaxFunction, Loss, Trainer, adam, lbfgs
from jaxfun.pinns.mesh import Line
from jaxfun.utils.common import Domain, lambdify, ulp

x = sp.Symbol("x", real=True)
N = 60
# Method of manufactured solution
ue = sp.exp(sp.cos(2 * sp.pi * x))

domain = Domain(-1, 1)
bcs = {
    "left": {"D": float(ue.subs(x, domain[0]))},
    "right": {"D": float(ue.subs(x, domain[1]))},
}
D = FunctionSpace(N, Chebyshev, bcs=bcs, name="D", fun_str="psi", domain=domain)

u = FlaxFunction(D, name="u", rngs=nnx.Rngs(101))
ue = D.system.expr_psi_to_base_scalar(ue)

N = 1000
mesh = Line(float(domain.lower), float(domain.upper), key=nnx.Rngs(1001)())

xj = mesh.get_points(N, domain="inside", kind="legendre")
wj = mesh.get_weights(N, domain="inside", kind="legendre")

eq1 = (Div(Grad(u)) + u * u) - (Div(Grad(ue)) + ue * ue)

loss_fn = Loss((eq1, xj, 0, wj))

trainer = Trainer(loss_fn)

opt_adam = adam(u, learning_rate=1e-3)
trainer.train(opt_adam, 2000, epoch_print=1000)
opt_lbfgs = lbfgs(u, memory_size=100, max_linesearch_steps=10)
trainer.train(opt_lbfgs, 1000, epoch_print=100, abs_limit_loss=float(ulp(1)) * 1)

uej = lambdify(x, ue)
t0 = jnp.linspace(-1, 1, 1000)[:, None]
error = jnp.linalg.norm((u(t0) - uej(t0)[:, 0]) / len(t0))
print("Error =", error)

if "PYTEST" in os.environ:
    assert error < jnp.sqrt(ulp(1)), error
    sys.exit(0)
