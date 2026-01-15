# ruff: noqa: E402
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.operators import Div, Grad
from jaxfun.pinns import FlaxFunction, Loss, Trainer
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.nnspaces import sPIKANSpace
from jaxfun.pinns.optimizer import adam, lbfgs
from jaxfun.utils import lambdify
from jaxfun.utils.common import Domain, ulp

domain = Domain(-jnp.pi, jnp.pi)
V = sPIKANSpace(4, [8], dims=1, rank=0, name="V", domains=[domain])
# V = KANMLPSpace(4, [16], dims=1, rank=0, name="V", domains=[domain])
w = FlaxFunction(V, name="w")

i = sp.symbols("i", integer=True)
x = V.system.x
ue = 5 + sp.summation(sp.sin(i * x), (i, 1, 3))

N = w.dim
mesh = Line(domain.lower, domain.upper)
xi = mesh.get_points(N, domain="inside", kind="random")
xb = mesh.get_points(N, domain="boundary")

eq = Div(Grad(w)) - Div(Grad(ue))

loss_fn = Loss((eq, xi), (w - ue, xb))
trainer = Trainer(loss_fn)

t0 = time.time()
opt_adam = adam(w, learning_rate=1e-3, end_learning_rate=1e-4, decay_steps=10000)
trainer.train(opt_adam, 1000, epoch_print=200, update_global_weights=10)
print(f"Adam time {time.time() - t0:.1f}s")

t1 = time.time()
opt_lbfgs = lbfgs(w, memory_size=20)
trainer.train(opt_lbfgs, 1000, epoch_print=100, update_global_weights=10)
print(f"L-BFGS time {time.time() - t1:.1f}s")

error = jnp.sqrt(loss_fn(w.module))
if "PYTEST" in os.environ:
    assert error < 1e-1, error
    sys.exit(1)

print(f"L2 Error {error:.2e} ({error / ulp(1):.1f} ulp)")
xj = jnp.linspace(domain.lower, domain.upper, 1000)
plt.plot(xj, lambdify(x, ue)(xj), label="Exact", lw=3)
plt.plot(xj, w(xj[:, None]), "--", label="Approx", lw=3)
plt.legend()
plt.title(f"L2 error {error:.2e}")
plt.show()
