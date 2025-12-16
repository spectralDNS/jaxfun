# ruff: noqa: E402
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxfun import Div, Grad
from jaxfun.galerkin import Chebyshev, TensorProduct
from jaxfun.pinns import FlaxFunction, Loss, Trainer
from jaxfun.pinns.mesh import Lshape
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs

# V = MLPSpace([30, 30], dims=2, rank=0, name="V")
# V = sPIKANSpace(5, [8], dims=2, rank=0, name="V")
# V = KANMLPSpace(5, [12, 12], dims=2, rank=0, name="V")
C = Chebyshev.Chebyshev(36, name="C")
V = TensorProduct(C, C, name="V")

w = FlaxFunction(V, name="w")

# mesh = Square_with_hole()
# mesh = Circle_with_hole()
mesh = Lshape()
xi = mesh.get_points_inside_domain(3000)
xb = mesh.get_points_on_domain(600, corners=True)

x, y = V.system.base_scalars()

f = Div(Grad(w)) - 1

loss_fn = Loss((f, xi, 0), (w, xb, 0))
trainer = Trainer(loss_fn)

t0 = time.time()

opt_adam = adam(w, learning_rate=1e-3, end_learning_rate=1e-4, decay_steps=2000)
trainer.train(opt_adam, 1000, epoch_print=100, update_global_weights=-100)
print("Time for Adam:", time.time() - t0)

t1 = time.time()
opt_lbfgs = lbfgs(w, memory_size=100)
trainer.train(opt_lbfgs, 1000, epoch_print=100, update_global_weights=-100)
print("Time for LBFGS:", time.time() - t1)

t2 = time.time()
opt_hess = GaussNewton(w, use_lstsq=True, use_GN=True)
trainer.train(opt_hess, 4, epoch_print=1, abs_limit_change=1e-8)
print("Time for Gauss-Newton:", time.time() - t2)

print("time", time.time() - t0)

X_all = jnp.vstack((xi, xb))
w_all = w(X_all)
mesh.plot_solution(X_all, w_all, xb=xb, levels=30)
