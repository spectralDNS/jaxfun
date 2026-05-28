# Solve Poisson's equation in 3D with two periodic directions
import os
import sys
import time

import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction, x, y, z
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

ue = (
    (1 - z**2)
    * (sp.cos(2 * x) * sp.cos(3 * y) + sp.sin(x - 2 * y))
    * sp.exp(sp.cos(sp.pi * z))
)

M, N = 64, 16
bcs = {"left": {"D": ue.subs(z, -1)}, "right": {"D": ue.subs(z, 1)}}
D = FunctionSpace(M, Chebyshev, bcs, scaling=n + 1, name="D", fun_str="psi")
F0 = FunctionSpace(N, Fourier, name="F0", fun_str="E")
F1 = FunctionSpace(N, Fourier, name="F1", fun_str="E")
T = TensorProduct(F0, F1, D, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
x, y, z = T.system.base_scalars()
ue = T.system.expr_psi_to_base_scalar(ue)

timings = {}

start = time.perf_counter()
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True, kind="system")
jax.block_until_ready(b)
timings["inner"] = time.perf_counter() - start

start = time.perf_counter()
uh = A.solve(b)
jax.block_until_ready(uh)
timings["solve"] = time.perf_counter() - start

N_eval = (2 * N, 2 * N, M)
start = time.perf_counter()
uj = T.backward(uh, N=N_eval)
jax.block_until_ready(uj)
timings["evaluate"] = time.perf_counter() - start

xj = T.mesh(N=N_eval, broadcast=True)
uej = lambdify((x, y, z), ue)(*xj)

error = jnp.linalg.norm(uj - uej) / jnp.sqrt(uj.size)
if "PYTEST" in os.environ:
    assert error < ulp(1000), error
    sys.exit(0)

print("Error =", error)
print("Timings:")
print(f"  inner:    {timings['inner']:.6f} s")
print(f"  solve:    {timings['solve']:.6f} s")
print(f"  evaluate: {timings['evaluate']:.6f} s")
