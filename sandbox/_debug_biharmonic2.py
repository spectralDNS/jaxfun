import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from jaxfun.coordinates import x, y
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp
from jaxfun.la import DiaMatrix, Matrix
from typing import cast

ue_orig = (x - x**2)**2 * (x - y**2)**2
M = 20
bcsx = {'left': {'D': ue_orig.subs(x,-1),'N': ue_orig.diff(x,1).subs(x,-1)},'right': {'D': ue_orig.subs(x,1),'N': ue_orig.diff(x,1).subs(x,1)}}
bcsy = {'left': {'D': ue_orig.subs(y,-1),'N': ue_orig.diff(y,1).subs(y,-1)},'right': {'D': ue_orig.subs(y,1),'N': ue_orig.diff(y,1).subs(y,1)}}
Dx = FunctionSpace(M, Chebyshev, scaling=n+1, bcs=bcsx, name="Dx", fun_str="psi")
Dy = FunctionSpace(M, Chebyshev, scaling=n+1, bcs=bcsy, name="Dy", fun_str="phi")
T = TensorProduct(Dx, Dy, name="T")
v, u = TestFunction(T, name="v"), TrialFunction(T, name="u")

# ue_sym is the expression in basis function coordinates
ue_sym = T.system.expr_psi_to_base_scalar(ue_orig)
print("ue_orig:", ue_orig)
print("ue_sym :", ue_sym)
print("Same?  :", ue_orig == ue_sym)

A, b = inner(Div(Grad(Div(Grad(u))))*v - Div(Grad(Div(Grad(ue_sym))))*v, sparse=True)

C = tpmats_to_kron(A)
print(f'C shape={C.shape}, b.shape={b.shape}')

uh_flat = C.solve(b.flatten())
uh = uh_flat.reshape(b.shape)

# Residual in coefficient space
Cuh = C.todense() @ uh_flat
residual = float(jnp.linalg.norm(Cuh - b.flatten())) / float(jnp.linalg.norm(b.flatten()))
print(f'relative residual: {residual:.3e}')

# Check uh
print(f'uh max: {float(jnp.max(jnp.abs(uh))):.3e}')
print(f'uh shape: {uh.shape}')

# Evaluate at uniform grid
N = 100
xj = T.mesh(kind="uniform", N=(N, N))
uj = T.backward(uh, kind="uniform", N=(N, N))  # backward transform with uh coefficients

ue_fn_orig = lambdify((x, y), ue_orig)
ue_fn_sym  = lambdify((x, y), ue_sym)
uej_orig   = ue_fn_orig(*xj)
uej_sym    = ue_fn_sym(*xj)

err_vs_orig = float(jnp.linalg.norm(uj - uej_orig)) / N
err_vs_sym  = float(jnp.linalg.norm(uj - uej_sym)) / N

print(f'error vs original ue:     {err_vs_orig:.3e}')
print(f'error vs transformed ue_sym: {err_vs_sym:.3e}')
print(f'diff orig vs sym at grid: {float(jnp.linalg.norm(uej_orig - uej_sym)) / N:.3e}')
