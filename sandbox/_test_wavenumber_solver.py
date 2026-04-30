"""
Test TPMatricesWavenumberSolver on a Fourier x Legendre Poisson problem.

Uses method of manufactured solutions:
  ue = cos(2*x) * (1 - y**2)   (proper real Fourier mode on [0, 2pi])

Checks:
  - max diff vs kron reference
  - L2 error (wavenumber solver direct)
  - L2 error (TPMatrices.solve dispatch)
"""
import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import (
    TensorProduct,
    TPMatrices,
    tpmats_lu_factor,
    tpmats_to_kron,
    tpmats_wavenumber_factor,
)
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify

bcs = {"left": {"D": 0}, "right": {"D": 0}}
F = FunctionSpace(16, Fourier)
D = FunctionSpace(16, Legendre, bcs)
T = TensorProduct(F, D)
v, u = TestFunction(T), TrialFunction(T)
x, y = T.system.base_scalars()

# Manufactured solution: cos(2x) * (1 - y^2)
# cos(2x) is a real Fourier mode on [0, 2pi]; satisfies Dirichlet BCs in y
ue = sp.cos(2 * x) * (1 - y**2)
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)

print(f"n_terms = {len(A)}")
for i, tp in enumerate(A):
    print(f"  term[{i}]: mats[0].offsets={tp.mats[0].offsets}  mats[1].offsets={tp.mats[1].offsets}")

# --- Direct wavenumber solver ---
wn = tpmats_wavenumber_factor(A)
uh = wn.solve(b)

# --- Kron reference ---
ref = tpmats_to_kron(A).solve(b.flatten()).reshape(b.shape)
print(f"\nmax diff vs kron:           {float(jnp.max(jnp.abs(uh - ref))):.2e}")

N = 50
uj = T.backward(uh, N=(N, N))
xj = T.mesh(N=(N, N), broadcast=True)
uej = lambdify((x, y), ue)(*xj)
print(f"L2 error (wavenumber):      {float(jnp.linalg.norm(uj - uej) / N):.2e}")

# --- TPMatrices.solve (auto-dispatch) ---
C = TPMatrices(A)
#uh2 = C.solve(b)
lu = tpmats_lu_factor(C.tpmats)
uh2 = lu.solve(b)
uj2 = T.backward(uh2, N=(N, N))
print(f"L2 error (TPMatrices.solve):{float(jnp.linalg.norm(uj2 - uej) / N):.2e}")
