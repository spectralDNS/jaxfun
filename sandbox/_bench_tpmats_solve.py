"""Benchmark TPMatrices.solve: diagonalization vs explicit Kronecker product.

Run with:
    uv run python sandbox/_bench_tpmats_solve.py
"""

import time

import jax

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as space
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import (
    TensorProduct,
    tpmats_lu_factor,
    tpmats_to_kron,
)
from jaxfun.operators import Div, Grad

REPEATS = 200

print(f"{'N':>6}  {'old (kron) ms':>15}  {'new (diag) ms':>15}  {'speedup':>8}")
print("-" * 55)

for M in [10, 20, 40, 80, 120]:
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    Dx = FunctionSpace(M, space, bcs)
    Dy = FunctionSpace(M+4, space, bcs)

    T = TensorProduct(Dx, Dy)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()

    ue = (1 - x**2) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) + u*v - v * Div(Grad(ue)) - v*ue, sparse=True)
    rhs = b.flatten()

    # Build old solver (explicit Kronecker product)
    C = tpmats_to_kron(A)
    jax.block_until_ready(C.solve(rhs))

    # Build new solver (diagonalization)
    lu = tpmats_lu_factor(A)
    jax.block_until_ready(lu.solve(rhs))

    # Time old
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(C.solve(rhs))
    old_ms = (time.perf_counter() - t0) / REPEATS * 1000

    # Time new
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(lu.solve(rhs))
    new_ms = (time.perf_counter() - t0) / REPEATS * 1000

    N = M - 2  # interior DOFs per axis
    print(f"{N:>6}  {old_ms:>15.3f}  {new_ms:>15.3f}  {old_ms / new_ms:>8.1f}x")
