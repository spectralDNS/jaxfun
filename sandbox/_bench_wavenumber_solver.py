"""Benchmark three solve methods for a Fourier x Legendre Poisson problem.

Methods compared:
  kron      — explicit Kronecker product (tpmats_to_kron)
  wavenumber — per-wavenumber banded DIA solver (tpmats_wavenumber_factor)
  tpmat     — TPMatrices.solve (auto-dispatches to wavenumber)

Run with:
    uv run python sandbox/_bench_wavenumber_solver.py
"""

import time

import jax
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

REPEATS = 200
bcs = {"left": {"D": 0}, "right": {"D": 0}}

print(f"{'N':>6}  {'kron ms':>12}  {'wavenumber ms':>14}  {'tpmat ms':>10}  {'kron/wn':>9}  {'kron/tp':>9}")
print("-" * 72)

for M in [10, 20, 40, 80, 120, 160, 200]:
    F = FunctionSpace(M, Fourier)
    D = FunctionSpace(M, Legendre, bcs)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)

    # --- build & warm up ---
    kron = tpmats_to_kron(A)
    jax.block_until_ready(kron.solve(b.flatten()))

    wn = tpmats_wavenumber_factor(A)
    jax.block_until_ready(wn.solve(b))

    #tp = TPMatrices(A)
    tp = tpmats_lu_factor(TPMatrices(A).tpmats)
    jax.block_until_ready(tp.solve(b))

    # --- time kron ---
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(kron.solve(b.flatten()))
    kron_ms = (time.perf_counter() - t0) / REPEATS * 1000

    # --- time wavenumber ---
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(wn.solve(b))
    wn_ms = (time.perf_counter() - t0) / REPEATS * 1000

    # --- time TPMatrices.solve ---
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        jax.block_until_ready(tp.solve(b))
    tp_ms = (time.perf_counter() - t0) / REPEATS * 1000

    print(
        f"{M:>6}  {kron_ms:>12.3f}  {wn_ms:>14.3f}  {tp_ms:>10.3f}"
        f"  {kron_ms / wn_ms:>9.1f}x  {kron_ms / tp_ms:>9.1f}x"
    )
