"""
Benchmark TPMatricesWavenumberSolver.solve vs solve2.

solve  — custom jax.lax.scan kernels via _make_wavenumber_vmap_solve
solve2 — vmaps directly over a stacked LUFactors pytree
"""

import time

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_wavenumber_factor
from jaxfun.operators import Div, Grad


def bench(fn, rhs, n: int = 200) -> float:
    fn(rhs).block_until_ready()  # warm-up / compile
    t0 = time.perf_counter()
    for _ in range(n):
        fn(rhs).block_until_ready()
    return (time.perf_counter() - t0) / n * 1000


bcs = {"left": {"D": 0}, "right": {"D": 0}}

print(f"{'N':>6}  {'solve ms':>10}  {'solve2 ms':>10}  {'ratio':>8}")
print("-" * 44)

for N in [10, 20, 40, 80, 120, 200]:
    F = FunctionSpace(N, Fourier)
    D = FunctionSpace(N, Legendre, bcs)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)

    wn = tpmats_wavenumber_factor(A)

    # Correctness check
    diff = float(jnp.max(jnp.abs(wn.solve(b) - wn.solve2(b))))
    assert diff < 1e-12, f"N={N}: solve vs solve2 max diff = {diff:.2e}"

    t1 = bench(wn.solve, b)
    t2 = bench(wn.solve2, b)
    print(f"{N:>6}  {t1:>10.3f}  {t2:>10.3f}  {t2 / t1:>7.2f}x")
