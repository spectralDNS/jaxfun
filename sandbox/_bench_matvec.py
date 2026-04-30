"""Speed comparison: DiaMatrix vs Matrix for matvec and matmat.

Measures wall-clock time (after JIT warm-up) for:
  - matvec :  A @ x          (n,) → (n,)
  - matvec2:  A @ X          (n, k) → (n, k)   [column batch, k = 32]
  - rmatvec:  x @ A          (n,) → (n,)
  - matmat :  A @ B          (n, n) → (n, n)    [DIA×DIA vs dense×dense]

Matrices are symmetric tridiagonal (-1, 2, -1) for DiaMatrix and the
equivalent dense array for Matrix, at sizes n = 64, 128, 256, 512, 1024.
"""

import time

import jax
import jax.numpy as jnp

from jaxfun.la import DiaMatrix
from jaxfun.la.diamatrix import diags
from jaxfun.la.matrix import Matrix

jax.config.update("jax_enable_x64", False)

SIZES = [64, 128, 256, 512, 1024]
BATCH = 32    # number of columns for the batched matvec
REPEATS = 200


def make_tridiag(n: int) -> tuple[DiaMatrix, Matrix]:
    A = diags(
        [-jnp.ones(n - 1), 2 * jnp.ones(n), -jnp.ones(n - 1)],
        offsets=(-1, 0, 1),
        shape=(n, n),
    )
    M = Matrix(A.todense())
    return A, M


def bench(fn, repeats: int) -> float:
    """Return wall-clock seconds per call (after one warm-up call)."""
    fn()                                # warm-up / JIT compile
    jax.block_until_ready(fn())        # second call to prime caches
    t0 = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / repeats


def fmt(seconds: float) -> str:
    us = seconds * 1e6
    return f"{us:>9.1f} µs"


def speedup(dia_s: float, mat_s: float) -> str:
    ratio = mat_s / dia_s
    return f"{ratio:>6.2f}x"


print(f"\n{'Operation':<16} {'n':>6}  {'DiaMatrix':>12}  {'Matrix':>12}  {'DIA speedup':>12}")
print("-" * 68)

for n in SIZES:
    A, M = make_tridiag(n)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n,))
    X = jax.random.normal(key, (n, BATCH))

    # --- matvec: A @ x ---
    t_dia = bench(lambda: A @ x, REPEATS)
    t_mat = bench(lambda: M @ x, REPEATS)
    print(f"{'A @ x':<16} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}")

    # --- batched matvec: A @ X  (n x k) ---
    t_dia = bench(lambda: A @ X, REPEATS)
    t_mat = bench(lambda: M @ X, REPEATS)
    print(f"{'A @ X (k=32)':<16} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}")

    # --- rmatvec: x @ A ---
    t_dia = bench(lambda: x @ A, REPEATS)
    t_mat = bench(lambda: x @ M, REPEATS)
    print(f"{'x @ A':<16} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}")

    # --- matmat: A @ A ---
    t_dia = bench(lambda: A @ A, REPEATS)
    t_mat = bench(lambda: M @ M, REPEATS)
    print(f"{'A @ A':<16} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}")

    print()
