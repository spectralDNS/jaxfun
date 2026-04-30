"""Speed comparison: DiaMatrix vs Matrix for linear solves.

Measures wall-clock time (after JIT warm-up + LU pre-factorisation) for:
  - solve vec :  A.solve(x)        (n,) → (n,)
  - solve mat :  A.solve(X)        (n, k) → (n, k)   [k = 32 columns]

DiaMatrix uses banded LU (Thomas-algorithm style scan).
Matrix uses jnp.linalg.solve (dense LAPACK via XLA).

The factorisation is pre-computed once before timing so we measure only
the forward/backward substitution cost (the hot-path in iterative solvers).

Matrices are symmetric tridiagonal (-1, 2, -1) — diagonally dominant,
so no pivoting is needed.  Sizes: n = 64, 128, 256, 512, 1024.
"""

import time

import jax
import jax.numpy as jnp

from jaxfun.la import DiaMatrix
from jaxfun.la.diamatrix import diags
from jaxfun.la.matrix import Matrix

jax.config.update("jax_enable_x64", False)

SIZES = [64, 128, 256, 512, 1024]
BATCH = 32
REPEATS = 200


def make_tridiag(n: int) -> tuple[DiaMatrix, Matrix]:
    A = diags(
        [-jnp.ones(n - 1), 2 * jnp.ones(n), -jnp.ones(n - 1)],
        offsets=(-1, 0, 1),
        shape=(n, n),
    )
    M = Matrix(A.todense())
    return A, M


def make_diag(n: int) -> tuple[DiaMatrix, Matrix]:
    A = diags([2 * jnp.ones(n)], offsets=(0,), shape=(n, n))
    M = Matrix(A.todense())
    return A, M


def bench(fn, repeats: int) -> float:
    """Return wall-clock seconds per call (after two warm-up calls)."""
    fn()
    jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / repeats


def fmt(seconds: float) -> str:
    us = seconds * 1e6
    return f"{us:>9.1f} µs"


def speedup(dia_s: float, mat_s: float) -> str:
    return f"{mat_s / dia_s:>6.2f}x"


# ── header ────────────────────────────────────────────────────────────────────
print(
    f"\n{'Operation':<20} {'n':>6}  {'DiaMatrix':>12}  {'Matrix':>12}  {'DIA speedup':>12}"
)
print("-" * 72)

for n in SIZES:
    A, M = make_tridiag(n)
    lu_dia = A.lu_factor()  # pre-factored; timing measures solve only
    lu_mat = M.lu_factor()

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n,))
    X = jax.random.normal(key, (n, BATCH))

    # ── solve vector ──────────────────────────────────────────────────────────
    t_dia = bench(lambda: lu_dia.solve(x), REPEATS)
    t_mat = bench(lambda: lu_mat.solve(x), REPEATS)
    print(
        f"{'solve (pre-LU) vec':<20} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}"
    )

    # ── solve matrix (k=32 cols) ──────────────────────────────────────────────
    t_dia = bench(lambda: lu_dia.solve(X), REPEATS)
    t_mat = bench(lambda: lu_mat.solve(X), REPEATS)
    print(
        f"{'solve (pre-LU) mat':<20} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}"
    )

    # ── full solve (factorisation included) ───────────────────────────────────
    t_dia = bench(lambda: A.solve(x), REPEATS)
    t_mat = bench(lambda: M.solve(x), REPEATS)
    print(
        f"{'solve (full) vec':<20} {n:>6}  {fmt(t_dia)}  {fmt(t_mat)}  {speedup(t_dia, t_mat)}"
    )

    print()
