"""
Compare: old Python-for vs new fori_loop LU kernel compile times.
"""
import jax
import jax.numpy as jnp
import functools
import time


# ── OLD kernel (Python for-loops) ────────────────────────────────────────────
@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _lu_old(band, p, q, center):
    n = band.shape[1]
    def elim_step(band, k):
        k = k.astype(int)
        pivot = band[center, k]
        for s in range(1, p + 1):
            in_i = (k + s) < n
            factor = jnp.where(in_i, band[center - s, k] / pivot, 0.0)
            band = band.at[center - s, k].set(jnp.where(in_i, factor, band[center - s, k]))
            for u in range(1, q + 1):
                j = k + u; in_j = j < n
                safe_j = jnp.where(in_j, j, 0)
                band = band.at[center + u - s, safe_j].add(
                    jnp.where(in_i & in_j, -factor * band[center + u, safe_j], 0.0))
        return band, None
    band_lu, _ = jax.lax.scan(elim_step, band, jnp.arange(n, dtype=int))
    return band_lu


# ── NEW kernel (fori_loop) ────────────────────────────────────────────────────
from jaxfun.la.diamatrix import _lu_banded_no_pivot_kernel as _lu_new


print(f"{'p':>4} {'q':>4} {'old compile':>14} {'new compile':>14} {'speedup':>10}")
print("-" * 52)

for p, q in [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (12, 12)]:
    n = 60
    center = p
    band = jnp.zeros((p + q + 1, n)).at[center].set(2.0).at[center+1, 1:].set(-1.0).at[center-1, :-1].set(-1.0)

    # old
    t0 = time.perf_counter()
    _lu_old(band, p, q, center).block_until_ready()
    t_old = time.perf_counter() - t0

    # new (different static args → fresh compile)
    t0 = time.perf_counter()
    _lu_new(band, p, q, center).block_until_ready()
    t_new = time.perf_counter() - t0

    print(f"{p:>4} {q:>4} {t_old:>13.3f}s {t_new:>13.3f}s {t_old/t_new:>9.1f}x")
