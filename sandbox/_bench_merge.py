import jax.numpy as jnp
import numpy as np
import time

from jaxfun.la import DiaMatrix
from jaxfun.galerkin import FunctionSpace, Legendre, TensorProduct, TestFunction, TrialFunction, inner
from jaxfun.galerkin.tensorproductspace import tpmats_to_kron

bcs = {"left": {"D": 0}, "right": {"D": 0}}
Dsp = FunctionSpace(10, Legendre.Legendre, bcs=bcs)
T = TensorProduct(Dsp, Dsp)
K = tpmats_to_kron(inner(TestFunction(T) * TrialFunction(T), sparse=True))
print(f"K shape: {K.shape}  n_diags: {len(K.offsets)}  offsets: {K.offsets}")

def old_merge(A, B):
    return DiaMatrix.from_dense(A.todense() + B.todense())

def numpy_once_merge(A, B, sign=1.0):
    np_dtype = np.result_type(A.data.dtype, B.data.dtype)
    a_np = np.asarray(A.data, dtype=np_dtype)
    b_np = np.asarray(B.data, dtype=np_dtype)
    self_map  = {off: i for i, off in enumerate(A.offsets)}
    other_map = {off: i for i, off in enumerate(B.offsets)}
    all_offsets = sorted(set(A.offsets) | set(B.offsets))
    m = a_np.shape[1]
    result_np = np.empty((len(all_offsets), m), dtype=np_dtype)
    for row_idx, k in enumerate(all_offsets):
        in_self, in_other = k in self_map, k in other_map
        if in_self and in_other:
            np.add(a_np[self_map[k]], sign * b_np[other_map[k]], out=result_np[row_idx])
        elif in_self:
            result_np[row_idx] = a_np[self_map[k]]
        else:
            result_np[row_idx] = sign * b_np[other_map[k]]
    keep = np.any(result_np != 0, axis=1)
    kept_offsets = tuple(int(k) for k, f in zip(all_offsets, keep) if f)
    return DiaMatrix(data=jnp.asarray(result_np[keep]), offsets=kept_offsets, shape=A.shape)

# warm-up
_ = old_merge(K, K); _ = numpy_once_merge(K, K); _ = K + K; _ = K + K

N = 500
print(f"\n{'Method':<32} {'total':>8}  {'ms/call':>9}")
print("-" * 55)

t0 = time.perf_counter()
for _ in range(N): old_merge(K, K)
t = time.perf_counter() - t0
print(f"{'old (dense roundtrip)':<32} {t:>8.3f}s  {t/N*1000:>8.3f}ms")

t0 = time.perf_counter()
for _ in range(N): numpy_once_merge(K, K)
t = time.perf_counter() - t0
print(f"{'numpy-once':<32} {t:>8.3f}s  {t/N*1000:>8.3f}ms")

t0 = time.perf_counter()
for _ in range(N): K + K   # same offsets → fast path
t = time.perf_counter() - t0
print(f"{'same-offsets fast path (K+K)':<32} {t:>8.3f}s  {t/N*1000:>8.3f}ms")

# different offsets case
from jaxfun.la import diags
A = diags([-jnp.ones(63), jnp.ones(64)], (-1, 0), shape=(64, 64))
B = diags([jnp.ones(64), -jnp.ones(63)], (0, 1), shape=(64, 64))

_ = A + B; _ = A + B  # warm-up
t0 = time.perf_counter()
for _ in range(N): A + B   # different offsets → general path
t = time.perf_counter() - t0
print(f"{'different-offsets gen path':<32} {t:>8.3f}s  {t/N*1000:>8.3f}ms")

