import jax.numpy as jnp
from jaxfun.la import diags, DiaMatrix
import numpy as np
import time

# Correctness tests
A = diags([-jnp.ones(4), 2*jnp.ones(5), -jnp.ones(4)], (-1, 0, 1))
B = diags([jnp.ones(3), -jnp.ones(5), jnp.ones(3)], (-2, 0, 2))

C = A + B
C2 = DiaMatrix.from_dense(A.todense() + B.todense())
print("max diff add:", float(jnp.max(jnp.abs(C.todense() - C2.todense()))))

D_ = A - B
D2 = DiaMatrix.from_dense(A.todense() - B.todense())
print("max diff sub:", float(jnp.max(jnp.abs(D_.todense() - D2.todense()))))

Z = A - A
print("A-A offsets:", Z.offsets)

# Timing on a larger problem
from jaxfun.galerkin import FunctionSpace, Legendre, TensorProduct, TestFunction, TrialFunction, inner
from jaxfun.galerkin.tensorproductspace import tpmats_to_kron

bcs = {"left": {"D": 0}, "right": {"D": 0}}
Dsp = FunctionSpace(10, Legendre.Legendre, bcs=bcs)
T = TensorProduct(Dsp, Dsp)
u = TrialFunction(T)
v = TestFunction(T)
A2 = inner(v * u, sparse=True)
K = tpmats_to_kron(A2)
print("K shape:", K.shape, "offsets:", K.offsets)

# Warm-up
_ = K + K

N = 50
t0 = time.perf_counter()
for _ in range(N):
    R = K + K
print(f"DIA add x{N}: {time.perf_counter()-t0:.3f}s")
