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
from scipy import sparse as scipy_sparse
from typing import cast

ue = (x - x**2)**2 * (x - y**2)**2
M = 20
bcsx = {'left': {'D': ue.subs(x,-1),'N': ue.diff(x,1).subs(x,-1)},'right': {'D': ue.subs(x,1),'N': ue.diff(x,1).subs(x,1)}}
bcsy = {'left': {'D': ue.subs(y,-1),'N': ue.diff(y,1).subs(y,-1)},'right': {'D': ue.subs(y,1),'N': ue.diff(y,1).subs(y,1)}}
Dx = FunctionSpace(M, Chebyshev, scaling=n+1, bcs=bcsx, name="Dx", fun_str="psi")
Dy = FunctionSpace(M, Chebyshev, scaling=n+1, bcs=bcsy, name="Dy", fun_str="phi")
T = TensorProduct(Dx, Dy, name="T")
v, u = TestFunction(T, name="v"), TrialFunction(T, name="u")
ue_sym = T.system.expr_psi_to_base_scalar(ue)
A, b = inner(Div(Grad(Div(Grad(u))))*v - Div(Grad(Div(Grad(ue_sym))))*v, sparse=True)

C = tpmats_to_kron(A)
print(f'C shape={C.shape}')

# Build scipy sparse version of same matrix
spmats = []
for tpm in A:
    s0 = scipy_sparse.csr_matrix(np.array(tpm.mats[0].todense() if isinstance(tpm.mats[0], DiaMatrix) else cast(Matrix, tpm.mats[0]).data))
    for m in tpm.mats[1:]:
        s1 = scipy_sparse.csr_matrix(np.array(m.todense() if isinstance(m, DiaMatrix) else cast(Matrix, m).data))
        s0 = scipy_sparse.kron(s0, s1, format='csr')
    s0 = s0 * float(tpm.scale)
    spmats.append(s0)
C_scipy = sum(spmats)

b_flat = np.array(b.flatten())
x_jax   = np.array(C.solve(b_flat))
x_scipy = scipy_sparse.linalg.spsolve(C_scipy, b_flat)

N = 100
xj = T.mesh(kind="uniform", N=(N, N))
ue_fn = lambdify((x, y), ue)

uh_jax   = T.backward(jnp.array(x_jax).reshape(b.shape), kind="uniform", N=(N, N))
uh_scipy = T.backward(jnp.array(x_scipy).reshape(b.shape), kind="uniform", N=(N, N))
uej = ue_fn(*xj)

err_jax   = float(jnp.linalg.norm(uh_jax - uej)) / N
err_scipy = float(jnp.linalg.norm(uh_scipy - uej)) / N

print(f'error C.solve:             {err_jax:.3e}')
print(f'error scipy spsolve:       {err_scipy:.3e}')
print(f'ulp(C.data.max()):         {float(ulp(C.data.max())):.3e}')
print(f'max diff C.solve vs scipy: {np.max(np.abs(x_jax - x_scipy)):.3e}')
print(f'max diff matrices:         {np.max(np.abs(np.array(C.todense()) - C_scipy.toarray())):.3e}')
