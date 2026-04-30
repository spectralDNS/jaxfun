import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from jaxfun.coordinates import x, y
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_kron, tpmats_to_scipy_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import n, lambdify, ulp
from scipy import sparse as scipy_sparse

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

print(f'Number of terms: {len(A)}')
for i, tpm in enumerate(A):
    print(f'  Term {i}: scale={tpm.scale}, mats shapes={[m.todense().shape for m in tpm.mats]}')

# Build scipy kron with scale applied manually
result_scipy_scaled = None
for tpm in A:
    d0 = np.array(tpm.mats[0].todense())
    d1 = np.array(tpm.mats[1].todense())
    term = scipy_sparse.csc_array(np.kron(d0, d1) * float(tpm.scale))
    result_scipy_scaled = term if result_scipy_scaled is None else result_scipy_scaled + term

C_dia = tpmats_to_kron(A)
C_sci_nosc = tpmats_to_scipy_kron(A)

b_flat = np.array(b.flatten())

print(f'\nmax diff dia vs scipy-no-scale: {np.max(np.abs(np.array(C_dia.todense()) - C_sci_nosc.toarray())):.3e}')
print(f'max diff dia vs scipy-with-scale: {np.max(np.abs(np.array(C_dia.todense()) - result_scipy_scaled.toarray())):.3e}')

uh_dia = C_dia.solve(jnp.array(b_flat)).reshape(b.shape)
uh_sci = jnp.array(scipy_sparse.linalg.spsolve(C_sci_nosc, b_flat).reshape(b.shape))

N = 100
xj = T.mesh(kind="uniform", N=(N, N))
uej = lambdify((x, y), ue)(*xj)
err_dia = float(jnp.linalg.norm(T.backward(uh_dia, kind="uniform", N=(N,N)) - uej)) / N
err_sci = float(jnp.linalg.norm(T.backward(uh_sci, kind="uniform", N=(N,N)) - uej)) / N
print(f'\nerror DIA path:           {err_dia:.3e}')
print(f'error scipy-no-scale:     {err_sci:.3e}')
