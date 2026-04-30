import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from jaxfun.coordinates import x, y
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import n
from jaxfun.la import DiaMatrix
from jaxfun.la.diamatrix import diakron
from jaxfun.utils.common import eliminate_near_zeros

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

# Check the FIRST term only
tol = 100
tpm = A[0]
m0 = tpm.mats[0]
m1 = tpm.mats[1]
print(f'Term 0: scale={tpm.scale}')
print(f'  m0 type={type(m0).__name__}, shape={m0.todense().shape}')
print(f'  m1 type={type(m1).__name__}, shape={m1.todense().shape}')

# Clean factor versions
m0_clean_dense = np.array(eliminate_near_zeros(m0.todense(), tol))
m1_clean_dense = np.array(eliminate_near_zeros(m1.todense(), tol))

m0_dia = DiaMatrix.from_dense(jnp.array(m0_clean_dense))
m1_dia = DiaMatrix.from_dense(jnp.array(m1_clean_dense))

# DIA kron of clean factors
kron_dia = diakron(m0_dia, m1_dia)
# numpy kron reference
kron_np = np.kron(m0_clean_dense, m1_clean_dense)

diff = np.max(np.abs(np.array(kron_dia.todense()) - kron_np))
print(f'  diakron vs np.kron diff: {diff:.3e}')
print(f'  m0 offsets: {m0_dia.offsets}')
print(f'  m1 offsets: {m1_dia.offsets}')
print(f'  result offsets (first 5): {kron_dia.offsets[:5]}...{kron_dia.offsets[-5:]}')

# Also check data directly
if diff > 0:
    # Find where they differ
    dense_dia = np.array(kron_dia.todense())
    mask = np.abs(dense_dia - kron_np) > 1e-10
    rows, cols = np.where(mask)
    print(f'  Differing entries: {len(rows)}')
    if len(rows) > 0:
        for r, c in zip(rows[:5], cols[:5]):
            print(f'    [{r},{c}]: dia={dense_dia[r,c]:.6e}, np={kron_np[r,c]:.6e}')
