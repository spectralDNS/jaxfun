import jax; jax.config.update('jax_enable_x64', True)
import numpy as np
from jaxfun.coordinates import x, y
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import n
from jaxfun.la import DiaMatrix, Matrix
from jaxfun.utils.common import eliminate_near_zeros
from typing import cast
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

# Check factor matrices before/after eliminate_near_zeros
print(f'Number of tensor product terms: {len(A)}')
for i, tpm in enumerate(A):
    for j, m in enumerate(tpm.mats):
        if isinstance(m, DiaMatrix):
            raw = np.array(m.todense())
        else:
            raw = np.array(cast(Matrix, m).data)
        cleaned = eliminate_near_zeros(raw.copy(), 100)
        diff = np.max(np.abs(raw - cleaned))
        print(f'Term {i}, mat {j}: shape={raw.shape}, max_val={np.max(np.abs(raw)):.3e}, eliminated_entries={np.sum(raw != cleaned)}, max_diff={diff:.3e}')
