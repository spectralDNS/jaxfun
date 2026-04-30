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

tol = 100

def _to_dia(mat):
    dense = eliminate_near_zeros(mat.todense(), tol)
    return DiaMatrix.from_dense(jnp.array(dense))

# Accumulate term by term, comparing DIA vs numpy at each step
result_np = None
result_dia = None

for i, tpm in enumerate(A):
    d0 = np.array(eliminate_near_zeros(tpm.mats[0].todense(), tol))
    d1 = np.array(eliminate_near_zeros(tpm.mats[1].todense(), tol))
    term_np = np.kron(d0, d1) * float(tpm.scale)

    m0_dia = DiaMatrix.from_dense(jnp.array(d0))
    m1_dia = DiaMatrix.from_dense(jnp.array(d1))
    term_dia = diakron(m0_dia, m1_dia) * jnp.asarray(tpm.scale)

    single_diff = np.max(np.abs(np.array(term_dia.todense()) - term_np))

    result_np = term_np if result_np is None else result_np + term_np
    result_dia = term_dia if result_dia is None else result_dia + term_dia

    accum_diff = np.max(np.abs(np.array(result_dia.todense()) - result_np))
    print(f'Term {i}: single_diff={single_diff:.2e}, accum_diff={accum_diff:.2e}, n_offsets_dia={len(result_dia.offsets)}')
    if accum_diff > 1e-6 and i > 0:
        print(f'  *** DIVERGED at term {i} ***')
        prev_offsets = set(prev_result_dia.offsets)
        new_offsets = set(term_dia.offsets)
        print(f'  prev offsets count={len(prev_offsets)}, new offsets count={len(new_offsets)}')
        print(f'  intersection size={len(prev_offsets & new_offsets)}')
        print(f'  union size={len(prev_offsets | new_offsets)}')
        break

    prev_result_dia = result_dia
