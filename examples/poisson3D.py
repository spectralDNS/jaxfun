# Solve Poisson's equation in 3D
import os
from scipy import sparse as scipy_sparse
from scipy.sparse import kron
import jax.numpy as jnp
from jaxfun.utils.common import lambdify
from jaxfun.Legendre import Legendre
from jaxfun.Chebyshev import Chebyshev
from jaxfun.inner import inner
from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.operators import Grad, Div, Dot
from jaxfun.Basespace import n
from jaxfun.utils.common import ulp
from jaxfun.functionspace import FunctionSpace
from jaxfun.tensorproductspace import TensorProductSpace, tpmats_to_scipy_sparse_list


M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, Legendre, bcs, scaling=n + 1, name="D", fun_str="psi")
T = TensorProductSpace((D, D, D), name="T")
v = TestFunction(T)
u = TrialFunction(T)

# Method of manufactured solution
x, y, z = T.system.base_scalars()
ue = (1 - x**2) * (1 - y**2) * (1 - z**2)

# A, b = inner(-Dot(Grad(u), Grad(v)) + v * Div(Grad(ue)), sparse=False)
A, b = inner(v * Div(Grad(u)) + v * Div(Grad(ue)), sparse=False)

a = tpmats_to_scipy_sparse_list(A)
A0 = (
    kron(kron(a[0], a[1]), a[2])
    + kron(kron(a[3], a[4]), a[5])
    + kron(kron(a[6], a[7]), a[8])
)
un = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

uj = T.backward(un, kind="uniform", N=(20, 20, 20))
xj = T.mesh(kind="uniform", N=(20, 20, 20))
uej = lambdify((x, y, z), ue)(*xj)
error = jnp.linalg.norm(uj - uej)
if 'pytest' in os.environ:
    assert error < ulp(1000), error
else:
    print("Error =", error)
