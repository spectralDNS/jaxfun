# Solve Poisson's equation in 3D
import os

import jax.numpy as jnp
from scipy import sparse as scipy_sparse
from scipy.sparse import kron

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct, tpmats_to_scipy_kron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, n, ulp

M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, Legendre, bcs, scaling=n + 1, name="D", fun_str="psi")
T = TensorProduct((D, D, D), name="T")
v = TestFunction(T)
u = TrialFunction(T)

# Method of manufactured solution
x, y, z = T.system.base_scalars()
ue = (1 - x**2) * (1 - y**2) * (1 - z**2)

# A, b = inner(-Dot(Grad(u), Grad(v)) + v * Div(Grad(ue)), sparse=False)
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=False)

A0 = tpmats_to_scipy_kron(A)
uh = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

uj = T.backward(uh, kind="uniform", N=(20, 20, 20))
xj = T.mesh(kind="uniform", N=(20, 20, 20))
uej = lambdify((x, y, z), ue)(*xj)
error = jnp.linalg.norm(uj - uej)
if "pytest" in os.environ:
    assert error < ulp(1000), error
else:
    print("Error =", error)
