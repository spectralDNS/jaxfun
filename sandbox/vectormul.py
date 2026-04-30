import sympy as sp

from jaxfun.galerkin import (
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    VectorTensorProductSpace,
    inner,
)
from jaxfun.galerkin.tensorproductspace import BlockTPMatrix
from jaxfun.operators import Dot

D = FunctionSpace(4, Legendre.Legendre, name="D")
T = TensorProduct(D, D, name="T")
V = VectorTensorProductSpace(T, name="V")
u = TrialFunction(V, name="u")
v = TestFunction(V, name="v")
i, j = T.system.base_vectors()
h = Dot(v, i) * j + Dot(v, j) * i
A = inner(Dot(h, u), sparse=True)
B = BlockTPMatrix(A, V, V)