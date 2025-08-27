# ruff: noqa: F401

from jaxfun import Chebyshev, Fourier, Jacobi, Legendre
from jaxfun.arguments import JAXFunction, TestFunction, TrialFunction
from jaxfun.basespace import BaseSpace, Domain
from jaxfun.coordinates import CoordSys, get_CoordSys
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.operators import (
    Cross,
    Curl,
    Div,
    Dot,
    Grad,
    Outer,
    cross,
    curl,
    divergence,
    dot,
    gradient,
    outer,
)
from jaxfun.tensorproductspace import (
    TensorProduct,
    TensorProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.utils import common, fastgl
