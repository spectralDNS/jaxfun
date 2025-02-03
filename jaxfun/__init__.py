from jaxfun import Chebyshev, Jacobi, Legendre
from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.Basespace import BaseSpace
from jaxfun.composite import Composite
from jaxfun.coordinates import CoordSys, get_CoordSys
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.operators import (
    Cross,
    Curl,
    Div,
    Dot,
    Grad,
    cross,
    curl,
    divergence,
    dot,
    gradient,
)
from jaxfun.tensorproductspace import (
    TensorProduct,
    TensorProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.utils import common, fastgl
