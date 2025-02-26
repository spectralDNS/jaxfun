from jaxfun import Chebyshev, Jacobi, Legendre, Fourier
from jaxfun.arguments import JAXFunction, TestFunction, TrialFunction
from jaxfun.Basespace import BaseSpace, Domain
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
