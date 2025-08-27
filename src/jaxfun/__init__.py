from jaxfun import (
    Chebyshev as Chebyshev,
    Fourier as Fourier,
    Jacobi as Jacobi,
    Legendre as Legendre,
)
from jaxfun.arguments import (
    JAXFunction as JAXFunction,
    TestFunction as TestFunction,
    TrialFunction as TrialFunction,
)
from jaxfun.basespace import BaseSpace as BaseSpace, Domain as Domain
from jaxfun.coordinates import CoordSys as CoordSys, get_CoordSys as get_CoordSys
from jaxfun.functionspace import FunctionSpace as FunctionSpace
from jaxfun.inner import inner as inner
from jaxfun.operators import (
    Cross as Cross,
    Curl as Curl,
    Div as Div,
    Dot as Dot,
    Grad as Grad,
    Outer as Outer,
    cross as cross,
    curl as curl,
    divergence as divergence,
    dot as dot,
    gradient as gradient,
    outer as outer,
)
from jaxfun.tensorproductspace import (
    TensorProduct as TensorProduct,
    TensorProductSpace as TensorProductSpace,
    VectorTensorProductSpace as VectorTensorProductSpace,
)
from jaxfun.utils import common as common, fastgl as fastgl
