from jaxfun.typing import MeshKind as MeshKind

from . import (
    Chebyshev as Chebyshev,
    ChebyshevU as ChebyshevU,
    Fourier as Fourier,
    Jacobi as Jacobi,
    Legendre as Legendre,
    Ultraspherical as Ultraspherical,
    orthogonal as orthogonal,
)
from .arguments import (
    JAXFunction as JAXFunction,
    TestFunction as TestFunction,
    TrialFunction as TrialFunction,
)
from .composite import Composite as Composite, DirectSum as DirectSum
from .functionspace import FunctionSpace as FunctionSpace
from .inner import inner as inner
from .tensorproductspace import (
    DirectSumTPS as DirectSumTPS,
    TensorProduct as TensorProduct,
    TensorProductSpace as TensorProductSpace,
    VectorTensorProductSpace as VectorTensorProductSpace,
)
