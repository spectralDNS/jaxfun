from . import (
    Chebyshev as Chebyshev,
    Fourier as Fourier,
    Jacobi as Jacobi,
    Legendre as Legendre,
    orthogonal as orthogonal,
)
from .arguments import (
    JAXArray as JAXArray,
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
