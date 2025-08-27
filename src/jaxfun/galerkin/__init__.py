from . import (
    Chebyshev as Chebyshev,
    Fourier as Fourier,
    Jacobi as Jacobi,
    Legendre as Legendre,
)
from .arguments import (
    JAXFunction as JAXFunction,
    TestFunction as TestFunction,
    TrialFunction as TrialFunction,
)
from .functionspace import FunctionSpace as FunctionSpace
from .inner import inner as inner
from .tensorproductspace import (
    TensorProduct as TensorProduct,
    TensorProductSpace as TensorProductSpace,
    VectorTensorProductSpace as VectorTensorProductSpace,
)
