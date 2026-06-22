"""Re-export CartesianProduct from its canonical home.

CartesianProduct lives in jaxfun.galerkin.cartesianproductspace and handles
both spectral spaces and NNSpace (neural) components. This module provides a
neutral import path: ``from jaxfun.spaces import CartesianProduct``.
"""

from jaxfun.galerkin.cartesianproductspace import (
    CartesianProduct as CartesianProduct,
)
