from typing import Literal

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike

type Loss_Tuple = (
    tuple[sp.Expr, Array]
    | tuple[sp.Expr, Array, complex | Array]
    | tuple[sp.Expr, Array, complex | Array, complex | Array]
)

type SampleMethod = Literal["uniform", "legendre", "chebyshev", "random"]

type DomainType = Literal["inside", "boundary", "intersection", "all"]
