from numbers import Number
from typing import Literal

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike

type Loss_Tuple = (
    tuple[sp.Expr, Array]
    | tuple[sp.Expr, Array, Number | Array]
    | tuple[sp.Expr, Array, Number | Array, Number | Array]
)

type SampleMethod = Literal["uniform", "legendre", "chebyshev", "random"]

type DomainType = Literal["inside", "boundary", "intersection", "all"]
