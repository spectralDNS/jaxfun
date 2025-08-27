from typing import Literal

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike

type LSQR_Tuple = (
    tuple[sp.Expr, Array]
    | tuple[sp.Expr, Array, Array]
    | tuple[sp.Expr, Array, Array, Array]
)

type SampleMethod = Literal["uniform", "legendre", "random"]
