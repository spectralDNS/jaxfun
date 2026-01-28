from collections.abc import Callable
from enum import StrEnum
from numbers import Number
from typing import Any, Literal, Protocol

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike

type FloatLike = float | sp.Number


class SympyExpr(Protocol):
    def doit(self, **hints: Any) -> Any: ...


type Activation = Callable[[ArrayLike], Array]
type LossValue = Number | sp.Number | int | float | complex | Array
type Loss_Tuple = (
    tuple[SympyExpr, Array]
    | tuple[SympyExpr, Array, LossValue]
    | tuple[SympyExpr, Array, LossValue, LossValue]
)


class SampleMethod(StrEnum):
    UNIFORM = "uniform"
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"
    RANDOM = "random"


type DomainType = Literal["inside", "boundary", "intersection", "all"]
