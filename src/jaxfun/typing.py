from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike

if TYPE_CHECKING:
    from jaxfun.galerkin import TensorProductSpace, VectorTensorProductSpace
    from jaxfun.galerkin.orthogonal import OrthogonalSpace

type FloatLike = float | sp.Number
type FunctionSpaceType = OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace


class SympyExpr(Protocol):
    def doit(self, **hints: Any) -> Any: ...


type Activation = Callable[[ArrayLike], Array]
type LossValue = sp.Number | complex | Array
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
