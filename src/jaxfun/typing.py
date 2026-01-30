from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike
from sympy.vector import Dyadic, DyadicAdd, Vector, VectorAdd

if TYPE_CHECKING:
    from jaxfun.coordinates import BaseDyadic, BaseScalar, BaseVector
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


@overload
def cast_args(t: VectorAdd) -> tuple[Vector, ...]: ...
@overload
def cast_args(t: DyadicAdd) -> tuple[Dyadic, ...]: ...
def cast_args(t: VectorAdd | DyadicAdd) -> tuple[Vector, ...] | tuple[Dyadic, ...]:
    if isinstance(t, VectorAdd):
        return cast(tuple[Vector, ...], t.args)
    else:
        return cast(tuple[Dyadic, ...], t.args)


def cast_bv(t: sp.Tuple[BaseVector]) -> tuple[BaseVector, ...]:
    return cast("tuple[BaseVector, ...]", t)


def cast_bs(t: sp.Tuple[BaseScalar]) -> tuple[BaseScalar, ...]:
    return cast("tuple[BaseScalar, ...]", t)


def cast_bd(t: sp.Tuple[BaseDyadic]) -> tuple[BaseDyadic, ...]:
    return cast("tuple[BaseDyadic, ...]", t)
