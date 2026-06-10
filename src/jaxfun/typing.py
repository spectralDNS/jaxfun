from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Protocol,
    cast,
    overload,
)

import sympy as sp
from jax import Array as Array
from jax.typing import ArrayLike as ArrayLike
from sympy.core.function import AppliedUndef
from sympy.vector import (
    Dyadic,
    DyadicAdd,
    DyadicMul,
    DyadicZero,
    Vector,
    VectorAdd,
    VectorMul,
    VectorZero,
)
from typing_extensions import TypedDict

from jaxfun.la import BaseMatrix, BlockArray, IndexedArray
from jaxfun.la.matrixprotocol import (
    DiaMatrixSolveMethod as DiaMatrixSolveMethod,
    SolverNotApplicable as SolverNotApplicable,
)

if TYPE_CHECKING:
    from jaxfun.coordinates import BaseDyadic, BaseScalar, BaseVector
    from jaxfun.galerkin import (
        CartesianProductSpace,
        DirectSum,
        DirectSumTPS,
        TensorProductSpace,
        VectorTensorProductSpace,
    )
    from jaxfun.galerkin.arguments import Jaxc
    from jaxfun.galerkin.orthogonal import OrthogonalSpace

type FloatLike = float | sp.Number
type Padding = int | tuple[int | None, ...] | tuple[tuple[int | None, ...], ...] | None
type FunctionSpaceType = (
    OrthogonalSpace
    | TensorProductSpace
    | VectorTensorProductSpace
    | CartesianProductSpace
    | DirectSum
    | DirectSumTPS
)
type TrialSpaceType = FunctionSpaceType
type TestSpaceType = (
    OrthogonalSpace
    | TensorProductSpace
    | VectorTensorProductSpace
    | CartesianProductSpace
)
type ComputationalSpaceType = (
    OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace
)
type RankedTrialSpaceType = (
    OrthogonalSpace
    | TensorProductSpace
    | VectorTensorProductSpace
    | DirectSum
    | DirectSumTPS
)
type RankedTestSpaceType = (
    OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace
)
type ScalarSpaceType = OrthogonalSpace | TensorProductSpace | DirectSum | DirectSumTPS

type VectorLike = BaseVector | Vector | VectorAdd | VectorMul | VectorZero
type DyadicLike = BaseDyadic | Dyadic | DyadicAdd | DyadicMul | DyadicZero
type TensorLike = VectorLike | DyadicLike


class SympyExpr(Protocol):
    def doit(self, **hints: Any) -> Any: ...


type ArrayFun = Callable[[Array], Array]
type TriDiagMatrixFun = Callable[[sp.Symbol | int, sp.Symbol | int], sp.Expr]
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


class MeshKind(StrEnum):
    QUADRATURE = "quadrature"
    UNIFORM = "uniform"


class InnerKind(StrEnum):
    BILINEAR = "bilinear"
    LINEAR = "linear"
    SYSTEM = "system"


type InnerKindLike = InnerKind | Literal["bilinear", "linear", "system"]


class TestSpaceKind(StrEnum):
    GALERKIN = "Galerkin"
    G = "Galerkin"
    PETROV_GALERKIN = "Petrov-Galerkin"
    PG = "Petrov-Galerkin"

    @classmethod
    def coerce(cls, value: str | TestSpaceKind) -> TestSpaceKind:
        """Accept a value, member name, or short alias and return the canonical member.

        Examples::

            TestSpaceKind.coerce("Galerkin")  # -> GALERKIN  (value lookup)
            TestSpaceKind.coerce("G")  # -> GALERKIN  (name lookup)
            TestSpaceKind.coerce("GALERKIN")  # -> GALERKIN  (name lookup)
            TestSpaceKind.coerce("PG")  # -> PETROV_GALERKIN  (name lookup)
        """
        if isinstance(value, cls):
            return value
        try:
            return cls[value]  # match by name: "G", "PG", "GALERKIN", "PETROV_GALERKIN"
        except KeyError:
            pass
        try:
            return cls(value)  # match by value: "Galerkin", "Petrov-Galerkin"
        except ValueError:
            raise ValueError(f"{value!r} is not a valid {cls.__name__}") from None


type DomainType = Literal["inside", "boundary", "intersection", "all"]
type InnerBilinearResult = Array | BaseMatrix
type InnerBilinearResults = list[Array | BaseMatrix]
type InnerLinearResults = list[Array]
type InnerItems = tuple[list[BaseMatrix], list[IndexedArray]]
type GalerkinOperator = BaseMatrix
type GalerkinAssembledForm = (
    GalerkinOperator | Array | BlockArray | tuple[GalerkinOperator, Array | BlockArray]
)


@overload
def cast_args(t: VectorAdd) -> tuple[VectorLike, ...]: ...
@overload
def cast_args(t: DyadicAdd) -> tuple[DyadicLike, ...]: ...
def cast_args(t: TensorLike) -> tuple[TensorLike, ...]:
    from jaxfun.coordinates import _is_vectorlike

    if _is_vectorlike(t):
        return cast(tuple[VectorLike, ...], t.args)
    else:
        return cast(tuple[DyadicLike, ...], t.args)


def cast_bv(t: sp.Tuple[BaseVector]) -> tuple[BaseVector, ...]:
    return cast("tuple[BaseVector, ...]", t)


def cast_bs(t: sp.Tuple[BaseScalar]) -> tuple[BaseScalar, ...]:
    return cast("tuple[BaseScalar, ...]", t)


def cast_bd(t: sp.Tuple[BaseDyadic]) -> tuple[BaseDyadic, ...]:
    return cast("tuple[BaseDyadic, ...]", t)


# from forms
class InnerResultDict(TypedDict, extra_items=sp.Expr):
    coeff: sp.Expr | float
    multivar: NotRequired[sp.Expr]
    jaxfunction: NotRequired[AppliedUndef | sp.Expr]


class ResultDict(TypedDict):
    linear: list[InnerResultDict]
    bilinear: list[InnerResultDict]


class LinearCoeffDict(TypedDict, total=False):
    scale: float
    jaxcoeff: NotRequired[Jaxc]


class CoeffDict(TypedDict, total=False):
    bilinear: complex
    linear: LinearCoeffDict
