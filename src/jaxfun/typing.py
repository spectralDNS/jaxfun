from __future__ import annotations

from collections.abc import Callable
from enum import Enum, StrEnum, unique
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Protocol,
    Self,
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

from jaxfun.la import BaseMatrix, BlockArray, GlobalArray, GlobalMatrix
from jaxfun.la.matrixprotocol import (
    DiaMatrixSolveMethod as DiaMatrixSolveMethod,
    SolverNotApplicable as SolverNotApplicable,
)

if TYPE_CHECKING:
    from jaxfun.coordinates import BaseDyadic, BaseScalar, BaseVector
    from jaxfun.galerkin import (
        CartesianProductSpace,
        CartesianTensorProductSpace,
        DirectSum,
        DirectSumTPS,
        TensorProductSpace,
        VectorTensorProductSpace,
    )
    from jaxfun.galerkin.arguments import Jaxc
    from jaxfun.galerkin.orthogonal import OrthogonalSpace

type FloatLike = float | sp.Number
type Padding = int | tuple[int | None, ...] | tuple[tuple[int | None, ...], ...] | None
type ScalarPadding = int | tuple[int | None, ...] | None
type IntegratorState = Array | tuple[Array, ...]
type FunctionSpaceType = (
    OrthogonalSpace
    | TensorProductSpace
    | VectorTensorProductSpace
    | CartesianTensorProductSpace
    | CartesianProductSpace
    | DirectSum
    | DirectSumTPS
)
type TrialSpaceType = FunctionSpaceType
type TestSpaceType = (
    OrthogonalSpace
    | TensorProductSpace
    | VectorTensorProductSpace
    | CartesianTensorProductSpace
    | CartesianProductSpace
)
type ComputationalSpaceType = (
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


class CoercibleStrEnum(StrEnum):
    """A `StrEnum` that accepts one of its members, a member name, or a value.

    Declaring members is all a subclass has to do; `coerce` comes with them.
    """

    @classmethod
    def coerce(cls, value: str | Self) -> Self:
        """Return the member `value` names, however it is spelled.

        Name lookup runs ahead of value lookup so that a short alias resolves to
        the member it aliases -- `PolynomialKind.L`, `TestSpaceKind.PG` -- and
        so that the upper-case spelling of any member works.

        Examples::

            PolynomialKind.coerce("legendre")  # -> LEGENDRE  (value lookup)
            PolynomialKind.coerce("L")  # -> LEGENDRE  (name lookup)
            TestSpaceKind.coerce("PG")  # -> PETROV_GALERKIN  (name lookup)
            InnerKind.coerce("system")  # -> SYSTEM  (value lookup)
        """
        if isinstance(value, cls):
            return value
        try:
            return cls[value]  # by name, including any alias
        except KeyError:
            pass
        try:
            return cls(value)  # by value
        except ValueError as e:
            valid = ", ".join(repr(member.value) for member in cls)
            e.add_note(f"Expected one of: {valid}")
            raise


class PolynomialKind(CoercibleStrEnum):
    LEGENDRE = "legendre"
    L = "legendre"
    CHEBYSHEV = "chebyshev"
    C = "chebyshev"
    CHEBYSHEVU = "chebyshevu"
    U = "chebyshevu"
    JACOBI = "jacobi"
    J = "jacobi"


class SampleMethod(CoercibleStrEnum):
    UNIFORM = "uniform"
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"
    RANDOM = "random"


class MeshKind(CoercibleStrEnum):
    QUADRATURE = "quadrature"
    UNIFORM = "uniform"


class InnerKind(CoercibleStrEnum):
    BILINEAR = "bilinear"
    LINEAR = "linear"
    SYSTEM = "system"


type InnerKindLike = InnerKind | Literal["bilinear", "linear", "system"]


class ProjectionKind(CoercibleStrEnum):
    """How `project` should represent an expression in a space.

    The two coincide whenever the space can represent the expression exactly,
    and part company only once it cannot -- which is also the only time the
    distinction is worth paying for.
    """

    INTERPOLATION = "interpolation"
    """Match the expression at the quadrature points -- the discrete transform.

    Cheap, and on a curvilinear system it is a projection in its own right:
    testing with ``v/sg`` cancels the measure, leaving an exact, analytic mass.
    The residual is orthogonal to the basis in the computational inner product.
    """

    L2 = "l2"
    """Minimise the error in the physical L2 inner product, measure included.

    The residual is orthogonal to the basis under ``sg*dxi``, which is the mass
    a Galerkin discretisation assembles -- so an initial condition projected
    this way is consistent with the equations it is fed to. Costs repeated
    assembly, since the load vector has to be integrated more accurately than
    the space itself resolves.
    """


type ProjectionKindLike = ProjectionKind | Literal["interpolation", "l2"]


class TestSpaceKind(CoercibleStrEnum):
    GALERKIN = "Galerkin"
    G = "Galerkin"
    PETROV_GALERKIN = "Petrov-Galerkin"
    PG = "Petrov-Galerkin"


@unique
class RankTag(Enum):
    SCALAR = 0
    VECTOR = 1
    DYADIC = 2
    NONE = -1


type DomainType = Literal["inside", "boundary", "intersection", "all"]
type InnerItems = tuple[list[BaseMatrix | GlobalMatrix], list[GlobalArray]]
type GalerkinAssembledForm = (
    BaseMatrix | Array | BlockArray | tuple[BaseMatrix, Array | BlockArray]
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
