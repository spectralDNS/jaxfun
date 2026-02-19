"""
Extended differential operators (Divergence, Gradient, Curl, Cross, Dot, Outer)
for curvilinear coordinate systems.

The expressions and notation adopted from:

[1] Kelly, P. A. Mechanics Lecture Notes: Foundations of Continuum Mechanics.
    http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/index.html
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any, Literal, Self, cast, overload

import sympy as sp
from sympy import Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import Derivative as sympy_Derivative, diff as df
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import Dot as sympy_Dot, VectorAdd, VectorMul, VectorZero
from sympy.vector.basisdependent import BasisDependent, BasisDependentZero
from sympy.vector.dyadic import Dyadic, DyadicAdd, DyadicMul, DyadicZero
from sympy.vector.operators import Curl as sympy_Curl, Divergence, Gradient
from sympy.vector.vector import Vector

from jaxfun.typing import (
    DyadicLike,
    TensorLike,
    VectorLike,
    cast_args,
    cast_bs,
)

from .coordinates import (
    BaseDyadic,
    BaseScalar,
    BaseVector,
    CoordSys,
    _is_dyadiclike,
    _is_tensorlike,
    _is_vectorlike,
)


def eijk(i: int, j: int, k: int) -> int:
    """Levi-Civita symbol ε_{ijk}."""
    if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        return 1
    elif (i, j, k) in ((2, 1, 0), (1, 0, 2), (0, 2, 1)):
        return -1
    return 0


def sign(i: int, j: int) -> int:
    return 1 if ((i + 1) % 3 == j) else -1


def _get_coord_systems(expr: Basic) -> frozenset[CoordSys]:
    ret = expr.atoms(CoordSys)
    return frozenset(ret)


def express(expr: Any, system: CoordSys) -> Any:
    system_set: set[CoordSys] = set()
    expr: Any = sp.sympify(expr)
    # Substitute all the coordinate variables
    for x in expr.atoms(BaseScalar):
        if x.system != system:
            system_set.add(x.system)
    subs_dict: dict[BaseScalar, BaseScalar] = {}
    for f in system_set:
        wrong_scalars = cast_bs(f.base_scalars())
        scalars = cast_bs(system.base_scalars())
        subs_dict.update({k: v for k, v in zip(wrong_scalars, scalars, strict=False)})

    return expr.subs(subs_dict)


@overload
def from_cartesian(v: VectorLike) -> VectorLike: ...
@overload
def from_cartesian(v: DyadicLike) -> DyadicLike: ...
def from_cartesian(v: TensorLike) -> TensorLike:
    """Return Cartesian vector/dyadic expressed in the non-Cartesian basis.

    For example, if v is a vector expressed in Cartesian basis vectors and
    Curvilinear BaseScalars, then return the same vector expressed in the
    Curvilinear basis vectors.

    Example:
    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import from_cartesian
    >>> import sympy as sp
    >>> r, theta, z = sp.symbols("r,theta,z", real=True, positive=True)
    >>> C = get_CoordSys(
    ...     "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    ... )
    >>> v = C.position_vector(True)
    >>> v
    r*cos(theta)*R.i + r*sin(theta)*R.j
    >>> from_cartesian(v)
    r*P.b_r + theta*P.b_theta
    """
    assert hasattr(v, "_sys") and isinstance(v._sys, CoordSys)
    assert v._sys.is_cartesian, "from_cartesian only defined for Cartesian tensors"
    coord_sys = _get_coord_systems(v)
    if len(coord_sys) == 1:
        return v
    assert len(coord_sys) == 2
    not_cart_sys: CoordSys = next(iter(coord_sys.difference({v._sys})))
    return not_cart_sys.from_cartesian(v)


def outer(v1: VectorLike, v2: VectorLike) -> DyadicLike:
    """Return the (tensor) outer product of two vectors.

    For unevaluated outer product, use Outer.

    Args:
        v1: Left Vector operand.
        v2: Right Vector operand.

    Returns:
        A Dyadic (rank-2 tensor) representing v1 ⊗ v2.

    Examples:
        >>> from jaxfun import get_CoordSys
        >>> from jaxfun.operators import outer
        >>> import sympy as sp
        >>> r, theta = sp.symbols("r,theta", real=True)
        >>> P = get_CoordSys(
        ...     "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
        ... )
        >>> v1 = P.r * P.b_r
        >>> v2 = P.theta * P.b_theta
        >>> outer(v1, v2)
        r*theta*(P.b_r⊗P.b_theta)
    """
    if isinstance(v1, VectorZero) or isinstance(v2, VectorZero):
        return Dyadic.zero
    if isinstance(v1, VectorAdd):
        return DyadicAdd.fromiter(outer(i, v2) for i in cast_args(v1))
    if isinstance(v2, VectorAdd):
        return DyadicAdd.fromiter(outer(v1, i) for i in cast_args(v2))
    if isinstance(v1, VectorMul):
        v1_inner, m1 = next(iter(v1.components.items()))
        return m1 * outer(v1_inner, v2)
    if isinstance(v2, VectorMul):
        v2_inner, m2 = next(iter(v2.components.items()))
        return m2 * outer(v1, v2_inner)

    args = [
        (c1 * c2) * BaseDyadic(k1, k2)
        for (k1, c1), (k2, c2) in product(v1.components.items(), v2.components.items())
    ]

    return DyadicAdd(*args)


def cross(v1: VectorLike, v2: VectorLike) -> VectorLike:
    """Return the cross product of two vectors.

        v1 x v2 = ε_{ijk} √g v1^i v2^j b^k

    where {b^k} are the contravariant basis vectors, v1^i = v1·b^i and
    ε_{ijk} = ε^{ijk} is the Levi-Civita symbol, and √g the scale factor
    product (square root of determinant of the Jacobian of the coordinate
    transformation). Summation implied by repeating indices.

    For unevaluated cross product, use Cross.

    Args:
        v1: First Vector.
        v2: Second Vector.

    Returns:
        Vector representing v1 × v2 (zero if colinear).

    Raises:
        AssertionError: If attempting cross outside 3D.

    Examples:
        >>> from jaxfun import get_CoordSys
        >>> from jaxfun.operators import cross
        >>> import sympy as sp
        >>> r, theta, z = sp.symbols("r,theta,z", real=True, positive=True)
        >>> C = get_CoordSys(
        ...     "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
        ... )
        >>> cross(C.b_r, C.b_theta)
        r*C.b_z
    """
    if isinstance(v1, VectorZero) or isinstance(v2, VectorZero):
        return Vector.zero
    if isinstance(v1, VectorAdd):
        return VectorAdd.fromiter(cross(i, v2) for i in cast_args(v1))
    if isinstance(v2, VectorAdd):
        return VectorAdd.fromiter(cross(v1, i) for i in cast_args(v2))
    if isinstance(v1, VectorMul):
        v1_inner, m1 = next(iter(v1.components.items()))
        return m1 * cross(v1_inner, v2)
    if isinstance(v2, VectorMul):
        v2_inner, m2 = next(iter(v2.components.items()))
        return m2 * cross(v1, v2_inner)
    assert isinstance(v1, BaseVector) and isinstance(v2, BaseVector), (
        "cross product vectors not recognized. Consider using Cross."
    )
    if v1._sys == v2._sys:
        n1 = v1.args[0]
        n2 = v2.args[0]
        if n1 == n2:
            return Vector.zero

        assert len(v1._sys.base_scalars()) == 3, "Can only compute cross product in 3D"

        n3: int = ({0, 1, 2}.difference({n1, n2})).pop()
        if v1._sys.is_cartesian:
            sgn = 1 if ((n1 + 1) % 3 == n2) else -1
            return sgn * v1._sys.base_vectors()[n3]

        gt = v1._sys.get_contravariant_metric_tensor()
        sg = v1._sys.sg
        ei = eijk(n1, n2, n3)
        b = v1._sys.base_vectors()
        return sg * ei * gt[n3] @ b

    return from_cartesian(cross(v1._sys.to_cartesian(v1), v2._sys.to_cartesian(v2)))


type Rank = Literal[0, 1, 2]
type AdderType = Add | VectorAdd | DyadicAdd


@overload
def _rank_of_dot(t1: VectorLike, t2: VectorLike) -> Literal[0]: ...
@overload
def _rank_of_dot(t1: VectorLike, t2: DyadicLike) -> Literal[1]: ...
@overload
def _rank_of_dot(t1: DyadicLike, t2: VectorLike) -> Literal[1]: ...
@overload
def _rank_of_dot(t1: DyadicLike, t2: DyadicLike) -> Literal[2]: ...
def _rank_of_dot(t1: TensorLike, t2: TensorLike) -> Rank:
    if _is_vectorlike(t1):
        return 0 if _is_vectorlike(t2) else 1
    else:
        return 1 if _is_vectorlike(t2) else 2


@overload
def _adder_for(rank: Literal[0]) -> type[Add]: ...
@overload
def _adder_for(rank: Literal[1]) -> type[VectorAdd]: ...
@overload
def _adder_for(rank: Literal[2]) -> type[DyadicAdd]: ...
def _adder_for(rank: Rank) -> type[AdderType]:
    match rank:
        case 0:
            return Add
        case 1:
            return VectorAdd
        case 2:
            return DyadicAdd


type BasisZero = BasisDependentZero | sp.core.numbers.Zero


@overload
def _zero_for_rank(rank: Literal[0]) -> sp.core.numbers.Zero: ...
@overload
def _zero_for_rank(rank: Literal[1]) -> VectorZero: ...
@overload
def _zero_for_rank(rank: Literal[2]) -> DyadicZero: ...
def _zero_for_rank(rank: Rank) -> BasisZero:
    match rank:
        case 0:
            return sp.S.Zero
        case 1:
            return Vector.zero
        case 2:
            return Dyadic.zero


def fromiter[T: AdderType](cls: type[T], args: Iterator[Expr], **assumptions) -> T:
    return cls.fromiter(args, **assumptions)


@overload
def dot(t1: VectorLike, t2: VectorLike) -> Expr: ...
@overload
def dot(t1: VectorLike, t2: DyadicLike) -> VectorLike: ...
@overload
def dot(t1: DyadicLike, t2: VectorLike) -> VectorLike: ...
@overload
def dot(t1: DyadicLike, t2: DyadicLike) -> DyadicLike: ...
def dot(t1: TensorLike, t2: TensorLike) -> TensorLike | Expr:
    """Return the inner contraction of vectors and dyadics.

    Supports Vector·Vector, Vector·Dyadic, Dyadic·Vector and Dyadic·Dyadic,
    recursively distributing over sums and scalar multiples.

    Note that we use the term "dot product" here in a generalized sense to
    mean the inner contraction of tensors.

    The result of the contraction is a tensor of rank equal to the sum of the ranks
    of the inputs minus 2. That is, Vector·Vector returns a scalar (rank 0),
    Vector·Dyadic and Dyadic·Vector return a vector (rank 1), and Dyadic·Dyadic
    returns a dyadic (rank 2).

    For unevaluated dot product, use Dot.

    Args:
        t1: First tensor (Vector or Dyadic).
        t2: Second tensor (Vector or Dyadic).

    Returns:
        Scalar, Vector, or Dyadic depending on contraction rank.

    Examples:
        >>> from jaxfun import get_CoordSys
        >>> from jaxfun.operators import dot
        >>> import sympy as sp
        >>> r, theta = sp.symbols("r,theta", real=True, positive=True)
        >>> P = get_CoordSys(
        ...     "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
        ... )
        >>> v1 = P.b_r + P.b_theta
        >>> v2 = P.r * P.b_r + P.theta * P.b_theta
        >>> dot(v1, v2)
        r**2*theta + r
    """
    rank = _rank_of_dot(t1, t2)
    adder = _adder_for(rank)
    rank_zero = _zero_for_rank(rank)

    if not _is_tensorlike(t1) or not _is_tensorlike(t2):
        raise TypeError(
            "dot operator only defined for Vector or Dyadic types. Consider using Dot."
        )

    if isinstance(t1, BasisDependentZero) or isinstance(t2, BasisDependentZero):
        return rank_zero
    if isinstance(t1, VectorAdd | DyadicAdd):
        return fromiter(adder, (dot(i, t2) for i in cast_args(t1)))
    if isinstance(t2, VectorAdd | DyadicAdd):
        return fromiter(adder, (dot(t1, i) for i in cast_args(t2)))
    if isinstance(t1, VectorMul | DyadicMul):
        v1, m1 = next(iter(t1.components.items()))
        return m1 * dot(v1, t2)
    if isinstance(t2, VectorMul | DyadicMul):
        v2, m2 = next(iter(t2.components.items()))
        return m2 * dot(t1, v2)

    if isinstance(t1, BaseVector | BaseDyadic) and isinstance(
        t2, BaseVector | BaseDyadic
    ):
        sys1 = t1._sys
        sys2 = t2._sys
        same_sys: bool = sys1 == sys2
        same_and_cartesian: bool = same_sys and sys1.is_cartesian
        if not same_sys:
            cart1 = sys1.to_cartesian(t1)
            cart2 = sys2.to_cartesian(t2)
            out = dot(cart1, cart2)
            return from_cartesian(out) if _is_tensorlike(out) else out

    if isinstance(t1, BaseVector) and isinstance(t2, BaseVector):
        if same_and_cartesian:
            return sp.S.One if t1 == t2 else rank_zero
        g = sys1.get_covariant_metric_tensor()
        return g[t1._id[0], t2._id[0]]

    if isinstance(t1, BaseDyadic) and isinstance(t2, BaseVector):
        if same_and_cartesian:
            return t1.args[0] if t1.args[1] == t2 else rank_zero
        g = sys1.get_covariant_metric_tensor()
        g0 = g[t1.args[1]._id[0], t2._id[0]]
        if g0 == 0:
            return rank_zero
        return g0 * t1.args[0]

    if isinstance(t1, BaseVector) and isinstance(t2, BaseDyadic):
        if same_and_cartesian:
            return t2.args[1] if t1 == t2.args[0] else rank_zero
        g = sys1.get_covariant_metric_tensor()
        g0 = g[t1._id[0], t2.args[0]._id[0]]
        if g0 == 0:
            return rank_zero
        return g0 * t2.args[1]

    assert isinstance(t1, BaseDyadic) and isinstance(t2, BaseDyadic)
    if same_and_cartesian:
        return t1.args[0] | t2.args[1] if t1.args[1] == t2.args[0] else rank_zero
    g = sys1.get_covariant_metric_tensor()
    g0 = g[t1.args[1]._id[0], t2.args[0]._id[0]]
    if g0 == 0:
        return rank_zero
    return g0 * t1.args[0] | t2.args[1]


@overload
def divergence(v: DyadicLike) -> VectorLike: ...
@overload
def divergence(v: VectorLike) -> Expr: ...
def divergence(v: TensorLike) -> VectorLike | Expr:
    """Return divergence of a Vector or Dyadic field

        div(v) = ∂v/∂q^j·b^j

    where {b^j} are the contravariant basis vectors and q^j the coordinates.
    For vectors the result equals ∇·v and for dyadics the result equals ∇·v^T,
    if ∇ is interpreted as a vector.

    The dyadic result is transformed to a covariant basis before returning.

    For unevaluated divergence, use Div.

    Args:
        v: Vector or Dyadic expression.

    Returns:
        Scalar (for Vector input) or Vector (for Dyadic input).

    Examples:
        >>> from jaxfun import get_CoordSys
        >>> from jaxfun.operators import divergence
        >>> import sympy as sp
        >>> r, theta = sp.symbols("r,theta", real=True, positive=True)
        >>> P = get_CoordSys(
        ...     "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
        ... )
        >>> v = P.r * P.b_r + P.theta * P.b_theta
        >>> divergence(v)
        3
    """
    v = v.doit()
    rank = 0 if _is_vectorlike(v) else 1
    if not _is_tensorlike(v):
        raise TypeError("divergence only defined for tensors. Consider using Div.")
    if isinstance(v, BasisDependentZero) or v == sp.S.Zero:
        return _zero_for_rank(rank)
    coord_sys = _get_coord_systems(v)
    if len(coord_sys) == 1:
        # v should be a vector/dyadic with Cartesian or covariant basis vectors
        coord_sys = next(iter(coord_sys))
        x = coord_sys.base_scalars()
        bt = coord_sys.get_contravariant_basis_vector
        res = Add.fromiter(Dot(Derivative(v, x[i]), bt(i)) for i in range(len(x)))
        return res.doit()

    assert hasattr(v, "_sys") and isinstance(v._sys, CoordSys)
    if len(coord_sys) == 2 and v._sys.is_cartesian:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        # The other way around, Cartesian BaseScalars and Curvilinear basis
        # vectors, is taken care of by v = v.doit() above.
        return divergence(from_cartesian(v))

    raise TypeError(
        "divergence not implemented for tensors expressed in multiple non-Cartesian coordinate systems."  # noqa: E501
    )


@overload
def gradient(field: VectorLike, transpose: bool = False) -> DyadicLike: ...
@overload
def gradient(field: Expr, transpose: bool = False) -> VectorLike: ...
def gradient(field: Expr | VectorLike, transpose: bool = False) -> TensorLike:
    """Return gradient of a scalar or (optionally transposed) gradient of a vector.

    For scalar f: returns ∇f = ∂f/∂q^j b^j.
    For vector v: returns (∇ ⊗ v)^T = (∂v/∂q^j) ⊗ b^j (or its transpose).
        The tensors are expressed in the covariant basis before returning.

    For unevaluated gradient, use Grad.

    Args:
        field: Scalar (Expr) or Vector expression.
        transpose: If True and field is a Vector, return grad(v)^T = ∇ ⊗ v.

    Returns:
        Vector if input is scalar, Dyadic if input is Vector.

    Examples:
        >>> from jaxfun import get_CoordSys
        >>> from jaxfun.operators import gradient
        >>> import sympy as sp
        >>> r, theta = sp.symbols("r,theta", real=True, positive=True)
        >>> P = get_CoordSys(
        ...     "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
        ... )
        >>> s = P.r * P.theta
        >>> gradient(s)
        theta*P.b_r + 1/r*P.b_theta
    """
    field = field.doit()
    coord_sys = _get_coord_systems(field)
    rank = 2 if _is_vectorlike(field) else 1

    if isinstance(field, BasisDependentZero) or field == sp.S.Zero:
        return _zero_for_rank(rank)
    if len(coord_sys) == 0:
        return _zero_for_rank(rank)
    if len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        x = cast_bs(coord_sys.base_scalars())
        b = tuple(coord_sys.get_contravariant_basis_vector(i) for i in range(len(x)))
        if _is_vectorlike(field):
            dv = Add.fromiter(
                Outer(b[i], Derivative(field, x[i]))
                if transpose
                else Outer(Derivative(field, x[i]), b[i])
                for i in range(len(x))
            )
        else:
            dv = Add.fromiter(Derivative(field, x[i]) * b[i] for i in range(len(x)))

        return dv.doit()

    assert hasattr(field, "_sys") and isinstance(field._sys, CoordSys)
    if len(coord_sys) == 2 and field._sys.is_cartesian:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        # The other way around, Cartesian BaseScalars and Curvilinear basis
        # vectors, is taken care of by field = field.doit() above.
        return gradient(from_cartesian(field), transpose)
    raise TypeError(
        "gradient not implemented for tensors expressed in multiple non-Cartesian coordinate systems."  # noqa: E501
    )


def curl(v: VectorLike) -> VectorLike:
    """Return curl of a 3D vector field.

        curl(v) = ∇×v = b^j×(∂v/∂q^j) = ε^{ijk} ∂v_k/∂q^j b_i / √g

    where {b^j} are the contravariant basis vectors, q^j the coordinates,
    ε^{ijk} the Levi-Civita symbol, and √g the scale factor product
    (square root of determinant of the Jacobian of the coordinate transformation).

    For unevaluated curl, use Curl.

    Args:
        v: Vector expression (must lie in a 3D system).
        doit: If True, evaluate derivatives; else return unevaluated form.

    Returns:
        Vector representing ∇×v.

    Raises:
        AssertionError: If system dimension != 3 when required.

    Examples:
        >>> from jaxfun import get_CoordSys
        >>> from jaxfun.operators import curl
        >>> import sympy as sp
        >>> r, theta, z = sp.symbols("r,theta,z", real=True, positive=True)
        >>> P = get_CoordSys(
        ...     "P", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
        ... )
        >>> v = P.b_r + P.b_theta
        >>> curl(v)
        2*P.b_z
    """
    v = v.doit()
    assert _is_vectorlike(v), "curl only defined for Vector types. Consider using Curl."

    coord_sys = _get_coord_systems(v)

    if isinstance(v, BasisDependentZero):
        return Vector.zero
    if len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        x = cast_bs(coord_sys.base_scalars())
        b = tuple(coord_sys.get_contravariant_basis_vector(i) for i in range(len(x)))
        outvec = Add.fromiter(Cross(b[i], v.diff(x[i])) for i in range(len(x)))
        return outvec.doit()

    assert hasattr(v, "_sys") and isinstance(v._sys, CoordSys)
    if len(coord_sys) == 2 and v._sys.is_cartesian:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        # The other way around, Cartesian BaseScalars and Curvilinear basis
        # vectors, is taken care of by v = v.doit() above.
        return curl(from_cartesian(v))
    raise TypeError(
        "curl not implemented for vectors expressed in multiple non-Cartesian coordinate systems."  # noqa: E501
    )


class Grad(Gradient):
    """Unevaluated gradient wrapper with optional transpose flag.

    Behaves like sympy.vector.Gradient but works also for vectors and tracks a
    transpose state for vector gradients without immediately expanding components.

    Args:
        expr: Scalar or vector expression.
        transpose: Whether to indicate transpose for vector gradients.

    Attributes:
        _expr: Stored expression.
        _transpose: Internal flag (False for scalar fields).
    """

    _expr: Expr
    _transpose: bool

    def __new__(cls: type[Self], expr: Expr, transpose: bool = False) -> Self:
        expr = sp.sympify(expr)
        obj: Self = Expr.__new__(cls, expr)
        obj._expr = expr
        obj._transpose = False if expr.is_scalar else transpose
        return obj

    def doit(self, **hints: Any) -> Vector | Dyadic:
        return gradient(self._expr.doit(**hints), transpose=self._transpose)

    def __hash__(self) -> int:
        return Gradient.__hash__(self) + int(self._transpose)

    @property
    def T(self) -> Grad:
        if self._expr.doit().is_scalar:
            return self
        return Grad(self._expr, transpose=not self._transpose)

    def __str__(self) -> str:
        w = "Grad(" + self._expr.__str__() + ")"
        return w + ".T" if self._transpose else w

    def _pretty(self, printer: Any = None) -> prettyForm:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        printer = printer if printer is not None else LatexPrinter()
        return (
            f"\\displaystyle (\\nabla \\left({printer._print(self._expr)}\\right))^T"
            if self._transpose
            else f"\\displaystyle \\nabla \\left({printer._print(self._expr)}\\right)"
        )


class Derivative(sympy_Derivative):
    """Unevaluated Derivative wrapper delegating to custom diff for tensors."""

    expr: TensorLike | Expr

    def doit(self, **hints: Any) -> TensorLike | Expr:
        if _is_tensorlike(self.expr):
            return covariant_diff(self.expr.doit(**hints), *self.variables)
        return df(self.expr.doit(**hints), *self.variables)


class Div(Divergence):
    """Unevaluated divergence wrapper using custom curvilinear implementation."""

    _expr: TensorLike | Expr

    def doit(self, **hints: Any) -> VectorLike | Expr:
        return divergence(self._expr.doit(**hints))


class Curl(sympy_Curl):
    """Unevaluated curl wrapper using custom curvilinear implementation."""

    _expr: Expr

    def doit(self, **hints: Any) -> VectorLike:
        return curl(self._expr.doit(**hints))


class Dot(sympy_Dot):
    """Unevaluated dot product wrapper delegating to custom dot().

    Args:
        expr1: Left tensor.
        expr2: Right tensor.
    """

    _expr1: TensorLike | Expr
    _expr2: TensorLike | Expr

    def __new__(
        cls: type[Self], expr1: TensorLike | Expr, expr2: TensorLike | Expr
    ) -> Self:
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj: Self = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints: Any) -> TensorLike | Expr:
        return dot(
            self._expr1.doit(**hints),
            self._expr2.doit(**hints),
        )


# Note: Sympy subclasses like Cross(Vector), which breaks operators for
# unevaluated expressions
class Cross(Expr):
    """Unevaluated cross product.

    Args:
        expr1: Left vector expression.
        expr2: Right vector expression.

    Returns:
        An unevaluated Cross object; use .doit() to compute.

    Examples:
        >>> from jaxfun.coordinates import CartCoordSys, x, y, z
        >>> from jaxfun.operators import Cross
        >>> N = CartCoordSys("N", (x, y, z))
        >>> v1 = N.i + N.j + N.k
        >>> v2 = N.x * N.i + N.y * N.j + N.z * N.k
        >>> Cross(v1, v2)
        Cross(N.i + N.j + N.k, x*N.i + y*N.j + z*N.k)
        >>> Cross(v1, v2).doit()
        (-y + z)*N.i + (x - z)*N.j + (-x + y)*N.k
    """

    _expr1: VectorLike | Expr
    _expr2: VectorLike | Expr

    def __new__(
        cls: type[Self], expr1: VectorLike | Expr, expr2: VectorLike | Expr
    ) -> Self:
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj: Self = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints: Any) -> VectorLike:
        return cross(
            self._expr1.doit(**hints),
            self._expr2.doit(**hints),
        )


class Outer(Expr):
    """Unevaluated outer (dyadic) product.

    Args:
        expr1: Left vector expression.
        expr2: Right vector expression.

    Returns:
        An unevaluated Outer object; call .doit() for the Dyadic result.

    Examples:
        >>> from jaxfun.coordinates import CartCoordSys, x, y
        >>> from jaxfun.operators import Outer
        >>> N = CartCoordSys("N", (x, y))
        >>> v1 = N.i
        >>> v2 = N.j
        >>> Outer(v1, v2)
        Outer(N.i, N.j)
        >>> Outer(v1, v2).doit()
        (N.i⊗N.j)
    """

    _expr1: VectorLike | Expr
    _expr2: VectorLike | Expr

    def __new__(
        cls: type[Self], expr1: VectorLike | Expr, expr2: VectorLike | Expr
    ) -> Self:
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj: Self = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    @property
    def T(self) -> Outer:
        return Outer(self._expr2, self._expr1)

    def transpose(self) -> Outer:
        return self.T

    def doit(self, **hints: Any) -> DyadicLike:
        return outer(self._expr1.doit(), self._expr2.doit())


class Source(Expr):
    _expr: Expr

    def __new__(cls: type[Self], expr: Expr) -> Self:
        expr = sp.sympify(expr)
        obj: Self = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def doit(self, **hints: Any) -> Expr:
        return self._expr.doit(**hints)


class Constant(sp.Symbol):
    val: Expr | float

    def __new__(cls: type[Self], name: str, val: Expr | float, **assumptions) -> Self:
        obj: Self = super().__new__(cls, name, **assumptions)
        obj.val = val
        return obj

    def doit(self, **hints: Any) -> Expr | float:
        return self.val


class Identity(Expr):
    def __init__(self, sys: CoordSys) -> None:
        self.sys = sys

    def doit(self, **hints: Any) -> Dyadic:
        return sum(self.sys.base_dyadics()[:: self.sys.dims + 1], DyadicZero())


def covariant_diff_cartesian(self, *args, **kwargs) -> TensorLike:
    """Differentiate a tensor (vector/dyadic) (component-wise) wrt provided variables.

    Converts to Cartesian to avoid differentiating basis vectors, applies
    SymPy diff to scalar components, then maps back.

    Args:
        *args: Differentiation variables / (variable, order) pairs.
        **kwargs: Passed to sympy.diff.

    Returns:
        Tensor with differentiated components.

    Raises:
        TypeError: If any differentiation target is itself a basis-dependent object.
    """
    for x in args:
        if isinstance(x, BasisDependent):
            raise TypeError("Invalid arg for differentiation")
    v0 = self._sys.to_cartesian(self)
    diff_components = [df(v, *args, **kwargs) * k for k, v in v0.components.items()]
    f = self._add_func(*diff_components)
    return self._sys.from_cartesian(f)


def diff_covariant_vector(
    self: VectorLike, *args: BaseScalar | int, **kwargs: Any
) -> VectorLike:
    r"""Covariant derivative of a vector (component-wise) wrt provided variables.

    .. math::

        \frac{\partial v}{\partial q^j} = (\partial_j v^i + \Gamma^i_{kj} v^k) b_i

    Args:
        *args: Differentiation variables (BaseScalars, ints).
        **kwargs: Passed to sympy.diff.

    Returns:
        Vector with differentiated components.

    Raises:
        TypeError: If any differentiation target is itself a basis-dependent object.
    """

    if isinstance(self, VectorZero):
        return self

    if len(args) > 1:  # (x, 2), (x, y), (x, 2, y, 2), etc
        res = self
        for arg in args:
            if isinstance(arg, BaseScalar):
                der_arg = arg
            if isinstance(arg, int):  # repeat arg-1 times
                for _ in range(1, arg):
                    res = diff_covariant_vector(res, der_arg, **kwargs)
            else:
                res = diff_covariant_vector(res, der_arg, **kwargs)
        return res

    assert hasattr(self, "_sys") and isinstance(self._sys, CoordSys)
    assert isinstance(args[0], BaseScalar)
    ct = self._sys.get_christoffel_second()
    b2i = self._sys._covariant_basis_map_inv
    x2i = self._sys._map_base_scalar_to_index
    arg = args[0]
    j = x2i[arg]
    res = _zero_for_rank(1)
    for bi, vi in self.components.items():
        i = b2i[bi]
        di = df(vi, arg, **kwargs)
        for bk, vk in self.components.items():
            k = b2i[bk]
            di += ct[i, k, j] * vk
        res += di * bi
    return res


def diff_covariant_dyadic(
    self: DyadicLike, *args: BaseScalar | int, **kwargs: Any
) -> DyadicLike:
    r"""Covariant derivative of a dyadic (component-wise) wrt provided variables.

    .. math::

        \frac{\partial a}{\partial q^k} = (\partial_k a^{ij} + \Gamma^i_{mk} a^{mj} + \Gamma^j_{mk} a^{im}) b_i \otimes b_j

    Args:
        *args: Differentiation variables (BaseScalars, ints).
        **kwargs: Passed to sympy.diff.

    Returns:
        Vector with differentiated components.

    Raises:
        TypeError: If any differentiation target is itself a basis-dependent object.
    """  # noqa: E501
    if isinstance(self, DyadicZero):
        return self

    if len(args) > 1:  # (x, 2), (x, y), (x, 2, y, 2), etc
        res = self
        for arg in args:
            if isinstance(arg, BaseScalar):
                der_arg = arg
            if isinstance(arg, int):  # repeat arg-1 times
                for _ in range(1, arg):
                    res = diff_covariant_dyadic(res, der_arg, **kwargs)
            else:
                res = diff_covariant_dyadic(res, der_arg, **kwargs)
        return res
    assert hasattr(self, "_sys") and isinstance(self._sys, CoordSys)
    assert isinstance(args[0], BaseScalar)
    ct = self._sys.get_christoffel_second()
    bb2ij = self._sys._covariant_basis_dyadic_map_inv
    x2i = self._sys._map_base_scalar_to_index
    arg = args[0]
    k = x2i[arg]
    res = _zero_for_rank(2)
    for bij, aij in self.components.items():
        i, j = bb2ij[bij]
        di = df(aij, arg, **kwargs)
        for bml, aml in self.components.items():
            m, l = bb2ij[bml]
            if l == j:
                di += ct[i, m, k] * aml
            if m == i:
                di += ct[j, l, k] * aml
        res += di * bij
    return res


@overload
def covariant_diff(
    self: VectorLike, *args: BaseScalar | int, **kwargs: Any
) -> VectorLike: ...
@overload
def covariant_diff(
    self: DyadicLike, *args: BaseScalar | int, **kwargs: Any
) -> DyadicLike: ...
def covariant_diff(
    self: TensorLike, *args: BaseScalar | int, **kwargs: Any
) -> TensorLike:
    r"""Covariant derivative of a tensor (vector/dyadic) wrt provided variables.

    Vector:

    .. math::
        \frac{\partial v}{\partial q^j} = (\partial_j v^i + \Gamma^i_{kj} v^k) b_i

    Dyadic:

    .. math::
        \frac{\partial a}{\partial q^k} = (\partial_k a^{ij} + \Gamma^i_{mk} a^{mj} + \Gamma^j_{mk} a^{im}) b_i \otimes b_j

    Uses Christoffel symbols :math:`\Gamma^i_{jk}` to differentiate basis vectors correctly.

    Args:
        *args: Differentiation variables / (variable, order) pairs.
        **kwargs: Passed to sympy.diff.

    Returns:
        Tensor with differentiated components.

    Raises:
        TypeError: If any differentiation target is itself a basis-dependent object.
    """  # noqa: E501
    for x in args:
        if isinstance(x, BasisDependent):
            raise TypeError("Invalid arg for differentiation")
    coord_sys = _get_coord_systems(self)
    assert hasattr(self, "_sys") and isinstance(self._sys, CoordSys)
    if len(coord_sys) == 2 and self._sys.is_cartesian:
        self = from_cartesian(self)
    assert len(coord_sys) == 1
    if _is_vectorlike(self):
        out = diff_covariant_vector(self, *args, **kwargs)
    elif _is_dyadiclike(self):
        out = diff_covariant_dyadic(self, *args, **kwargs)
    else:
        raise TypeError("covariant_diff only defined for Vector or Dyadic types.")
    coord_sys_out = _get_coord_systems(out)
    if len(coord_sys_out) > 1:
        return cast(TensorLike, express(out.doit(), list(coord_sys)[0]))
    return out


# Note: covariant_diff and covariant_diff_cartesian should give
# the same result.

# Regular doit() is problematic for vectors and dyadics when these are
# used unevaluated. For example
# from sympy.vector import CoordSys3D, Gardient
# N = CoordSys3D("N")
# f = N.x*N.y
# h = 4*Gradient(f)
# z = h.doit()
# # -> z = 4*N.x*N.j + 4*N.y*N.i
# z.is_Vector
# # -> False
# z is now a type Add and not VectorAdd as it should be.
# Using the doit function below is a hack around it
# from jaxfun.coordinates import CartCoordSys, x, y, z
# from jaxfun.operators import Grad
# N = CartCoordSys("N", (x, y, z))
# f = N.x*N.y
# h = 4*Grad(f) # This is correctly of type Mul
# h.doit().is_Vector
# # -> True


def doit(self, **hints: Any) -> Basic:
    if hints.get("deep", True):
        terms = [
            term.doit(**hints) if isinstance(term, sp.Basic) else term
            for term in self.args
        ]
        z = self.func(*terms)
    else:
        z = self
    if isinstance(z, sp.Add):
        # Check if should be VectorAdd
        for p in sp.core.traversal.preorder_traversal(z):
            if isinstance(p, BaseVector):
                return VectorAdd.fromiter(z.args)
            elif isinstance(p, BaseDyadic):
                return DyadicAdd.fromiter(z.args)

    elif isinstance(z, sp.Mul):
        # Check if should be VectorMul
        for p in sp.core.traversal.preorder_traversal(z):
            if isinstance(p, BaseVector):
                return VectorMul.fromiter(z.args)
            elif isinstance(p, BaseDyadic):
                return DyadicMul.fromiter(z.args)
    return z


sp.core.Expr.doit = doit  # ty:ignore[invalid-assignment]
sp.vector.vector.dot = dot  # ty:ignore[possibly-missing-attribute]
sp.vector.vector.cross = cross  # ty:ignore[possibly-missing-attribute]
sp.vector.operators.gradient = gradient  # ty:ignore[possibly-missing-attribute]
sp.vector.operators.curl = curl  # ty:ignore[possibly-missing-attribute]
sp.vector.operators.divergence = divergence  # ty:ignore[possibly-missing-attribute]
sp.vector.vector.Cross = Cross  # ty:ignore[possibly-missing-attribute]
sp.vector.vector.Dot = Dot  # ty:ignore[possibly-missing-attribute]
sp.vector.operators.Curl = Curl  # ty:ignore[possibly-missing-attribute]
sp.vector.operators.Gradient = Grad  # ty:ignore[possibly-missing-attribute]
sp.vector.operators.Divergence = Div  # ty:ignore[possibly-missing-attribute]
sp.vector.Cross = Cross  # ty:ignore[possibly-missing-attribute]
sp.vector.Dot = Dot  # ty:ignore[possibly-missing-attribute]
sp.vector.Curl = Curl  # ty:ignore[possibly-missing-attribute]
sp.vector.Gradient = Grad  # ty:ignore[possibly-missing-attribute]
sp.vector.Divergence = Div  # ty:ignore[possibly-missing-attribute]
sp.vector.basisdependent.BasisDependent.diff = covariant_diff  # ty:ignore[possibly-missing-attribute]
sp.vector.Dyadic.is_Dyadic = True  # ty:ignore[possibly-missing-attribute]
