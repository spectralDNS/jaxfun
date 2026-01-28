"""
Extended differential operators (Divergence, Gradient, Curl, Cross, Dot, Outer)
for curvilinear coordinate systems.

The expressions and notation adopted from:

[1] Kelly, P. A. Mechanics Lecture Notes: Foundations of Continuum Mechanics.
    http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/index.html
"""

from __future__ import annotations

import collections
from itertools import product
from numbers import Number
from typing import Any, Literal, Protocol, Self, cast, overload

import numpy as np
import sympy as sp
from sympy import Expr
from sympy.core import preorder_traversal
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import Derivative, diff as df
from sympy.core.mul import Mul
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import Dot as sympy_Dot, VectorAdd, VectorMul, VectorZero
from sympy.vector.basisdependent import BasisDependent
from sympy.vector.dyadic import Dyadic, DyadicAdd, DyadicMul, DyadicZero
from sympy.vector.operators import (
    Curl as sympy_Curl,
    Divergence,
    Gradient,
)
from sympy.vector.vector import Vector

from jaxfun.coordinates import BaseDyadic, BaseScalar, BaseVector, CoordSys


class _IsCartesian(Protocol):
    is_cartesian: bool


class _HasSys(Protocol):
    _sys: CoordSys


class _HasId(Protocol):
    _id: tuple[int, ...]


class _BaseVectorLike(_HasSys, _HasId, Protocol):
    args: tuple[Any, ...]


class _BaseDyadicLike(_HasSys, Protocol):
    args: tuple[Any, ...]


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
    g = preorder_traversal(expr)
    ret: set[CoordSys] = set()
    for i in g:
        if isinstance(i, CoordSys):
            ret.add(i)
            g.skip()
    return frozenset(ret)


def _split_mul_args_wrt_coordsys(expr: Expr) -> list[Expr]:
    d = collections.defaultdict(lambda: sp.S.One)
    for i in expr.args:
        d[_get_coord_systems(i)] *= i
    return list(d.values())


def express(expr: Basic, system: CoordSys) -> Basic:
    system_set = set()
    expr = sp.sympify(expr)
    # Substitute all the coordinate variables
    for x in expr.atoms(BaseScalar):
        if x.system != system:
            system_set.add(x.system)
    subs_dict = {}
    for f in system_set:
        wrong_scalars = f.base_scalars()
        scalars = system.base_scalars()
        subs_dict.update({k: v for k, v in zip(wrong_scalars, scalars, strict=False)})
    return expr.subs(subs_dict)


def outer(v1: Vector, v2: Vector) -> Dyadic:
    """Return the (tensor) outer product of two vectors.

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
    if isinstance(v1, VectorAdd):
        return DyadicAdd.fromiter(outer(cast(Vector, i), v2) for i in v1.args)
    if isinstance(v2, VectorAdd):
        return DyadicAdd.fromiter(outer(v1, cast(Vector, i)) for i in v2.args)
    if isinstance(v1, VectorMul):
        v1_inner, m1 = next(iter(v1.components.items()))
        return m1 * outer(v1_inner, v2)
    if isinstance(v2, VectorMul):
        v2_inner, m2 = next(iter(v2.components.items()))
        return m2 * outer(v1, v2_inner)

    if isinstance(v1, VectorZero) or isinstance(v2, VectorZero):
        return Dyadic.zero

    args = [
        (c1 * c2) * BaseDyadic(k1, k2)
        for (k1, c1), (k2, c2) in product(v1.components.items(), v2.components.items())
    ]

    return DyadicAdd(*args)


@overload
def cross(v1: BaseVector, v2: BaseVector) -> Vector: ...


@overload
def cross(v1: Vector, v2: Vector) -> Vector | Cross: ...


def cross(v1: Vector, v2: Vector) -> Vector | Cross:
    """Return the cross product of two vectors.

        v1 x v2 = ε_{ijk} √g v1^i v2^j b^k

    where {b^k} are the contravariant basis vectors, v1^i = v1·b^i and
    ε_{ijk} = ε^{ijk} is the Levi-Civita symbol, and √g the scale factor
    product (square root of determinant of the Jacobian of the coordinate
    transformation). Summation implied by repeating indices.

    Handles:
      * Linear (Add) combinations
      * Scalar multiples (VectorMul)
      * Base vectors in Cartesian or curvilinear coordinates

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
    if isinstance(v1, Add):
        return VectorAdd.fromiter(cross(cast(Vector, i), v2) for i in v1.args)
    if isinstance(v2, Add):
        return VectorAdd.fromiter(cross(v1, cast(Vector, i)) for i in v2.args)
    if isinstance(v1, BaseVector) and isinstance(v2, BaseVector):
        v1v = cast(_BaseVectorLike, v1)
        v2v = cast(_BaseVectorLike, v2)
        if v1v._sys == v2v._sys:
            n1 = cast(int, v1v.args[0])
            n2 = cast(int, v2v.args[0])
            if n1 == n2:
                return Vector.zero

            assert len(v1v._sys.base_scalars()) == 3, (
                "Can only compute cross product in 3D"
            )

            if v1v._sys.is_cartesian:
                n3 = ({0, 1, 2}.difference({n1, n2})).pop()
                sgn = 1 if ((n1 + 1) % 3 == n2) else -1
                return sgn * v1v._sys.base_vectors()[n3]
            else:
                gt = v1v._sys.get_contravariant_metric_tensor()
                sg = v1v._sys.sg
                n3 = ({0, 1, 2}.difference({n1, n2})).pop()
                ei = eijk(n1, n2, n3)
                b = v1v._sys.base_vectors()
                return sg * ei * gt[n3] @ b

        return cross(
            cast(Vector, v1v._sys.to_cartesian(v1v)),
            cast(Vector, v2v._sys.to_cartesian(v2v)),
        )

    if isinstance(v1, VectorZero) or isinstance(v2, VectorZero):
        return Vector.zero
    if isinstance(v1, VectorMul):
        v1_inner, m1 = next(iter(v1.components.items()))
        return m1 * cross(v1_inner, v2)
    if isinstance(v2, VectorMul):
        v2_inner, m2 = next(iter(v2.components.items()))
        return m2 * cross(v1, v2_inner)

    return Cross(v1, v2)


def dot(t1: Vector | Dyadic, t2: Vector | Dyadic) -> BasisDependent | Expr | Dot:
    """Return the (possibly contracted) inner product of two tensors.

    Supports Vector·Vector, Vector·Dyadic, Dyadic·Vector and Dyadic·Dyadic,
    recursively distributing over sums and scalar multiples.

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
    rank: int = 0
    if (isinstance(t1, Vector) and isinstance(t2, Dyadic)) or (
        isinstance(t1, Dyadic) and isinstance(t2, Vector)
    ):
        rank = 1
    elif isinstance(t1, Dyadic) and isinstance(t2, Dyadic):
        rank = 2

    if isinstance(t1, VectorAdd | DyadicAdd):
        if rank == 0:
            return Add.fromiter(dot(cast(Vector | Dyadic, i), t2) for i in t1.args)
        elif rank == 1:
            return VectorAdd.fromiter(
                dot(cast(Vector | Dyadic, i), t2) for i in t1.args
            )
        else:
            return DyadicAdd.fromiter(
                dot(cast(Vector | Dyadic, i), t2) for i in t1.args
            )
    if isinstance(t2, VectorAdd | DyadicAdd):
        if rank == 0:
            return Add.fromiter(dot(t1, cast(Vector | Dyadic, i)) for i in t2.args)
        elif rank == 1:
            return VectorAdd.fromiter(
                dot(t1, cast(Vector | Dyadic, i)) for i in t2.args
            )
        else:
            return DyadicAdd.fromiter(
                dot(t1, cast(Vector | Dyadic, i)) for i in t2.args
            )

    if isinstance(t1, BaseVector) and isinstance(t2, BaseVector):
        t1v = cast(_BaseVectorLike, t1)
        t2v = cast(_BaseVectorLike, t2)
        if t1v._sys == t2v._sys:
            if t1v._sys.is_cartesian:
                return sp.S.One if t1v == t2v else sp.S.Zero
            else:
                g = t1v._sys.get_covariant_metric_tensor()
                return g[t1v._id[0], t2v._id[0]]

        return dot(t1v._sys.to_cartesian(t1v), t2v._sys.to_cartesian(t2v))

    if isinstance(t1, BaseDyadic) and isinstance(t2, BaseVector):
        t1d = cast(_BaseDyadicLike, t1)
        t2v = cast(_BaseVectorLike, t2)
        if t1d._sys == t2v._sys:
            if t1d._sys.is_cartesian:
                return t1d.args[0] if t1d.args[1] == t2v else VectorZero()
            else:
                g = t1d._sys.get_covariant_metric_tensor()
                g0 = g[t1d.args[1]._id[0], t2v._id[0]]
                if g0 == 0:
                    return Vector.zero
                else:
                    return g0 * t1d.args[0]

        return dot(t1d._sys.to_cartesian(t1d), t2v._sys.to_cartesian(t2v))

    if isinstance(t1, BaseVector) and isinstance(t2, BaseDyadic):
        t1v = cast(_BaseVectorLike, t1)
        t2d = cast(_BaseDyadicLike, t2)
        if t1v._sys == t2d._sys:
            if t1v._sys.is_cartesian:
                return t2d.args[1] if t1v == t2d.args[0] else VectorZero()
            else:
                g = t1v._sys.get_covariant_metric_tensor()
                g0 = g[t1v._id[0], t2d.args[0]._id[0]]
                if g0 == 0:
                    return Vector.zero
                else:
                    return g0 * t2d.args[1]

        return dot(t1v._sys.to_cartesian(t1v), t2d._sys.to_cartesian(t2d))

    if isinstance(t1, BaseDyadic) and isinstance(t2, BaseDyadic):
        t1d = cast(_BaseDyadicLike, t1)
        t2d = cast(_BaseDyadicLike, t2)
        if t1d._sys == t2d._sys:
            if t1d._sys.is_cartesian:
                return (
                    cast(Vector, t1d.args[0]) | cast(Vector, t2d.args[1])
                    if t1d.args[1] == t2d.args[0]
                    else DyadicZero()
                )
            else:
                g = t1d._sys.get_covariant_metric_tensor()
                g0 = g[t1d.args[1]._id[0], t2d.args[0]._id[0]]
                if g0 == 0:
                    return DyadicZero()
                else:
                    return g0 * cast(Vector, t1d.args[0]) | cast(Vector, t2d.args[1])

        return dot(t1d._sys.to_cartesian(t1d), t2d._sys.to_cartesian(t2d))
    if isinstance(t1, DyadicZero) and isinstance(t2, BaseVector):
        return Vector.zero
    if isinstance(t1, DyadicZero) and isinstance(t2, BaseDyadic):
        return Dyadic.zero
    if isinstance(t1, BaseVector) and isinstance(t2, DyadicZero):
        return Vector.zero
    if isinstance(t1, BaseDyadic) and isinstance(t2, DyadicZero):
        return Dyadic.zero
    if isinstance(t1, DyadicZero) and isinstance(t2, DyadicZero):
        return Dyadic.zero

    if isinstance(t1, VectorZero) and isinstance(t2, BaseVector):
        return sp.S.Zero
    if isinstance(t1, VectorZero) and isinstance(t2, BaseDyadic):
        return Vector.zero
    if isinstance(t1, BaseVector) and isinstance(t2, VectorZero):
        return sp.S.Zero
    if isinstance(t1, BaseDyadic) and isinstance(t2, VectorZero):
        return Vector.zero
    if isinstance(t1, VectorZero) and isinstance(t2, VectorZero):
        return sp.S.Zero
    if isinstance(t1, VectorMul | DyadicMul):
        v1, m1 = next(iter(t1.components.items()))
        return m1 * dot(v1, t2)
    if isinstance(t2, VectorMul | DyadicMul):
        v2, m2 = next(iter(t2.components.items()))
        return m2 * dot(t1, v2)

    return Dot(t1, t2)


@overload
def divergence(
    v: Vector | Dyadic | Cross | sympy_Curl | Gradient | BasisDependent,
    doit: Literal[False],
) -> Div: ...


@overload
def divergence(
    v: Vector | Dyadic | Cross | sympy_Curl | Gradient | BasisDependent,
    doit: Literal[True] = True,
) -> Vector | Expr: ...


def divergence(
    v: Vector | Dyadic | Cross | sympy_Curl | Gradient | BasisDependent,
    doit: bool = True,
) -> Vector | Expr | Basic | Div:
    """Return divergence of a Vector or Dyadic field

        div(v) = ∂v/∂q^j·b^j

    where {b^j} are the contravariant basis vectors and q^j the coordinates.
    For vectors the result equals ∇·v and for dyadics the result equals ∇·v^T.
    The dyadic result is transformed to a covariant basis before returning.

    Handles:
      * Linear (Add) combinations
      * Scalar multiples (Mul)
      * Cross, Curl, Gradient expressions
      * Base vectors in Cartesian or curvilinear coordinates
    Args:
        v: Vector or Dyadic expression.
        doit: If True, evaluates derivatives; if False returns unevaluated Derivative
        nodes.

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
    coord_sys = _get_coord_systems(v)
    if len(coord_sys) == 0:
        if v.is_Vector:
            return sp.S.Zero
        else:
            return Vector.zero
    elif len(coord_sys) == 1:
        if isinstance(v, Cross | Curl | Gradient):
            return Div(v)
        # v should be a vector/dyadic with Cartesian or covariant basis vectors
        # Note: In the formulas below we could also simply use the definition
        # bt = coord_sys.get_contravariant_basis(True)
        # res = sp.Add.fromiter(Dot(v.diff(x[i]), bt[i]).doit() for i in range(len(x)))
        # However, it is much faster to use precomputed metrics and Christoffel symbols
        coord_sys = next(iter(coord_sys))
        x = coord_sys.base_scalars()
        comp = coord_sys.get_contravariant_component
        sg = coord_sys.sg
        if v.is_Vector:
            res = sp.S.Zero
            for i in range(len(x)):
                res += Derivative(express(comp(v, i) * sg, coord_sys), x[i]) / sg

        else:
            bv = coord_sys.base_vectors()
            ct = coord_sys.get_christoffel_second()
            res = []
            for i in range(len(x)):
                r0 = sp.S.Zero
                for j in range(len(x)):
                    r0 += comp(v, i, j).diff(x[j])
                    for k in range(len(x)):
                        r0 += ct[i, k, j] * comp(v, k, j) + ct[j, k, j] * comp(v, i, k)
                res.append(r0 * bv[i])
            res = VectorAdd(*res)

        if doit:
            return express(res.doit(), coord_sys)
        return res

    elif len(coord_sys) == 2:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        coord_sys_arr = np.array(list(coord_sys))
        mask = np.array(
            [not cast(_IsCartesian, si).is_cartesian for si in coord_sys_arr],
            dtype=bool,
        )
        not_cart = np.nonzero(mask)[0]
        assert len(not_cart) == 1
        return divergence(
            cast(CoordSys, coord_sys_arr[not_cart[0]]).from_cartesian(v), doit=doit
        )

    else:
        if isinstance(v, DyadicAdd):
            return VectorAdd.fromiter(
                divergence(cast(Vector | Dyadic, i), doit=doit) for i in v.args
            )
        elif isinstance(v, Add | VectorAdd):
            return Add.fromiter(
                divergence(cast(Vector | Dyadic, i), doit=doit) for i in v.args
            )
        elif isinstance(v, Mul | VectorMul):
            vector = [
                i for i in v.args if isinstance(i, Dyadic | Vector | Cross | Gradient)
            ][0]
            scalar = Mul.fromiter(
                i
                for i in v.args
                if not isinstance(i, Dyadic | Vector | Cross | Gradient)
            )
            if not vector.is_Vector:
                res = VectorAdd(
                    dot(
                        cast(Vector | Dyadic, vector),
                        cast(Vector | Dyadic, gradient(scalar)),
                    ),
                    scalar * divergence(cast(Vector | Dyadic, vector), doit=doit),
                )
            else:
                res = Dot(vector, gradient(scalar)) + scalar * divergence(
                    cast(Vector | Dyadic, vector), doit=doit
                )
            if doit:
                return res.doit()
            return res
        elif isinstance(v, Cross | Curl | Gradient):
            return Div(v)
        else:
            raise Div(v)  # type: ignore[invalid-raise]


@overload
def gradient(
    field: Expr | Vector,
    doit: Literal[False],
    transpose: bool = False,
) -> Grad: ...


@overload
def gradient(
    field: Expr | Vector,
    doit: Literal[True] = True,
    transpose: bool = False,
) -> BasisDependent: ...


def gradient(
    field: Expr | Vector, doit: bool = True, transpose: bool = False
) -> BasisDependent | Grad:
    """Return gradient of a scalar or (optionally transposed) gradient of a vector.

    For scalar f: returns ∇f = ∂f/∂q^j b^j.
    For vector v: returns (∇ ⊗ v)^T = (∂v/∂q^j) ⊗ b^j (or its transpose).
        The tensors are expressed in the covariant basis before returning.

    Handles:
      * Linear (Add) combinations
      * Scalar multiples (Mul)
      * Base vectors in Cartesian or curvilinear coordinates

    Args:
        field: Scalar (Expr) or Vector expression.
        doit: If True, evaluate derivatives immediately.
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
    coord_sys = _get_coord_systems(field)

    if len(coord_sys) == 0:
        return Vector.zero
    elif len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        v = coord_sys.base_vectors()
        x = coord_sys.base_scalars()

        if coord_sys.is_cartesian and field.is_scalar:
            h = coord_sys.hi
            vi = Vector.zero
            for i in range(len(h)):
                vi += v[i] * field.diff(x[i]) / h[i]

            if doit:
                return (vi).doit()
            return vi
        else:
            gt = coord_sys.get_contravariant_metric_tensor()
            vi = []
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]):
                    if field.is_Vector:
                        if transpose:
                            vi.append(v[j] | field.diff(x[i]) * gt[i, j])
                        else:
                            vi.append(field.diff(x[i]) * gt[i, j] | v[j])
                    else:
                        vi.append(field.diff(x[i]) * gt[i, j] * v[j])
            vi: DyadicAdd | VectorAdd = (
                DyadicAdd(*vi) if field.is_Vector else VectorAdd(*vi)
            )

            if doit:
                return vi.doit()
            return vi

    elif len(coord_sys) == 2:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        coord_sys_arr = np.array(list(coord_sys))
        mask = np.array(
            [not cast(_IsCartesian, si).is_cartesian for si in coord_sys_arr],
            dtype=bool,
        )
        not_cart = np.nonzero(mask)[0]
        assert len(not_cart) == 1
        return gradient(
            cast(CoordSys, coord_sys_arr[not_cart[0]]).from_cartesian(field),
            doit=doit,
            transpose=transpose,
        )

    else:
        if isinstance(field, Add | VectorAdd):
            return VectorAdd.fromiter(
                gradient(i, doit=doit, transpose=transpose) for i in field.args
            )
        if isinstance(field, Mul | VectorMul):
            s = _split_mul_args_wrt_coordsys(field)
            return VectorAdd.fromiter(
                field / i * gradient(i, doit=doit, transpose=transpose) for i in s
            )
        return Grad(field, transpose=transpose)


def curl(v: Vector, doit: bool = True) -> Vector | Curl:
    """Return curl of a 3D vector field.

        curl(v) = ∇×v = b^j×(∂v/∂q^j) = ε^{ijk} ∂v_k/∂q^j b_i / √g

    where {b^j} are the contravariant basis vectors, q^j the coordinates,
    ε^{ijk} the Levi-Civita symbol, and √g the scale factor product
    (square root of determinant of the Jacobian of the coordinate transformation).

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
    assert v.is_Vector

    coord_sys = _get_coord_systems(v)

    if len(coord_sys) == 0:
        return VectorZero()
    elif len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        bv = coord_sys.base_vectors()
        x = coord_sys.base_scalars()
        sg = coord_sys.sg
        comp = coord_sys.get_covariant_component
        assert len(x) == 3
        v0 = Derivative(comp(v, 2), x[1]) - Derivative(comp(v, 1), x[2])
        v1 = Derivative(comp(v, 0), x[2]) - Derivative(comp(v, 2), x[0])
        v2 = Derivative(comp(v, 1), x[0]) - Derivative(comp(v, 0), x[1])
        outvec = (v0 * bv[0] + v1 * bv[1] + v2 * bv[2]) / sg
        if doit:
            return outvec.doit()
        return outvec

    elif len(coord_sys) == 2:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        coord_sys_arr = np.array(list(coord_sys))
        mask = np.array(
            [not cast(_IsCartesian, si).is_cartesian for si in coord_sys_arr],
            dtype=bool,
        )
        not_cart = np.nonzero(mask)[0]
        assert len(not_cart) == 1
        return curl(
            cast(Vector, cast(CoordSys, coord_sys_arr[not_cart[0]]).from_cartesian(v)),
            doit=doit,
        )

    else:
        if isinstance(v, Add | VectorAdd):
            from sympy.vector import express

            try:
                cs = next(iter(coord_sys))
                args = [express(i, cs, variables=True) for i in v.args]
            except ValueError:
                args = v.args
            return VectorAdd.fromiter(curl(cast(Vector, i), doit=doit) for i in args)
        elif isinstance(v, Mul | VectorMul):
            vector = [i for i in v.args if isinstance(i, Vector | Cross | Gradient)][0]
            scalar = Mul.fromiter(
                i for i in v.args if not isinstance(i, Vector | Cross | Gradient)
            )
            res = Cross(gradient(scalar), vector).doit() + scalar * curl(
                cast(Vector, vector), doit=doit
            )
            if doit:
                return res.doit()
            return res
        elif isinstance(v, Cross | Curl | Gradient):
            return Curl(v)
        else:
            raise Curl(v)  # type: ignore[invalid-raise]


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

    def __new__(cls, expr, transpose: bool = False) -> Self:
        expr = sp.sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        obj._transpose = False if expr.is_scalar else transpose
        return obj

    def doit(self, **hints: Any) -> BasisDependent | Grad:
        return gradient(self._expr.doit(**hints), doit=True, transpose=self._transpose)

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
            f"\\displaystyle (\\nabla {printer._print(self._expr)})^T"
            if self._transpose
            else f"\\displaystyle \\nabla {printer._print(self._expr)}"
        )


class Div(Divergence):
    """Unevaluated divergence wrapper using custom curvilinear implementation."""

    _expr: Expr

    def doit(self, **hints: Any) -> Vector | Expr | Basic | Div:
        return divergence(self._expr.doit(**hints), doit=True)


class Curl(sympy_Curl):
    """Unevaluated curl wrapper using custom curvilinear implementation."""

    _expr: Expr

    def doit(self, **hints: Any) -> Vector | Curl:
        return curl(self._expr.doit(**hints), doit=True)


class Dot(sympy_Dot):
    """Unevaluated dot product wrapper delegating to custom dot().

    Args:
        expr1: Left tensor.
        expr2: Right tensor.
    """

    _expr1: Expr
    _expr2: Expr

    def __new__(cls, expr1, expr2) -> Self:
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints: Any) -> BasisDependent | Expr | Dot:
        return dot(
            cast(Vector | Dyadic, self._expr1.doit(**hints)),
            cast(Vector | Dyadic, self._expr2.doit(**hints)),
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

    _expr1: Expr
    _expr2: Expr

    def __new__(cls, expr1, expr2) -> Self:
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints: Any) -> Vector | Cross:
        return cross(
            cast(Vector, self._expr1.doit(**hints)),
            cast(Vector, self._expr2.doit(**hints)),
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

    _expr1: Expr
    _expr2: Expr

    def __new__(cls, expr1, expr2) -> Self:
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    @property
    def T(self) -> Outer:
        return Outer(self._expr2, self._expr1)

    def transpose(self) -> Outer:
        return self.T

    def doit(self, **hints) -> Dyadic | Outer:
        return outer(self._expr1.doit(), self._expr2.doit())


class Source(Expr):
    _expr: Expr

    def __new__(cls, expr) -> Self:
        expr = sp.sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        return obj

    def doit(self, **hints) -> Expr:
        return self._expr.doit(**hints)


class Constant(sp.Symbol):
    val: sp.Expr | Number

    def __new__(cls, name: str, val: sp.Expr | Number | float, **assumptions) -> Self:
        obj = super().__new__(cls, name, **assumptions)
        obj.val = val
        return obj

    def doit(self, **hints) -> sp.Expr | Number:
        return self.val


class Identity(sp.Expr):
    def __init__(self, sys: CoordSys) -> None:
        self.sys = sys

    def doit(self) -> Dyadic:  # type: ignore[override]
        return sum(self.sys.base_dyadics()[:: self.sys.dims + 1], DyadicZero())


def diff(self, *args, **kwargs) -> BasisDependent:
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
    # Move to Cartesian because the basis vectors are then constant and
    # non-differentiable. Alternatively use Christoffel symbols, but this gets messy
    # for more than one args.
    v0 = self._sys.to_cartesian(self)
    diff_components = [df(v, *args, **kwargs) * k for k, v in v0.components.items()]
    f = self._add_func(*diff_components)
    return self._sys.from_cartesian(f)


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
# h = 4*Grad(f)
# h.doit().is_Vector
# # -> True


def doit(self, **hints) -> Basic:
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
sp.vector.basisdependent.BasisDependent.diff = diff  # ty:ignore[possibly-missing-attribute]
sp.vector.Dyadic.is_Dyadic = True  # ty:ignore[possibly-missing-attribute]
