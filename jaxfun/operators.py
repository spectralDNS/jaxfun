import collections
from itertools import product
from typing import Any

import numpy as np
import sympy as sp
from sympy import Expr
from sympy.core import preorder_traversal
from sympy.core.add import Add
from sympy.core.function import Derivative
from sympy.core.function import diff as df
from sympy.core.mul import Mul
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import Dot as sympy_Dot
from sympy.vector import VectorAdd, VectorMul, VectorZero
from sympy.vector.dyadic import Dyadic, DyadicAdd, DyadicMul, DyadicZero
from sympy.vector.operators import Curl as sympy_Curl
from sympy.vector.operators import (
    Divergence,
    Gradient,
)
from sympy.vector.vector import Cross as sympy_Cross
from sympy.vector.vector import Vector

from jaxfun.coordinates import BaseDyadic, BaseScalar, BaseVector, CoordSys


def eijk(i: int, j: int, k: int) -> int:
    if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        return 1
    elif (i, j, k) in ((2, 1, 0), (1, 0, 2), (0, 2, 1)):
        return -1
    return 0


def sign(i: int, j: int):
    return 1 if ((i + 1) % 3 == j) else -1


def _get_coord_systems(expr: Expr) -> set:
    g = preorder_traversal(expr)
    ret = set()
    for i in g:
        if isinstance(i, CoordSys):
            ret.add(i)
            g.skip()
    return frozenset(ret)


def _split_mul_args_wrt_coordsys(expr):
    d = collections.defaultdict(lambda: sp.S.One)
    for i in expr.args:
        d[_get_coord_systems(i)] *= i
    return list(d.values())


def express(expr, system):
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


def outer(vect1: Vector, vect2: Vector) -> Expr:
    """
    Returns outer product of two vectors.

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import dot
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
    if isinstance(vect1, Add | VectorAdd):
        return DyadicAdd.fromiter(outer(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add | VectorAdd):
        return DyadicAdd.fromiter(outer(vect1, i) for i in vect2.args)
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1 * outer(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2 * outer(vect1, v2)

    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Dyadic.zero

    args = [
        (v1 * v2) * BaseDyadic(k1, k2)
        for (k1, v1), (k2, v2) in product(
            vect1.components.items(), vect2.components.items()
        )
    ]

    return DyadicAdd(*args)


def cross(vect1: Vector, vect2: Vector) -> Vector:
    """
    Returns cross product of two vectors.

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import dot
    >>> import sympy as sp
    >>> r, theta, z = sp.symbols("r,theta,z", real=True)
    >>> C = get_CoordSys(
    ...     "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    ... )
    >>> cross(C.b_r, C.b_theta)
    r*C.b_z

    """
    if isinstance(vect1, Add):
        return VectorAdd.fromiter(cross(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return VectorAdd.fromiter(cross(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            n1 = vect1.args[0]
            n2 = vect2.args[0]
            if n1 == n2:
                return Vector.zero

            if vect1._sys.is_cartesian:
                n3 = ({0, 1, 2}.difference({n1, n2})).pop()
                sign = 1 if ((n1 + 1) % 3 == n2) else -1
                return sign * vect1._sys.base_vectors()[n3]
            else:
                assert len(vect1._sys.base_scalars()) == 3, (
                    "Can only compute cross product in 3D"
                )
                gt = vect1._sys.get_contravariant_metric_tensor()
                sg = vect1._sys.sg
                n3 = ({0, 1, 2}.difference({n1, n2})).pop()
                ei = eijk(n1, n2, n3)
                C = vect1._sys
                b = C.base_vectors()
                s = gt[n3, 0] * b[0]
                for i in range(1, gt.shape[1]):
                    s += gt[n3, i] * b[i]
                return sg * ei * s

        return cross(vect1._sys.to_cartesian(vect1), vect2._sys.to_cartesian(vect2))

    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1 * cross(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2 * cross(vect1, v2)

    return Cross(vect1, vect2)


def dot(vect1: Vector | Dyadic, vect2: Vector | Dyadic) -> Expr:
    """
    Returns dot product of two tensors.

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import dot
    >>> import sympy as sp
    >>> r, theta = sp.symbols("r,theta", real=True)
    >>> P = get_CoordSys(
    ...     "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
    ... )
    >>> v1 = P.b_r + P.b_theta
    >>> v2 = P.r * P.b_r + P.theta * P.b_theta
    >>> dot(v1, v2)
    r**2*theta + r

    """
    rank: int = 0
    if (isinstance(vect1, Vector) and isinstance(vect2, Dyadic)) or (
        isinstance(vect1, Dyadic) and isinstance(vect2, Vector)
    ):
        rank = 1
    elif isinstance(vect1, Dyadic) and isinstance(vect2, Dyadic):
        rank = 2

    if isinstance(vect1, Add | VectorAdd | DyadicAdd):
        if rank == 0:
            return Add.fromiter(dot(i, vect2) for i in vect1.args)
        elif rank == 1:
            return VectorAdd.fromiter(dot(i, vect2) for i in vect1.args)
        else:
            return DyadicAdd.fromiter(dot(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add | VectorAdd | DyadicAdd):
        if rank == 0:
            return Add.fromiter(dot(vect1, i) for i in vect2.args)
        elif rank == 1:
            return VectorAdd.fromiter(dot(vect1, i) for i in vect2.args)
        else:
            return DyadicAdd.fromiter(dot(vect1, i) for i in vect2.args)

    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            if vect1._sys.is_cartesian:
                return sp.S.One if vect1 == vect2 else sp.S.Zero
            else:
                g = vect1._sys.get_covariant_metric_tensor()
                return g[vect1._id[0], vect2._id[0]]

        return dot(vect1._sys.to_cartesian(vect1), vect2._sys.to_cartesian(vect2))

    if isinstance(vect1, BaseDyadic) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            if vect1._sys.is_cartesian:
                return (
                    vect1.args[0] if vect1.args[1] == vect2 else sp.vector.VectorZero()
                )
            else:
                g = vect1._sys.get_covariant_metric_tensor()
                g0 = g[vect1.args[1]._id[0], vect2._id[0]]
                if g0 == 0:
                    return VectorZero()
                else:
                    return g0 * vect1.args[0]

        return dot(vect1._sys.to_cartesian(vect1), vect2._sys.to_cartesian(vect2))

    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseDyadic):
        if vect1._sys == vect2._sys:
            if vect1._sys.is_cartesian:
                return (
                    vect2.args[1] if vect1 == vect2.args[0] else sp.vector.VectorZero()
                )
            else:
                g = vect1._sys.get_covariant_metric_tensor()
                g0 = g[vect1._id[0], vect2.args[0]._id[0]]
                if g0 == 0:
                    return VectorZero()
                else:
                    return g0 * vect2.args[1]

        return dot(vect1._sys.to_cartesian(vect1), vect2._sys.to_cartesian(vect2))

    if isinstance(vect1, BaseDyadic) and isinstance(vect2, BaseDyadic):
        if vect1._sys == vect2._sys:
            if vect1._sys.is_cartesian:
                return (
                    vect1.args[0] | vect2.args[1]
                    if vect1.args[1] == vect2.args[0]
                    else DyadicZero()
                )
            else:
                g = vect1._sys.get_covariant_metric_tensor()
                g0 = g[vect1.args[1]._id[0], vect2.args[0]._id[0]]
                if g0 == 0:
                    return DyadicZero()
                else:
                    return g0 * vect1.args[0] | vect2.args[1]

        return dot(vect1._sys.to_cartesian(vect1), vect2._sys.to_cartesian(vect2))
    if isinstance(vect1, DyadicZero) and isinstance(vect2, BaseVector):
        return VectorZero()
    if isinstance(vect1, DyadicZero) and isinstance(vect2, BaseDyadic):
        return DyadicZero()
    if isinstance(vect1, BaseVector) and isinstance(vect2, DyadicZero):
        return VectorZero()
    if isinstance(vect1, BaseDyadic) and isinstance(vect2, DyadicZero):
        return DyadicZero()
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return sp.S.Zero
    if isinstance(vect1, VectorMul | DyadicMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1 * dot(v1, vect2)
    if isinstance(vect2, VectorMul | DyadicMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2 * dot(vect1, v2)

    return Dot(vect1, vect2)


def divergence(vect: Vector | Dyadic, doit: bool = True) -> Expr:
    """
    Returns the divergence of a vector/dyadic field computed wrt the base
    scalars of the given coordinate system.

    Parameters
    ==========

    vect : Vector | Dyadic
        The vector/dyadic operand

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import divergence
    >>> import sympy as sp
    >>> r, theta = sp.symbols("r,theta", real=True)
    >>> P = get_CoordSys(
    ...     "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
    ... )
    >>> v = P.r * P.b_r + P.theta * P.b_theta
    >>> divergence(v)
    3

    """
    coord_sys = _get_coord_systems(vect)
    if len(coord_sys) == 0:
        if vect.is_Vector:
            return sp.S.Zero
        else:
            return sp.vector.VectorZero()
    elif len(coord_sys) == 1:
        if isinstance(vect, Cross | Curl | Gradient):
            return Div(vect)
        # vect should be a vector/dyadic with Cartesian or covariant basis vectors
        coord_sys = next(iter(coord_sys))
        x = coord_sys.base_scalars()
        comp = coord_sys.get_contravariant_component
        sg = coord_sys.sg
        if vect.is_Vector:
            res = sp.S.Zero
            for i in range(len(x)):
                res += Derivative(express(comp(vect, i) * sg, coord_sys), x[i]) / sg
        else:
            bv = coord_sys.base_vectors()
            ct = coord_sys.get_christoffel_second()
            res = []
            for i in range(len(x)):
                r0 = sp.S.Zero
                for j in range(len(x)):
                    r0 += comp(vect, i, j).diff(x[j])
                    for k in range(len(x)):
                        r0 += ct[i, k, j] * comp(vect, k, j) + ct[j, k, j] * comp(
                            vect, i, k
                        )
                res.append(r0 * bv[i])
            res = VectorAdd(*res)

        if doit:
            return express(res.doit(), coord_sys)
        return res

    elif len(coord_sys) == 2:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        coord_sys_arr = np.array(list(coord_sys))
        not_cart = np.nonzero([not si.is_cartesian for si in coord_sys_arr])[0]
        assert len(not_cart) == 1
        return divergence(coord_sys_arr[not_cart[0]].from_cartesian(vect), doit=doit)

    else:
        if isinstance(vect, DyadicAdd):
            return VectorAdd.fromiter(divergence(i, doit=doit) for i in vect.args)
        elif isinstance(vect, Add | VectorAdd):
            return Add.fromiter(divergence(i, doit=doit) for i in vect.args)
        elif isinstance(vect, Mul | VectorMul):
            vector = [
                i
                for i in vect.args
                if isinstance(i, Dyadic | Vector | Cross | Gradient)
            ][0]
            scalar = Mul.fromiter(
                i
                for i in vect.args
                if not isinstance(i, Dyadic | Vector | Cross | Gradient)
            )
            if not vector.is_Vector:
                res = sp.vector.VectorAdd(
                    dot(vector, gradient(scalar)),
                    scalar * divergence(vector, doit=doit),
                )
            else:
                res = Dot(vector, gradient(scalar)) + scalar * divergence(
                    vector, doit=doit
                )
            if doit:
                return res.doit()
            return res
        elif isinstance(vect, Cross | Curl | Gradient):
            return Div(vect)
        else:
            raise Div(vect)


def gradient(field: Expr, doit: bool = True, transpose: bool = False) -> Vector:
    """
    Returns the gradient of a scalar/vector field computed wrt the
    base scalars of the given coordinate system.

    Parameters
    ==========

    field : SymPy Expr
        The field to compute the gradient of

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import gradient
    >>> import sympy as sp
    >>> r, theta = sp.symbols("r,theta", real=True)
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
            vi = VectorZero()
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
            vi = DyadicAdd(*vi) if field.is_Vector else VectorAdd(*vi)

            if doit:
                return vi.doit()
            return vi

    elif len(coord_sys) == 2:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        coord_sys_arr = np.array(list(coord_sys))
        not_cart = np.nonzero([not si.is_cartesian for si in coord_sys_arr])[0]
        assert len(not_cart) == 1
        return gradient(
            coord_sys_arr[not_cart[0]].from_cartesian(field),
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


def curl(vect: Vector, doit: bool = True) -> Vector:
    """
    Returns the curl of a vector field computed wrt the base scalars
    of the given coordinate system.

    Parameters
    ==========

    vect : Vector
        The vector operand

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========
    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import curl
    >>> import sympy as sp
    >>> r, theta, z = sp.symbols("r,theta,z", real=True)
    >>> P = get_CoordSys(
    ...     "P", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    ... )
    >>> v = P.b_r + P.b_theta
    >>> curl(v)
    2*P.b_z

    """
    assert vect.is_Vector

    coord_sys = _get_coord_systems(vect)

    if len(coord_sys) == 0:
        return VectorZero()
    elif len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        v = coord_sys.base_vectors()
        x = coord_sys.base_scalars()
        sg = coord_sys.sg
        comp = coord_sys.get_covariant_component
        assert len(x) == 3
        v0 = Derivative(comp(vect, 2), x[1]) - Derivative(comp(vect, 1), x[2])
        v1 = Derivative(comp(vect, 0), x[2]) - Derivative(comp(vect, 2), x[0])
        v2 = Derivative(comp(vect, 1), x[0]) - Derivative(comp(vect, 0), x[1])
        outvec = (v0 * v[0] + v1 * v[1] + v2 * v[2]) / sg
        if doit:
            return outvec.doit()
        return outvec

    elif len(coord_sys) == 2:
        # For example a vector expressed using Cartesian basis vectors and
        # Curvilinear BaseScalars. Like CoordSys.position_vector(True)
        coord_sys_arr = np.array(list(coord_sys))
        not_cart = np.nonzero([not si.is_cartesian for si in coord_sys_arr])[0]
        assert len(not_cart) == 1
        return curl(coord_sys_arr[not_cart[0]].from_cartesian(vect), doit=doit)

    else:
        if isinstance(vect, Add | VectorAdd):
            from sympy.vector import express

            try:
                cs = next(iter(coord_sys))
                args = [express(i, cs, variables=True) for i in vect.args]
            except ValueError:
                args = vect.args
            return VectorAdd.fromiter(curl(i, doit=doit) for i in args)
        elif isinstance(vect, Mul | VectorMul):
            vector = [i for i in vect.args if isinstance(i, Vector | Cross | Gradient)][
                0
            ]
            scalar = Mul.fromiter(
                i for i in vect.args if not isinstance(i, Vector | Cross | Gradient)
            )
            res = Cross(gradient(scalar), vector).doit() + scalar * curl(
                vector, doit=doit
            )
            if doit:
                return res.doit()
            return res
        elif isinstance(vect, Cross | Curl | Gradient):
            return Curl(vect)
        else:
            raise Curl(vect)


class Grad(Gradient):
    def __new__(cls, expr, transpose: bool = False):
        expr = sp.sympify(expr)
        obj = Expr.__new__(cls, expr)
        obj._expr = expr
        obj._transpose = False if expr.is_scalar else transpose
        return obj

    def doit(self, **hints: dict[Any]) -> Expr:
        return gradient(self._expr.doit(**hints), doit=True, transpose=self._transpose)

    @property
    def T(self):
        if self._expr.doit().is_scalar:
            return self
        return Grad(self._expr, transpose=not self._transpose)

    def __str__(self) -> str:
        w = "Grad(" + self._expr.__str__() + ")"
        return w + ".T" if self._transpose else w

    def _pretty(self, printer: Any = None) -> str:
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
    def doit(self, **hints: dict[Any]) -> Expr:
        return divergence(self._expr.doit(**hints), doit=True)


class Curl(sympy_Curl):
    def doit(self, **hints: dict[Any]) -> Expr:
        return curl(self._expr.doit(**hints), doit=True)


class Dot(sympy_Dot):
    def __new__(cls, expr1, expr2):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints: dict[Any]) -> Expr:
        return dot(self._expr1.doit(), self._expr2.doit())


class Cross(sympy_Cross):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from jaxfun.arguments import CartCoordSys, x, y, z
    >>> from jaxfun.operators import Cross
    >>> N = CartCoordSys("N", (x, y, z))
    >>> v1 = N.i + N.j + N.k
    >>> v2 = N.x * N.i + N.y * N.j + N.z * N.k
    >>> Cross(v1, v2)
    Cross(N.i + N.j + N.k, x*N.i + y*N.j + z*N.k)
    >>> Cross(v1, v2).doit()
    (-y + z)*N.i + (x - z)*N.j + (-x + y)*N.k

    """

    def __new__(cls, expr1, expr2):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints: dict[Any]) -> Expr:
        return cross(self._expr1.doit(), self._expr2.doit())


class Outer(Expr):
    """
    Represents unevaluated Outer product.

    Examples
    ========

    >>> from jaxfun.arguments import CartCoordSys, x, y
    >>> from jaxfun.operators import Outer
    >>> N = CartCoordSys("N", (x, y))
    >>> v1 = N.i
    >>> v2 = N.j
    >>> Outer(v1, v2)
    Outer(N.i, N.j)
    >>> Outer(v1, v2).doit()
    (N.i⊗N.j)

    """

    def __new__(cls, expr1, expr2):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    @property
    def T(self):
        return Outer(self._expr2, self._expr1)

    def transpose(self):
        return self.T

    def doit(self, **hints):
        return outer(self._expr1.doit(), self._expr2.doit())


def diff(self, *args, **kwargs):
    """
    Implements the SymPy diff routine, for vectors.

    diff's documentation
    ========================

    """
    for x in args:
        if isinstance(x, sp.vector.basisdependent.BasisDependent):
            raise TypeError("Invalid arg for differentiation")
    # Move to Cartesian because the basis vectors are then constant and non-differentiable
    # Alternatively use Christoffel symbols, but this gets messy for more than one args.
    v0 = self._sys.to_cartesian(self)
    diff_components = [df(v, *args, **kwargs) * k for k, v in v0.components.items()]
    f = self._add_func(*diff_components)
    return self._sys.from_cartesian(f)


# Regular doit() is problematic for vectors and dyadics when such are
# used unevaluated. For example
# from sympy.vector import CoordSys3D, Gardient
# N = CoordSys3D("N")
# f = N.x*N.y
# h = 4*Gradient(f)
# z = h.doit()
## -> z = 4*N.x*N.j + 4*N.y*N.i
# z.is_Vector
## -> False
# z is now a type Add and not VectorAdd. Using the doit function
# below is a hack around it
# from jaxfun.arguments import CartCoordSys, x, y, z
# from jaxfun.operators import Grad
# N = CartCoordSys("N", (x, y, z))
# f = N.x*N.y
# h = 4*Grad(f)
# h.doit().is_Vector
## -> True


def doit(self, **hints):
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
                return sp.vector.VectorAdd.fromiter(z.args)
            elif isinstance(p, BaseDyadic):
                return sp.vector.DyadicAdd.fromiter(z.args)

    elif isinstance(z, sp.Mul):
        # Check if should be VectorMul
        for p in sp.core.traversal.preorder_traversal(z):
            if isinstance(p, BaseVector):
                return sp.vector.VectorMul.fromiter(z.args)
            elif isinstance(p, BaseDyadic):
                return sp.vector.DyadicMul.fromiter(z.args)
    return z


sp.core.Expr.doit = doit
sp.vector.vector.dot = dot
sp.vector.vector.cross = cross
sp.vector.operators.gradient = gradient
sp.vector.operators.curl = curl
sp.vector.operators.divergence = divergence
sp.vector.vector.Cross = Cross
sp.vector.vector.Dot = Dot
sp.vector.operators.Curl = Curl
sp.vector.operators.Gradient = Grad
sp.vector.operators.Divergence = Div
sp.vector.Cross = Cross
sp.vector.Dot = Dot
sp.vector.Curl = Curl
sp.vector.Gradient = Grad
sp.vector.Divergence = Div
sp.vector.basisdependent.BasisDependent.diff = diff
sp.vector.Dyadic.is_Dyadic = True
