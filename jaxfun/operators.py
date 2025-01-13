from typing import Set, Any
import sympy as sp
import numpy as np
from sympy import Expr
from sympy.core import preorder_traversal
from sympy.vector.vector import Vector
from sympy.vector import VectorAdd, VectorMul, VectorZero
from sympy.vector.operators import (
    Divergence,
    Gradient,
    _split_mul_args_wrt_coordsys,
)
from sympy.vector.operators import Curl as sympy_Curl
from sympy.vector.vector import Cross as sympy_Cross
from sympy.vector import Dot as sympy_Dot
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.function import Derivative
from jaxfun.coordinates import BaseVector
from jaxfun.coordinates import CoordSys


def eijk(i: int, j: int, k: int) -> int:
    if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        return 1
    elif (i, j, k) in ((2, 1, 0), (1, 0, 2), (0, 2, 1)):
        return -1
    return 0


def _get_coord_systems(expr: Expr) -> Set:
    g = preorder_traversal(expr)
    ret = set()
    for i in g:
        if isinstance(i, CoordSys):
            ret.add(i)
            g.skip()
    return frozenset(ret)


def cross(vect1: Vector, vect2: Vector) -> Vector:
    """
    Returns cross product of two vectors.

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import dot
    >>> import sympy as sp
    >>> r, theta = sp.symbols('r,theta', real=True)
    >>> C = get_CoordSys('C', sp.Lambda((r, theta, z), (r*sp.cos(theta), r*sp.sin(theta), z)))
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
                p = np.setxor1d(range(3), (vect1._id[0], vect2._id[0]))[0]
                ei = eijk(vect1._id[0], vect2._id[0], p)
                C = vect1._sys
                b = C.base_vectors()
                s = gt[p, 0] * b[0]
                for i in range(1, gt.shape[1]):
                    s += gt[p, i] * b[i]
                return sg * ei * s

        return cross(
            vect1._sys.to_cartesian_vector(vect1), vect2._sys.to_cartesian_vector(vect2)
        )

    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1 * cross(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2 * cross(vect1, v2)

    return Cross(vect1, vect2)


def dot(vect1: Vector, vect2: Vector) -> Expr:
    """
    Returns dot product of two vectors.

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import dot
    >>> import sympy as sp
    >>> r, theta = sp.symbols('r,theta', real=True)
    >>> P = get_CoordSys('P', sp.Lambda((r, theta), (r*sp.cos(theta), r*sp.sin(theta))))
    >>> v1 = P.b_r + P.b_theta
    >>> v2 = P.r * P.b_r + P.theta * P.b_theta
    >>> dot(v1, v2)
    r**2*theta + r

    """
    if isinstance(vect1, Add):
        return Add.fromiter(dot(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return Add.fromiter(dot(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            if vect1._sys.is_cartesian:
                return sp.S.One if vect1 == vect2 else sp.S.Zero
            else:
                g = vect1._sys.get_covariant_metric_tensor()
                return g[vect1._id[0], vect2._id[0]]

        return dot(
            vect1._sys.to_cartesian_vector(vect1), vect2._sys.to_cartesian_vector(vect2)
        )

    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return sp.S.Zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1 * dot(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2 * dot(vect1, v2)

    return Dot(vect1, vect2)


def divergence(vect: Vector, doit: bool = True) -> Expr:
    """
    Returns the divergence of a vector field computed wrt the base
    scalars of the given coordinate system.

    Parameters
    ==========

    vector : Vector
        The vector operand

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import divergence
    >>> import sympy as sp
    >>> r, theta = sp.symbols('r,theta', real=True)
    >>> P = get_CoordSys('P', sp.Lambda((r, theta), (r*sp.cos(theta), r*sp.sin(theta))))
    >>> v = P.r*P.b_r + P.theta*P.b_theta
    >>> divergence(v)
    3

    """
    coord_sys = _get_coord_systems(vect)
    if len(coord_sys) == 0:
        return sp.S.Zero
    elif len(coord_sys) == 1:
        if isinstance(vect, (Cross, Curl, Gradient)):
            return Divergence(vect)
        # vect should be a vector with Cartesian or covariant basis vectors
        coord_sys = next(iter(coord_sys))
        x = coord_sys.base_scalars()
        comp = coord_sys.get_contravariant_component
        sg = coord_sys.sg
        res = sp.S.Zero
        for i in range(len(x)):
            res += Derivative(comp(vect, i) * sg, x[i]) / sg

        if doit:
            return res.doit()
        return res
    else:
        if isinstance(vect, (Add, VectorAdd)):
            return Add.fromiter(divergence(i, doit=doit) for i in vect.args)
        elif isinstance(vect, (Mul, VectorMul)):
            vector = [i for i in vect.args if isinstance(i, (Vector, Cross, Gradient))][
                0
            ]
            scalar = Mul.fromiter(
                i for i in vect.args if not isinstance(i, (Vector, Cross, Gradient))
            )
            res = Dot(vector, gradient(scalar)) + scalar * divergence(vector, doit=doit)
            if doit:
                return res.doit()
            return res
        elif isinstance(vect, (Cross, Curl, Gradient)):
            return Divergence(vect)
        else:
            raise Divergence(vect)


def gradient(scalar_field: Expr, doit: bool = True) -> Vector:
    """
    Returns the vector gradient of a scalar field computed wrt the
    base scalars of the given coordinate system.

    Parameters
    ==========

    scalar_field : SymPy Expr
        The scalar field to compute the gradient of

    doit : bool
        If True, the result is returned after calling .doit() on
        each component. Else, the returned expression contains
        Derivative instances

    Examples
    ========

    >>> from jaxfun import get_CoordSys
    >>> from jaxfun.operators import gradient
    >>> import sympy as sp
    >>> r, theta = sp.symbols('r,theta', real=True)
    >>> P = get_CoordSys('P', sp.Lambda((r, theta), (r*sp.cos(theta), r*sp.sin(theta))))
    >>> s = P.r*P.theta
    >>> gradient(s)
    theta*C.b_r + 1/r*C.b_theta

    """
    coord_sys = _get_coord_systems(scalar_field)

    if len(coord_sys) == 0:
        return Vector.zero
    elif len(coord_sys) == 1:
        coord_sys = next(iter(coord_sys))
        v = coord_sys.base_vectors()
        x = coord_sys.base_scalars()

        if coord_sys.is_cartesian:
            h = coord_sys.hi
            vi = sp.vector.VectorZero()
            for i in range(len(h)):
                vi += v[i] * Derivative(scalar_field, x[i]) / h[i]

            if doit:
                return (vi).doit()
            return vi
        else:
            gt = coord_sys.get_contravariant_metric_tensor()
            vi = sp.vector.VectorZero()
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]):
                    vi = vi + v[j] * Derivative(scalar_field, x[i]) * gt[i, j]

            if doit:
                return vi.doit()
            return vi

    else:
        if isinstance(scalar_field, (Add, VectorAdd)):
            return VectorAdd.fromiter(gradient(i) for i in scalar_field.args)
        if isinstance(scalar_field, (Mul, VectorMul)):
            s = _split_mul_args_wrt_coordsys(scalar_field)
            return VectorAdd.fromiter(scalar_field / i * gradient(i) for i in s)
        return Gradient(scalar_field)


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
    >>> r, theta, z = sp.symbols('r,theta,z', real=True)
    >>> P = get_CoordSys('P', sp.Lambda((r, theta, z), (r*sp.cos(theta), r*sp.sin(theta), z)))
    >>> v = P.b_r + P.b_theta
    >>> curl(v)
    2*C.b_z

    """

    coord_sys = _get_coord_systems(vect)

    if len(coord_sys) == 0:
        return Vector.zero
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
    else:
        if isinstance(vect, (Add, VectorAdd)):
            from sympy.vector import express

            try:
                cs = next(iter(coord_sys))
                args = [express(i, cs, variables=True) for i in vect.args]
            except ValueError:
                args = vect.args
            return VectorAdd.fromiter(curl(i, doit=doit) for i in args)
        elif isinstance(vect, (Mul, VectorMul)):
            vector = [i for i in vect.args if isinstance(i, (Vector, Cross, Gradient))][
                0
            ]
            scalar = Mul.fromiter(
                i for i in vect.args if not isinstance(i, (Vector, Cross, Gradient))
            )
            res = Cross(gradient(scalar), vector).doit() + scalar * curl(
                vector, doit=doit
            )
            if doit:
                return res.doit()
            return res
        elif isinstance(vect, (Cross, Curl, Gradient)):
            return Curl(vect)
        else:
            raise Curl(vect)


class Grad(Gradient):
    def doit(self, **hints: dict[Any]) -> Expr:
        return gradient(self._expr.doit(**hints), doit=True)


class Div(Divergence):
    def doit(self, **hints: dict[Any]) -> Expr:
        return divergence(self._expr.doit(**hints), doit=True)


class Curl(sympy_Curl):
    def doit(self, **hints: dict[Any]) -> Expr:
        return curl(self._expr.doit(**hints), doit=True)


class Dot(sympy_Dot):
    def doit(self, **hints: dict[Any]) -> Expr:
        return dot(self._expr1.doit(), self._expr2.doit())

    
class Cross(sympy_Cross):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """

    #def __new__(cls, expr1, expr2):
    #    expr1 = sp.sympify(expr1)
    #    expr2 = sp.sympify(expr2)
    #    obj = Expr.__new__(cls, expr1, expr2)
    #    obj._expr1 = expr1
    #    obj._expr2 = expr2
    #    return obj

    def doit(self, **hints: dict[Any]) -> Expr:
        return cross(self._expr1.doit(), self._expr2.doit())



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
