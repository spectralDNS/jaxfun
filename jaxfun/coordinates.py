from __future__ import annotations

import numbers
from collections import UserDict
from collections.abc import Iterable
from types import MethodType
from typing import Any

import numpy as np
import sympy as sp
from sympy.assumptions.ask import AssumptionKeys
from sympy.core import AtomicExpr, Expr, Lambda, Symbol, Tuple
from sympy.core.assumptions import StdFactKB
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector.vector import Vector

latex_sym_dict = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "zeta": r"\zeta",
    "eta": r"\eta",
    "theta": r"\theta",
    "iota": r"\iota",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "omicron": r"\omicron",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "upsilon": r"\upsilon",
    "phi": r"\phi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
}


class defaultdict(UserDict):
    def __missing__(self, key) -> str:
        return key


latex_symbols = defaultdict(latex_sym_dict)


class BaseScalar(AtomicExpr):
    """
    A coordinate symbol/base scalar.

    """

    def __new__(
        cls, index: int, system: CoordSys, pretty_str: str = None, latex_str: str = None
    ) -> BaseScalar:
        if pretty_str is None:
            pretty_str = f"x{index}"
        elif isinstance(pretty_str, sp.Symbol):
            pretty_str = pretty_str.name
        if latex_str is None:
            latex_str = f"x_{index}"
        elif isinstance(latex_str, sp.Symbol):
            latex_str = latex_str.name

        index = _sympify(index)
        system = _sympify(system)
        obj = super().__new__(cls, index, system)
        if index not in range(0, 3):
            raise ValueError("Invalid index specified.")
        # The _id is used for equating purposes, and for hashing
        obj._id = (index, system)
        obj._name = obj.name = system._variable_names[index]
        obj._pretty_form = "" + pretty_str
        obj._latex_form = latex_str
        obj._system = system

        return obj

    is_commutative = True
    is_symbol = True
    is_Symbol = True
    is_real = True
    is_positive = True

    @property
    def free_symbols(self) -> set:
        return {self}

    _diff_wrt = True

    def _eval_derivative(self, s: Symbol) -> sp.Number:
        if self == s:
            return sp.S.One
        return sp.S.Zero

    def _latex(self, printer: Any = None) -> str:
        return self._latex_form

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self._pretty_form)

    precedence = PRECEDENCE["Atom"]

    @property
    def system(self) -> CoordSys:
        return self._system

    def _sympystr(self, printer: Any) -> str:
        return self._name

    def doit(self, **hints: dict) -> BaseScalar:
        return self

    def to_symbol(self) -> Symbol:
        return self.system._map_base_scalar_to_symbol[self]


class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """

    def __new__(
        cls, index: int, system: CoordSys, pretty_str: str = None, latex_str: str = None
    ) -> BaseVector:
        if pretty_str is None:
            pretty_str = f"x{index}"
        if latex_str is None:
            latex_str = f"x_{index}"
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        # Verify arguments
        if index not in range(0, 3):
            raise ValueError("index must be 0, 1 or 2")
        name = system._vector_names[index]
        # Initialize an object
        obj = super().__new__(cls, sp.S(index), system)
        # Assign important attributes
        obj._base_instance = obj
        obj._components = {obj: sp.S.One}
        obj._measure_number = sp.S.One
        obj._name = system._name + "." + name
        obj._pretty_form = "" + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        # The _id is used for printing purposes
        obj._id = (index, system)
        assumptions = {"commutative": True}
        obj._assumptions = StdFactKB(assumptions)

        # This attr is used for re-expression to one of the systems
        # involved in the definition of the Vector. Applies to
        # VectorMul and VectorAdd too.
        obj._sys = system

        return obj

    @property
    def system(self) -> CoordSys:
        return self._system

    def _sympystr(self, printer: Any) -> str:
        return self._name

    def _sympyrepr(self, printer: Any) -> str:
        index, system = self._id
        return printer._print(system) + "." + system._vector_names[index]

    @property
    def free_symbols(self) -> set[Symbol]:
        return {self}


class CoordSys(Basic):
    """
    Represents a coordinate system.
    """

    def __new__(
        cls,
        name: str,
        transformation: Lambda = None,
        vector_names: list[str] = None,
        parent: CoordSys | None = None,
        assumptions: AssumptionKeys = True,
        replace: list[tuple] | tuple[tuple] = (),
        measure: Function = sp.count_ops,
    ) -> CoordSys:
        """
        Coordinate system

        Parameters
        ==========

        name : str
            The name of the new CoordSys instance.

        transformation : Lambda
            Transformation defined by the position vector

        vector_names : iterable(optional)
            Iterables of 3 strings each, with custom names for base
            vectors and base scalars of the new system respectively.
            Used for simple str printing.

        parent : CoordSys(optional) or None
            Should be a simple Cartesian CoordSys("N") or None

        assumptions : Sympy assumptions
            Assumptions for the Sympy refine, helping with simplifying

        replace : iterable(optional)
            Iterable of 2-tuples, replacing the first item with the
            second. For helping Sympy with simplifications.

        measure : Python function to replace Sympy's count_ops.
            For example, to discourage the use of powers in an
            expression use::

            def discourage_powers(expr):
                POW = sp.Symbol('POW')
                count = sp.count_ops(expr, visual=True)
                count = count.replace(POW, 100)
                count = count.replace(sp.Symbol, type(sp.S.One))
                return count

        """

        name = str(name)

        if not isinstance(name, str):
            raise TypeError("name should be a string")

        psi, position_vector = transformation.args
        variable_names = [i.name for i in psi]
        is_cartesian = False
        if np.all(np.array(position_vector) == np.array(psi)):
            is_cartesian = True
        if vector_names is None:
            if is_cartesian:
                vector_names = ["i", "j", "k"]
            else:
                vector_names = [f"b_{s}" for s in variable_names]

        obj = super().__new__(cls, Str(name), transformation)
        obj._name = name

        vector_names = list(vector_names)
        if is_cartesian:
            latex_vects = [r"\mathbf{{%s}}" % (x,) for x in vector_names]
        else:
            latex_vects = [
                r"\mathbf{b_{%s}}" % (latex_symbols[x],) for x in variable_names
            ]
        pretty_vects = vector_names

        obj._vector_names = vector_names

        # Create covariant basis vectors in case of curvilinear
        v = []
        for i in range(len(psi)):
            v.append(BaseVector(i, obj, pretty_vects[i], latex_vects[i]))

        obj._base_vectors = Tuple(*v)

        variable_names = list(variable_names)
        latex_scalars = [latex_symbols[x] for x in variable_names]
        pretty_scalars = variable_names

        obj._variable_names = variable_names
        obj._vector_names = vector_names

        base_scalars = []
        for i in range(len(psi)):
            base_scalars.append(BaseScalar(i, obj, pretty_scalars[i], latex_scalars[i]))
        obj._psi = psi
        obj._cartesian_xyz = base_scalars if parent is None else parent._cartesian_xyz

        obj._map_base_scalar_to_symbol = {k: v for k, v in zip(base_scalars, obj._psi, strict=False)}
        obj._map_symbol_to_base_scalar = {k: v for k, v in zip(obj._psi, base_scalars, strict=False)}

        position_vector = position_vector.xreplace(obj._map_symbol_to_base_scalar)
        obj._map_xyz_to_base_scalar = {
            k: v for k, v in zip(obj._cartesian_xyz, position_vector, strict=False)
        }

        # Add doit to Cartesian coordinates, such that x, y, x are evaluated in computational space as x(psi), y(psi), z(psi)
        if not is_cartesian:
            for s in obj._cartesian_xyz:
                s.doit = MethodType(
                    lambda self, **hints: obj._map_xyz_to_base_scalar[self], s
                )

        obj._base_scalars = Tuple(*base_scalars)
        obj._position_vector = position_vector
        obj._is_cartesian = is_cartesian
        obj._transformation = transformation
        obj._measure = measure
        obj._assumptions = assumptions
        obj._replace = replace
        obj._hi = None
        obj._b = None
        obj._bt = None
        obj._e = None
        obj._g = None
        obj._gt = None
        obj._gn = None
        obj._ct = None
        obj._det_g = {True: None, False: None}
        obj._sqrt_det_g = {True: None, False: None}
        obj._covariant_basis_map = {
            k: v for k, v in zip(range(len(obj._base_vectors)), obj._base_vectors, strict=False)
        }

        for i in range(len(base_scalars)):
            setattr(obj, variable_names[i], base_scalars[i])
            setattr(obj, vector_names[i], v[i])

        for k in obj._cartesian_xyz:
            setattr(obj, k.name, k)

        # Assign params
        obj._parent = parent
        if obj._parent is not None:
            obj._root = obj._parent._root
        else:
            obj._root = obj

        # Return the instance
        return obj

    def sub_system(self, index: int = 0) -> SubCoordSys:
        return SubCoordSys(self, index)

    @property
    def dims(self) -> int:
        return len(self._base_scalars)

    def _sympystr(self, printer: Any) -> str:
        return self._name

    def __iter__(self) -> Iterable[BaseVector]:
        return iter(self.base_vectors())

    def base_vectors(self) -> tuple[BaseVector]:
        return self._base_vectors

    def base_scalars(self) -> tuple[BaseScalar]:
        return self._base_scalars

    @property
    def rv(self) -> tuple[Expr]:
        return self._position_vector

    @property
    def psi(self) -> tuple[Symbol]:
        return self._base_scalars

    @property
    def b(self) -> np.ndarray[Any, np.dtype[object]]:
        return self.get_covariant_basis()

    @property
    def bt(self) -> np.ndarray[Any, np.dtype[object]]:
        return self.get_contravariant_basis()

    @property
    def e(self) -> np.ndarray[Any, np.dtype[object]]:
        return self.get_normal_basis()

    @property
    def hi(self) -> np.ndarray[Any, np.dtype[object]]:
        return self.get_scaling_factors()

    @property
    def sg(self) -> Expr:
        if self.is_cartesian:
            return 1
        return self.get_sqrt_det_g(True)

    @property
    def is_orthogonal(self) -> bool:
        return sp.Matrix(self.get_covariant_metric_tensor()).is_diagonal()

    @property
    def is_cartesian(self) -> bool:
        return self._is_cartesian

    def to_cartesian_vector(self, v) -> Vector:
        # v either Cartesian or a vector with covariant basis vectors
        if v._sys.is_cartesian:
            return v

        cart_map = {
            k: v for k, v in zip(self.base_vectors(), self.get_covariant_basis(True), strict=False)
        }
        return v.xreplace(cart_map)

    def expr_base_scalar_to_psi(self, v) -> Expr:
        return sp.sympify(v).xreplace(self._map_base_scalar_to_symbol)

    def expr_psi_to_base_scalar(self, v: Expr) -> Expr:
        return sp.sympify(v).xreplace(self._map_symbol_to_base_scalar)

    def components(self, v: Vector = None) -> dict[BaseVector, Any]:
        c = {k: 0 for k in self.base_vectors()}
        if v is not None:
            c.update(v.components)
        return c

    def get_contravariant_component(self, v: Vector, k: int) -> Any:
        return v.components[self._covariant_basis_map[k]]

    def get_covariant_component(self, v: Vector, k: int) -> Any:
        g = self.get_covariant_metric_tensor()
        a = self.components(v)
        return (g @ np.array(list(a.values())))[k]

    def get_det_g(self, covariant: bool = True) -> Expr:
        """Return determinant of covariant metric tensor"""
        if self._det_g[covariant] is not None:
            return self._det_g[covariant]
        if covariant:
            g = sp.Matrix(self.get_covariant_metric_tensor()).det()
        else:
            g = sp.Matrix(self.get_contravariant_metric_tensor()).det()
        g = g.factor()
        g = self.simplify(self.refine(g))
        self._det_g[covariant] = g
        return g

    def get_sqrt_det_g(self, covariant: bool = True) -> Expr:
        """Return square root of determinant of covariant metric tensor"""
        if self._sqrt_det_g[covariant] is not None:
            return self._sqrt_det_g[covariant]
        g = self.get_det_g(covariant)
        sg = self.simplify(self.refine(sp.sqrt(g)))
        if isinstance(sg, numbers.Number):
            if isinstance(sg, numbers.Real):
                sg = float(sg)
            elif isinstance(sg, numbers.Complex):
                sg = complex(sg)
            else:
                raise RuntimeError

        self._sqrt_det_g[covariant] = sg
        return sg

    def get_scaling_factors(self) -> np.ndarray[Any, np.dtype[object]]:
        """Return scaling factors"""
        if self._hi is not None:
            return self._hi
        hi = np.zeros_like(self.psi)

        for i, s in enumerate(np.sum(self.b**2, axis=1)):
            hi[i] = sp.sqrt(self.refine(self.simplify(s)))
            hi[i] = self.refine(hi[i])

        self._hi = hi
        return hi

    def get_cartesian_basis(
        self, as_coordsys3d: bool = False
    ) -> np.ndarray[Any, np.dtype[object]]:
        """Return Cartesian basis vectors"""
        if as_coordsys3d:
            return np.array(self.base_vectors())

        return np.eye(len(self.rv), dtype=object)

    def get_normal_basis(
        self, as_coordsys3d: bool = False
    ) -> np.ndarray[Any, np.dtype[object]]:
        if self._e is not None:
            if as_coordsys3d:
                return self._e @ self._parent.base_vectors()[: self._e.shape[1]]
            return self._e

        b = self.b
        e = np.zeros_like(b)
        for i, bi in enumerate(b):
            l = sp.sqrt(self.simplify(np.dot(bi, bi)))
            l = self.refine(l)
            e[i] = bi / l
        self._e = e
        if as_coordsys3d:
            return e @ self._parent.base_vectors()[: e.shape[1]]
        return e

    def get_covariant_basis(
        self, as_coordsys3d: bool = False
    ) -> np.ndarray[Any, np.dtype[object]]:
        """Return covariant basisvectors"""
        if self._b is not None:
            if as_coordsys3d:
                return self._b @ self._parent.base_vectors()[: self._b.shape[1]]
            return self._b

        b = np.zeros((len(self.psi), len(self.rv)), dtype=object)
        for i, ti in enumerate(self.psi):
            for j, rj in enumerate(self.rv):
                b[i, j] = rj.diff(ti, 1)
                b[i, j] = self.refine(self.simplify(b[i, j]))

        self._b = b
        if as_coordsys3d:
            return b @ self._parent.base_vectors()[: b.shape[1]]
        return b

    def get_contravariant_basis(
        self, as_coordsys3d: bool = False
    ) -> np.ndarray[Any, np.dtype[object]]:
        """Return contravariant basisvectors"""
        if self._bt is not None:
            if as_coordsys3d:
                return self._bt @ self._parent.base_vectors()[: self._bt.shape[1]]
            return self._bt

        bt = np.zeros_like(self.b)
        g = self.get_contravariant_metric_tensor()
        b = self.b
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                bt[i] += g[i, j] * b[j]

        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                bt[i, j] = self.simplify(bt[i, j])
        self._bt = bt
        if as_coordsys3d:
            return bt @ self._parent.base_vectors()[: bt.shape[1]]
        return bt

    def get_normal_metric_tensor(self) -> np.ndarray[Any, np.dtype[object]]:
        """Return normal metric tensor"""
        if self._gn is not None:
            return self._gn
        gn = np.zeros((len(self.psi), len(self.psi)), dtype=object)
        e = self.e
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                gn[i, j] = self.refine(self.simplify(np.dot(e[i], e[j]).expand()))

        self._gn = gn
        return gn

    def get_covariant_metric_tensor(self) -> np.ndarray[Any, np.dtype[object]]:
        """Return covariant metric tensor"""
        if self._g is not None:
            return self._g
        g = np.zeros((len(self.psi), len(self.psi)), dtype=object)
        b = self.b
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                g[i, j] = self.refine(self.simplify(np.dot(b[i], b[j]).expand()))

        self._g = g
        return g

    def get_contravariant_metric_tensor(self) -> np.ndarray[Any, np.dtype[object]]:
        """Return contravariant metric tensor"""
        if self._gt is not None:
            return self._gt
        g = self.get_covariant_metric_tensor()
        gt = sp.Matrix(g).inv()
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                gt[i, j] = self.simplify(gt[i, j])
        gt = np.array(gt)
        self._gt = gt
        return gt

    def get_christoffel_second(self) -> np.ndarray[Any, np.dtype[object]]:
        """Return Christoffel symbol of second kind"""
        if self._ct is not None:
            return self._ct
        b = self.get_covariant_basis()
        bt = self.get_contravariant_basis()
        ct = np.zeros((len(self.psi),) * 3, object)
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                for k in range(len(self.psi)):
                    ct[k, i, j] = self.simplify(
                        np.dot(
                            np.array([bij.diff(self.psi[j], 1) for bij in b[i]]), bt[k]
                        )
                    )
        self._ct = ct
        return ct

    def get_metric_tensor(self, kind="normal") -> np.ndarray[Any, np.dtype[object]]:
        if kind == "covariant":
            gij = self.get_covariant_metric_tensor()
        elif kind == "contravariant":
            gij = self.get_contravariant_metric_tensor()
        elif kind == "normal":
            gij = self.get_normal_metric_tensor()
        else:
            raise NotImplementedError
        return gij

    def get_basis(self, kind: str = "normal") -> np.ndarray[Any, np.dtype[object]]:
        if kind == "covariant":
            return self.get_covariant_basis()
        assert kind == "normal"
        return self.get_normal_basis()

    def simplify(self, expr: Expr) -> Expr:
        return self.expr_psi_to_base_scalar(
            sp.simplify(self.expr_base_scalar_to_psi(expr), measure=self._measure)
        )

    def refine(self, sc: Expr) -> Expr:
        sc = self.expr_base_scalar_to_psi(sc)
        sc = sp.refine(sc, self._assumptions)
        for a, b in self._replace:
            sc = sc.replace(a, b)
        return self.expr_psi_to_base_scalar(sc)

    def subs(self, s0: Expr, s1: Expr) -> Expr:
        b = self.get_covariant_basis()
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                b[i, j] = b[i, j].subs(s0, s1)

        g = self.get_covariant_metric_tensor()
        gt = self.get_contravariant_metric_tensor()
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                g[i, j] = g[i, j].subs(s0, s1)
                gt[i, j] = gt[i, j].subs(s0, s1)

        sg = self.get_sqrt_det_g().subs(s0, s1)
        self._sqrt_det_g[True] = sg

        hi = self.get_scaling_factors()
        for i in range(len(hi)):
            hi[i] = hi[i].subs(s0, s1)

        self._psi = tuple([p.subs(s0, s1) for p in self._psi])
        self._rv = tuple([r.subs(s0, s1) for r in self._rv])


class SubCoordSys:
    def __init__(self, system: CoordSys, index: int = 0) -> None:
        assert system.dims > 1
        self._base_scalars = (system._base_scalars[index],)
        self._base_vectors = (system._base_vectors[index],)
        self._psi = (system._psi[index],)
        self._cartesian_xyz = [system._cartesian_xyz[index]]
        self._variable_names = [system._variable_names[index]]
        self._position_vector = system._position_vector[index]
        self._parent = system
        self.sg = 1
        for k in self._cartesian_xyz:
            setattr(self, k.name, k)

    def __iter__(self) -> Iterable[BaseVector]:
        return iter(self.base_vectors())

    def base_vectors(self) -> tuple[BaseVector]:
        return self._base_vectors

    def base_scalars(self) -> tuple[BaseScalar]:
        return self._base_scalars

    @property
    def rv(self) -> tuple[Expr]:
        return self._position_vector

    @property
    def psi(self) -> tuple[Symbol]:
        return self._base_scalars


def get_CoordSys(
    name: str,
    transformation: Lambda,
    vector_names: list[str] = None,
    assumptions: AssumptionKeys = True,
    replace: list[tuple] | tuple[tuple] = (),
    measure: Function = sp.count_ops,
    cartesian_name: str = "R",
) -> CoordSys:
    """Return a curvilinear coordinate system.

    Parameters
    ----------
    name : str
        The name of the new CoordSys instance.

    transformation : Lambda
        Transformation defined by the position vector

    vector_names : iterable(optional)
        Iterables of 3 strings each, with custom names for base
        vectors and base scalars of the new system respectively.
        Used for simple str printing.

    assumptions : Sympy assumptions
        For example: sp.Q.positive & sp.Q.real

    replace : iterable(optional)
        Iterable of 2-tuples, replacing the first item with the
        second. For helping Sympy with simplifications.

    measure : Python function to replace Sympy's count_ops.
        For example, to discourage the use of powers in an
        expression use::

        def discourage_powers(expr):
            POW = sp.Symbol('POW')
            count = sp.count_ops(expr, visual=True)
            count = count.replace(POW, 100)
            count = count.replace(sp.Symbol, type(sp.S.One))
            return count

    """
    from jaxfun.arguments import CartCoordSys, x, y, z

    return CoordSys(
        name,
        transformation,
        vector_names=vector_names,
        parent=CartCoordSys(
            cartesian_name,
            {1: (x,), 2: (x, y), 3: (x, y, z)}[len(transformation.args[1])],
        ),
        assumptions=assumptions,
        replace=replace,
        measure=measure,
    )


sp.vector.vector.BaseVector = BaseVector
sp.vector.vector.BaseScalar = BaseScalar
# sp.vector.Vector._base_func = BaseVector
sp.vector.vector.VectorMul._base_func = BaseVector
# sp.vector.vector.VectorMul._base_instance = BaseVector
# sp.vector.functions.BaseVector = BaseVector
# sp.vector.functions.BaseScalar = BaseScalar
