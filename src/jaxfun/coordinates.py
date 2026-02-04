"""
Curvilinear coordinate systems.

Much of the theory and notation is taken from:

[1] Kelly, P. A. Mechanics Lecture Notes: Foundations of Continuum Mechanics.
    http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/index.html
"""

from __future__ import annotations

import numbers
from collections import UserDict
from collections.abc import Callable, Iterable, Sequence
from itertools import product
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, Self, TypeGuard, cast, overload

import numpy as np
import sympy as sp
from sympy.assumptions.ask import AssumptionKeys
from sympy.core import AtomicExpr, Expr, Lambda, Symbol, Tuple
from sympy.core.assumptions import StdFactKB
from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector.dyadic import Dyadic, DyadicAdd, DyadicMul, DyadicZero
from sympy.vector.vector import Vector, VectorAdd, VectorMul, VectorZero

from jaxfun.typing import DyadicLike, TensorLike, VectorLike


def _is_tensorlike(obj: Basic) -> TypeGuard[TensorLike]:
    return isinstance(
        obj,
        BaseVector
        | Vector
        | VectorAdd
        | VectorMul
        | VectorZero
        | BaseDyadic
        | Dyadic
        | DyadicAdd
        | DyadicMul
        | DyadicZero,
    )


def _is_vectorlike(obj: Basic) -> TypeGuard[VectorLike]:
    return isinstance(
        obj,
        BaseVector | Vector | VectorAdd | VectorMul | VectorZero,
    )


def _is_dyadiclike(obj: Basic) -> TypeGuard[DyadicLike]:
    return isinstance(
        obj,
        BaseDyadic | Dyadic | DyadicAdd | DyadicMul | DyadicZero,
    )


def _is_expr(obj: Basic) -> TypeGuard[Expr]:
    return isinstance(obj, Expr)


tensor_product_symbol = "\u2297"

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
t, x, y, z = sp.symbols("t,x,y,z", real=True)
t: sp.Symbol
x: sp.Symbol
y: sp.Symbol
z: sp.Symbol


def CartCoordSys(
    name: str | int, s: sp.Symbol | tuple[sp.Symbol, ...] | sp.Tuple
) -> CoordSys:
    return CoordSys(str(name), sp.Lambda(s, s))


class defaultdict(UserDict):
    def __missing__(self, key) -> str:
        return key


latex_symbols = defaultdict(latex_sym_dict)


def get_system(a: sp.Expr) -> CoordSys:
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, CoordSys):
            return p
    raise RuntimeError("CoordSys not found")


class BaseTime(Symbol):
    """Symbol representing (physical) time for a coordinate system.

    This is a lightweight wrapper around a SymPy Symbol named ``t`` that
    participates in differentiation and hashing with a system-specific id.

    Attributes:
        _id: Tuple used for hashing / equality (system dimensional index).
        is_commutative: Always True for scalar symbols.
        is_symbol: Marker for SymPy.
        is_Symbol: Marker for SymPy.
        is_real: Time assumed real-valued.
    """

    def __new__(cls, sys: CoordSys) -> Self:
        index: int = _sympify(sys.dims)
        obj = super().__new__(cls, "t")
        # The _id is used for equating purposes, and for hashing
        obj._id = (index,)
        return obj

    _id: tuple[int, ...]

    is_commutative = True
    is_symbol = True
    is_Symbol = True
    is_real = True

    @property
    def free_symbols(self) -> set:
        return {self}

    _diff_wrt = True

    def _eval_derivative(self, s: Symbol) -> sp.Number:
        if self == s:
            return sp.S.One
        return sp.S.Zero

    precedence: int = PRECEDENCE["Atom"]

    def doit(self, **hints: dict) -> BaseTime:
        return self


class BaseScalar(AtomicExpr):
    """Scalar coordinate symbol belonging to a coordinate system.

    Represents contravariant curvilinear coordinates q^j. For Cartesian systems,
    these BaseScalars are the regular spatial coordinates.

    Provides custom pretty/LaTeX printing and differentiation behavior.

    Args:
        index: Coordinate index (0, 1 or 2).
        system: The parent coordinate system.
        pretty_str: Optional pretty (ASCII) representation; defaults to x{index}.
        latex_str: Optional LaTeX representation; defaults to ``x_{index}``.

    Raises:
        ValueError: If index is not in {0, 1, 2}.

    Attributes:
        _id: Tuple (index, system) for hashing / equality.
        _name: String name used for standard printing.
        _pretty_form: Stored pretty string.
        _latex_form: Stored LaTeX string.
        _system: Reference back to the CoordSys.
    """

    _id: tuple[int, CoordSys]
    _system: CoordSys
    _name: str
    _pretty_form: str
    _latex_form: str

    def __new__(
        cls,
        index: int,
        system: CoordSys,
        pretty_str: str | None = None,
        latex_str: str | None = None,
    ) -> Self:
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

    @property
    def free_symbols(self) -> set[Basic]:
        return {self}

    _diff_wrt = True

    def _eval_derivative(self, s: Symbol) -> sp.Number:
        if self == s:
            return sp.S.One
        return sp.S.Zero

    def _latex(self, printer: Any = None) -> str:
        return self._latex_form

    def _pretty(self, printer: Any = None) -> prettyForm:
        return prettyForm(self._pretty_form)

    precedence = PRECEDENCE["Atom"]

    @property
    def system(self) -> CoordSys:
        return self._system

    def _sympystr(self, printer: Any) -> str:
        return self._name

    def _jaxcode(self, printer: Any) -> str:
        return self._name

    def doit(self, **hints: Any) -> BaseScalar:
        return self

    def to_symbol(self) -> Symbol:
        return self.system._map_base_scalar_to_symbol[self]


class BaseVector(Vector, AtomicExpr):
    """Covariant base vector of a coordinate system

        b_j = ∂r/∂q^j

    where r is the position vector and q^j the curvilinear coordinates.
    For Cartesian systems, these are the regular constant unit vectors.

    Wraps a SymPy Vector basis element with additional metadata and pretty/LaTeX
    formats.

    Args:
        index: Basis vector index (0, 1 or 2).
        system: The parent coordinate system.
        pretty_str: Optional pretty (ASCII) representation.
        latex_str: Optional LaTeX representation.

    Raises:
        ValueError: If index not in {0, 1, 2}.

    Attributes:
        _components: Mapping of this base vector to its coefficient (always 1).
        _measure_number: Always 1 for a base vector.
        _name: Fully qualified name (system.name + '.' + base label).
        _system: Reference to the parent system.
        _id: (index, system) tuple for hashing.
        _assumptions: SymPy assumptions (commutative=True).
    """

    _id: tuple[int, CoordSys]
    _system: CoordSys
    _sys: CoordSys
    _name: str
    _pretty_form: str
    _latex_form: str
    args: tuple[Literal[0, 1, 2]]

    def __new__(
        cls,
        index: int,
        system: CoordSys,
        pretty_str: str | None = None,
        latex_str: str | None = None,
    ) -> Self:
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
    def free_symbols(self) -> set[Basic]:
        return {self}

    def to_cartesian(self) -> Vector | Dyadic:
        return self._system.to_cartesian(self)


class BaseDyadic(Dyadic, AtomicExpr):
    """Dyadic (tensor product) of two base vectors.

    Represents a rank-2 basis tensor constructed from two covariant base vectors

        b_i ⊗ b_j

    Args:
        vector1: First base vector.
        vector2: Second base vector.

    Raises:
        TypeError: If either operand is not a base vector (or zero).
    """

    _sys: CoordSys
    args: tuple[BaseVector, BaseVector]

    def __new__(cls, vector1: BaseVector, vector2: BaseVector) -> Self:
        # Verify arguments
        if not isinstance(vector1, BaseVector) or not isinstance(vector2, BaseVector):
            raise TypeError("BaseDyadic cannot be composed of non-base vectors")
        # Initialize instance
        obj = super().__new__(cls, vector1, vector2)
        obj._base_instance = obj
        obj._measure_number = 1
        obj._components = {obj: sp.S.One}
        obj._sys = vector1._sys
        obj._pretty_form = (
            "("
            + vector1._pretty_form
            + tensor_product_symbol
            + vector2._pretty_form
            + ")"
        )
        obj._latex_form = (
            r"\left("
            + vector1._latex_form
            + tensor_product_symbol
            + vector2._latex_form
            + r"\right)"
        )

        return obj

    def _sympystr(self, printer) -> str:
        arg0 = printer._print(self.args[0])
        arg1 = printer._print(self.args[1])
        return f"({arg0}{tensor_product_symbol}{arg1})"

    def _sympyrepr(self, printer) -> str:
        arg0 = printer._print(self.args[0])
        arg1 = printer._print(self.args[1])
        return f"BaseDyadic({arg0}, {arg1})"

    def to_cartesian(self):
        cart_arg0 = self._sys.to_cartesian(self.args[0])
        cart_arg1 = self._sys.to_cartesian(self.args[1])
        return cart_arg0 | cart_arg1


class CoordSys(Basic):
    """Curvilinear or Cartesian coordinate system.

    Encapsulates:
      * Base scalars (coordinates) and base vectors
      * Transformation to a parent (Cartesian) system
      * Metric tensors and derived geometric quantities
      * Helper routines for simplification and expression refinement

    Typical usage:
        R = CoordSys('R', Lambda((x, y), (x, y)))
        # polar example
        r, th = sp.symbols('r theta', positive=True, real=True)
        polar = CoordSys('P', Lambda((r, th), (r*sp.cos(th), r*sp.sin(th))))

    Attributes:
        _name: System name.
        _base_scalars: Tuple of BaseScalar objects (psi / computational coordinates).
        _base_vectors: Tuple of BaseVector objects (covariant basis).
        _base_dyadics: Tuple of BaseDyadic objects.
        _position_vector: Position vector expressed in parent Cartesian coordinates.
        _is_cartesian: True if transformation is identity.
        _parent: Parent coordinate system (Cartesian root) or None.
        _psi: Tuple of underlying symbolic parameters.
        _map_base_scalar_to_symbol: Mapping BaseScalar -> underlying Symbol.
        _map_symbol_to_base_scalar: Inverse of the above.
        _measure: Complexity metric used during simplification.
    """

    _name: str
    _vector_names: tuple[str, ...]
    _variable_names: tuple[str, ...]
    _base_scalars: Tuple
    _base_vectors: Tuple
    _base_dyadics: Tuple
    _position_vector: Tuple
    _is_cartesian: bool
    _parent: CoordSys | None
    _root: CoordSys
    _psi: Tuple
    _cartesian_xyz: Tuple
    _map_base_scalar_to_symbol: dict[BaseScalar, Symbol]
    _map_symbol_to_base_scalar: dict[Symbol, BaseScalar]
    _map_xyz_to_base_scalar: dict[Symbol, BaseScalar]
    _transformation: Lambda
    _measure: Callable[..., Any]
    _assumptions: AssumptionKeys | bool
    _replace: Sequence[tuple[Any, Any]]
    _hi: np.ndarray[Any, np.dtype[np.object_]] | None
    _b: np.ndarray[Any, np.dtype[np.object_]] | None
    _bt: np.ndarray[Any, np.dtype[np.object_]] | None
    _e: np.ndarray[Any, np.dtype[np.object_]] | None
    _g: np.ndarray[Any, np.dtype[np.object_]] | None
    _gt: np.ndarray[Any, np.dtype[np.object_]] | None
    _gn: np.ndarray[Any, np.dtype[np.object_]] | None
    _ct: np.ndarray[Any, np.dtype[np.object_]] | None
    _det_g: dict[bool, Expr | None]
    _sqrt_det_g: dict[bool, Expr | None]
    _covariant_basis_map: dict[int, BaseVector]
    _covariant_basis_dyadic_map: dict[tuple[int, int], BaseDyadic]
    _covariant_basis_map_inv: dict[BaseVector, int]
    _covariant_basis_dyadic_map_inv: dict[BaseDyadic, tuple[int, int]]
    _map_base_scalar_to_index: dict[BaseScalar, int]

    # NOTE: Many coordinate and basis attributes are set dynamically during
    # construction (e.g. CartCoordSys adds x/y/z, get_CoordSys adds r/theta/zz and
    # corresponding basis vectors). We annotate common ones here for static type
    # checkers without changing runtime behavior.
    if TYPE_CHECKING:
        t: BaseScalar
        x: BaseScalar
        y: BaseScalar
        z: BaseScalar
        r: BaseScalar
        theta: BaseScalar
        zz: BaseScalar

        i: BaseVector
        j: BaseVector
        k: BaseVector
        b_r: BaseVector
        b_theta: BaseVector
        b_z: BaseVector
        b_zz: BaseVector

    def __new__(
        cls,
        name: str,
        transformation: Lambda | None = None,
        vector_names: list[str] | None = None,
        parent: CoordSys | None = None,
        assumptions: AssumptionKeys | bool = True,
        replace: Sequence[tuple[Any, Any]] = (),
        measure: Callable[..., Any] = sp.count_ops,
    ) -> Self:
        """Creates and initializes a coordinate system.

        Args:
            name: Name identifier.
            transformation: A SymPy Lambda mapping computational
                coordinates (psi) to Cartesian coordinates (position vector).
            vector_names: Optional custom names for the base vectors.
            parent: Parent (typically Cartesian) system. If None and the
                transformation is identity, system is Cartesian.
            assumptions: SymPy logical assumption set (e.g. sp.Q.real & sp.Q.positive).
            replace: List/tuple of (pattern, replacement) pairs to aid simplification.
            measure: Custom operation counting function for complexity-guided
                simplification.

        Returns:
            The constructed coordinate system instance.

        Raises:
            TypeError: If name is not a string.
        """

        name = str(name)

        if not isinstance(name, str):
            raise TypeError("name should be a string")

        if transformation is None:
            raise TypeError("transformation must be provided")
        psi, position_vector = transformation.args
        assert isinstance(psi, Tuple)
        assert isinstance(position_vector, Tuple)
        variable_names = [i.name for i in psi]
        is_cartesian = False
        if len(position_vector) == len(psi):  # noqa: SIM102
            if np.all(np.array(position_vector) == np.array(psi)):
                is_cartesian = True
        if vector_names is None:
            if is_cartesian:
                vector_names = ["i", "j", "k"]
            else:
                vector_names = [f"b_{s}" for s in variable_names]

        obj: Self = super().__new__(cls, Str(name), transformation)
        obj._name = name

        vector_names = list(vector_names)
        if is_cartesian:
            latex_vects = [r"\mathbf{{%s}}" % (x,) for x in vector_names]  # noqa: UP031
        else:
            latex_vects = [
                r"\mathbf{b_{%s}}" % (latex_symbols[x],)  # noqa: UP031
                for x in variable_names
            ]
        pretty_vects = vector_names

        obj._vector_names = tuple(vector_names)

        # Create covariant basis vectors in case of curvilinear
        v: list[BaseVector] = []
        for i in range(len(psi)):
            v.append(BaseVector(i, obj, pretty_vects[i], latex_vects[i]))

        obj._base_vectors = Tuple(*v)

        variable_names = list(variable_names)
        latex_scalars = [latex_symbols[x] for x in variable_names]
        pretty_scalars = variable_names

        obj._variable_names = tuple(variable_names)

        base_scalars: list[BaseScalar] = []
        for i in range(len(psi)):
            base_scalars.append(BaseScalar(i, obj, pretty_scalars[i], latex_scalars[i]))
        obj._psi = psi
        obj._cartesian_xyz = (
            Tuple(*base_scalars) if parent is None else parent._cartesian_xyz
        )

        obj._map_base_scalar_to_symbol = {
            k: v for k, v in zip(base_scalars, obj._psi, strict=False)
        }
        obj._map_symbol_to_base_scalar = {
            k: v for k, v in zip(obj._psi, base_scalars, strict=False)
        }
        obj._map_base_scalar_to_index = {k: i for i, k in enumerate(base_scalars)}

        position_vector = position_vector.xreplace(obj._map_symbol_to_base_scalar)
        assert isinstance(position_vector, Tuple)
        obj._map_xyz_to_base_scalar = {
            k: v for k, v in zip(obj._cartesian_xyz, position_vector, strict=False)
        }

        # Add doit to Cartesian coordinates, such that x, y, z are evaluated in
        # computational space as x(psi), y(psi), z(psi)
        if not is_cartesian:
            for s in obj._cartesian_xyz:
                s.doit = MethodType(
                    lambda self, **hints: obj._map_xyz_to_base_scalar[self], s
                )

        obj._base_scalars = Tuple(*base_scalars)
        obj._base_dyadics = Tuple(
            *[d[0] | d[1] for d in product(*(obj._base_vectors,) * 2)]
        )
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
            k: v
            for k, v in zip(
                range(len(obj._base_vectors)), obj._base_vectors, strict=False
            )
        }
        obj._covariant_basis_map_inv = {
            v: k for k, v in obj._covariant_basis_map.items()
        }
        obj._covariant_basis_dyadic_map = {
            (i // len(obj._base_vectors), i % len(obj._base_vectors)): v
            for i, v in enumerate(obj._base_dyadics)
        }
        obj._covariant_basis_dyadic_map_inv = {
            v: k for k, v in obj._covariant_basis_dyadic_map.items()
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

    def base_vectors(self) -> Tuple:
        return self._base_vectors

    def base_scalars(self) -> Tuple:
        return self._base_scalars

    def base_dyadics(self) -> Tuple:
        return self._base_dyadics

    def base_time(self) -> BaseTime:
        return BaseTime(self)

    @property
    def rv(self) -> Tuple:
        return self._position_vector

    def get_cartesian_basis_vectors(self) -> Tuple:
        return self._parent.base_vectors() if self._parent else self.base_vectors()

    @overload
    def position_vector(self, as_Vector: Literal[False] = False) -> Tuple: ...
    @overload
    def position_vector(self, as_Vector: Literal[True]) -> Vector: ...
    def position_vector(self, as_Vector: bool = False) -> Tuple | Vector:
        r_out = self.refine_replace(self.rv)
        assert isinstance(r_out, Tuple)
        r = r_out
        base_vectors = self.get_cartesian_basis_vectors()
        if as_Vector:
            out: VectorLike = np.array(r) @ base_vectors
            return out
        return r

    @property
    def psi(self) -> Tuple:
        return self._base_scalars

    @property
    def b(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        return self.get_covariant_basis()

    @property
    def bt(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        return self.get_contravariant_basis()

    @property
    def e(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        return self.get_normal_basis()

    @property
    def hi(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        return self.get_scaling_factors()

    @property
    def sg(self) -> Expr:
        if self.is_cartesian:
            return sp.Integer(1)
        return self.get_sqrt_det_g(True)

    def get_normal_basis(
        self, as_Vector: bool = False
    ) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns orthonormal ("normal") basis vectors.

        For orthogonal coordinate systems, the normal basis vectors are
        e_i = b_i / h_i where b_i are covariant basis vectors and h_i are
        the scaling factors.

        Args:
            as_Vector: If True, returns SymPy Vectors; otherwise returns a
                raw component array.

        Returns:
            Numpy object array representing the normal basis.
        """
        if self._e is not None:
            if as_Vector:
                return (
                    self._e
                    @ np.array(self.get_cartesian_basis_vectors())[: self._e.shape[1]]
                )
            return self._e

        if not self.is_orthogonal:
            raise RuntimeError("Normal basis only defined for orthogonal systems")

        b = self.get_covariant_basis()
        hi = self.get_scaling_factors()
        e = np.array(b / hi[:, None])
        self._e = e
        if as_Vector:
            return e @ np.array(self.get_cartesian_basis_vectors())[: e.shape[1]]
        return e

    @property
    def is_orthogonal(self) -> bool:
        return sp.Matrix(self.get_covariant_metric_tensor()).is_diagonal()

    @property
    def is_cartesian(self) -> bool:
        return self._is_cartesian

    @overload
    def to_cartesian(self, v: VectorLike) -> VectorLike: ...
    @overload
    def to_cartesian(self, v: DyadicLike) -> DyadicLike: ...
    def to_cartesian(self, v: TensorLike) -> TensorLike:
        # v either Cartesian or a vector/dyadic with covariant basis vectors
        if isinstance(v, VectorZero | DyadicZero):
            return v

        if v._sys.is_cartesian:  # type: ignore[attr-defined]
            return v

        cart_map = {
            k: v
            for k, v in zip(
                self.base_vectors(), self.get_covariant_basis(True), strict=False
            )
        }

        if not v.is_Vector:
            cart_map.update(
                {
                    k: (cart_map[k.args[0]] | cart_map[k.args[1]])
                    for k in self.base_dyadics()
                }
            )

        return v.xreplace(cart_map)

    # @overload
    # def from_cartesian(self, v: BaseVector) -> BaseVector: ...
    # @overload
    # def from_cartesian(self, v: BaseDyadic) -> BaseDyadic: ...
    def from_cartesian(self, v: TensorLike) -> TensorLike:
        from jaxfun.operators import express
        from jaxfun.typing import cast_bd, cast_bv

        if self.is_cartesian:
            return v

        v = v.doit()
        bt = cast(tuple[Vector, ...], self.get_contravariant_basis(True))
        bv = cast_bv(self.base_vectors())
        if isinstance(v, Vector):
            terms: list[Vector] = []
            for i in range(len(bv)):
                terms.append(self.simplify(v & bt[i]) * bv[i])
            expr = VectorAdd(*terms)
        else:
            bd = cast_bd(self.base_dyadics())
            terms: list[Dyadic] = []
            for i in range(len(bt)):
                for j in range(len(bt)):
                    terms.append(self.simplify(bt[i] & v & bt[j]) * bd[i * len(bv) + j])
            expr = DyadicAdd(*terms)
        out = express(expr, self)
        if _is_tensorlike(out):
            return out
        raise TypeError("from_cartesian produced a non-tensor expression")

    def expr_base_scalar_to_psi[T: sp.Basic](self, v: T) -> T:
        return sp.sympify(v).xreplace(self._map_base_scalar_to_symbol)

    def expr_psi_to_base_scalar[T: sp.Basic](self, v: T) -> T:
        return sp.sympify(v).xreplace(self._map_symbol_to_base_scalar)

    def get_contravariant_component(
        self, v: TensorLike, k: int, j: int | None = None
    ) -> Expr:
        """Return a contravariant component of a vector or dyadic.

        For a vector expressed in this coordinate system using the covariant
        basis {b_i}, the stored components are already the contravariant
        components v^i (since v = sum_i v^i b_i). For a dyadic (2nd-order
        tensor) expressed with dyadic bases {b_i ⊗ b_j}, the stored components
        are v^{ij}.

        Args:
            v: A SymPy Vector or Dyadic expressed in this coordinate system
                (i.e. its components are keyed by the covariant basis (or
                dyadic) objects).
            k: Index of the (first) basis direction.
            j: (Optional) Second index for a Dyadic component. Must be
                provided when v is a Dyadic.

        Returns:
            The scalar SymPy expression representing v^k (if v is a Vector) or
            v^{kj} (if v is a Dyadic). Returns 0 if the component is not
            present.

        Raises:
            ValueError: If j is None while querying a Dyadic component.
        """
        # We use covariant basis vectors, so the vector v already contains the
        # contravariant components.
        if isinstance(v, VectorZero | DyadicZero):
            return sp.S.Zero
        if _is_vectorlike(v):
            vk = self._covariant_basis_map[k]
            res: Expr = v.components.get(vk, sp.S.Zero)
            return res
        if j is None:
            raise ValueError("Second index j must be provided for Dyadic components.")
        vkj = self._covariant_basis_dyadic_map[k, j]
        res: Expr = v.components.get(vkj, sp.S.Zero)
        return res

    def get_covariant_component(
        self, v: TensorLike, k: int, j: int | None = None
    ) -> Expr:
        """Return a covariant component of a vector or dyadic.

        The covariant components are obtained by contracting with the covariant
        basis vectors (or dyadics):
            v_k = v · b_k
            v_{kj} = b_k · v · b_j

        Args:
            v: A SymPy Vector or Dyadic expressed in this coordinate system.
            k: Index of the (first) basis direction.
            j: (Optional) Second index for a Dyadic component. Must be
                provided when v is a Dyadic.

        Returns:
            The scalar SymPy expression for the covariant component v_k (vector)
            or v_{kj} (dyadic).

        Raises:
            ValueError: If j is None when requesting a Dyadic component.
        """
        from jaxfun.typing import cast_bv

        b = cast_bv(self.base_vectors())
        if _is_vectorlike(v):
            return v & b[k]
        if j is None:
            raise ValueError("Second index j must be provided for Dyadic components.")
        return b[k] & v & b[j]

    def get_det_g(self, covariant: bool = True) -> Expr:
        """Returns determinant of the metric tensor.

        Args:
            covariant: If True, returns det(g_ij); otherwise det(g^ij).

        Returns:
            SymPy expression representing the determinant.
        """
        from jaxfun.operators import express

        cached = self._det_g[covariant]
        if cached is not None:
            return cached
        if covariant:
            g: sp.Basic = sp.Matrix(self.get_covariant_metric_tensor()).det()
        else:
            g: sp.Basic = sp.Matrix(self.get_contravariant_metric_tensor()).det()
        g = sp.factor(self.replace(g))
        g_out = express(self.refine(self.simplify(g)), self)
        if not _is_expr(g_out):
            raise TypeError("get_det_g produced a non-scalar expression")
        g = g_out
        self._det_g[covariant] = g
        return g

    def get_sqrt_det_g(self, covariant: bool = True) -> Expr:
        """Returns square root of the metric determinant.

        Args:
            covariant: If True, uses det(g_ij); else uses det(g^ij).

        Returns:
            SymPy expression (possibly simplified) for sqrt(det(g)).
        """
        from jaxfun.operators import express

        cached = self._sqrt_det_g[covariant]
        if cached is not None:
            return cached
        g = self.get_det_g(covariant)
        # sg = self.refine(self.simplify(sp.sqrt(g)))
        sg = self.refine(sp.sqrt(g))
        if isinstance(sg, numbers.Number):
            if isinstance(sg, numbers.Real):
                sg = float(sg)
            elif isinstance(sg, numbers.Complex):
                sg = complex(sg)
            else:
                raise RuntimeError
        sg_out = express(sp.sympify(sg), self)
        if not _is_expr(sg_out):
            raise TypeError("get_sqrt_det_g produced a non-scalar expression")
        sg = sg_out
        self._sqrt_det_g[covariant] = sg
        return sg

    def get_scaling_factors(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns orthogonal scaling factors {h_i}.

        Raises:
            RuntimeError: If the coordinate system is not orthogonal.

        Returns:
            Numpy object array of SymPy expressions for each coordinate scale factor.
        """
        from jaxfun.operators import express

        if not self.is_orthogonal:
            raise RuntimeError("Scaling factors only defined for orthogonal systems")

        if self._hi is not None:
            return self._hi
        hi = np.zeros_like(self.psi)

        for i, s in enumerate(np.sum(self.b**2, axis=1)):
            hi[i] = sp.sqrt(self.refine(self.simplify(s)))
            hi[i] = self.refine(hi[i])

        hi = np.array(express(hi, self))
        self._hi = hi
        return hi

    def get_covariant_basis(
        self, as_Vector: bool = False
    ) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns covariant basis vectors.

        Args:
            as_Vector: If True, returns an array of SymPy Vectors (Cartesian).
                If False, returns the Jacobian matrix entries (∂x^j/∂q^i).

        Returns:
            Numpy object array representing covariant basis rows.
        """
        from jaxfun.operators import express

        if self._b is not None:
            if as_Vector:
                return (
                    self._b
                    @ np.array(self.get_cartesian_basis_vectors())[: self._b.shape[1]]
                )
            return self._b

        b = np.zeros((len(self.psi), len(self.rv)), dtype=object)
        for i, ti in enumerate(self.psi):
            for j, rj in enumerate(self.rv):
                b[i, j] = rj.diff(ti, 1)
                b[i, j] = express(self.refine_replace(self.simplify(b[i, j])), self)

        self._b = b
        if as_Vector:
            return b @ np.array(self.get_cartesian_basis_vectors())[: b.shape[1]]
        return b

    def get_contravariant_basis_vector(self, i: int) -> VectorLike:
        """Returns contravariant basis vector i.

            b^i = grad(q^i) = g^ij b_j

        where g^ij is the contravariant metric tensor and b_j the covariant
        basis vector.

        Args:
            i: Basis index.

        Returns:
            SymPy Vector representing b^i.
        """
        return self.get_contravariant_metric_tensor()[i] @ self.base_vectors()

    def get_covariant_basis_vector(self, i: int) -> BaseVector:
        """Returns covariant basis vector i.

            b_i = ∂r/∂q^i

        where r is the position vector and q^i the curvilinear coordinates.

        Args:
            i: Basis index.

        Returns:
            SymPy Vector representing b_i.
        """
        return self.base_vectors()[i]

    def get_contravariant_basis(
        self, as_Vector: bool = False
    ) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns contravariant basis vectors as Cartesian vectors.

        Args:
            as_Vector: If True, returns SymPy Vectors; otherwise raw component array.

        Returns:
            Numpy object array with contravariant basis rows.
        """
        # Note. Since we only require the transformation to Cartesian in the creation
        # of this CoordSys, we need to make use of b and gt in order to find bt.

        if self._bt is not None:
            if as_Vector:
                return (
                    self._bt
                    @ np.array(self.get_cartesian_basis_vectors())[: self._bt.shape[1]]
                )
            return self._bt

        gt = self.get_contravariant_metric_tensor()
        b = self.b
        bt = np.array(self.simplify(gt @ b))
        self._bt = bt
        if as_Vector:
            return bt @ np.array(self.get_cartesian_basis_vectors())[: bt.shape[1]]
        return bt

    def get_covariant_metric_tensor(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns covariant metric tensor g_ij.

            g_ij = b_i · b_j

        Returns:
            Numpy object array (square matrix) of SymPy expressions.
        """
        if self._g is not None:
            return self._g
        b = self.b
        g = np.array(self.refine(self.simplify(b @ b.T)))
        self._g = g
        return g

    def get_contravariant_metric_tensor(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns contravariant metric tensor g^ij (inverse of g_ij).

            g^ij = b^i · b^j

        Returns:
            Numpy object array of SymPy expressions.
        """
        if self._gt is not None:
            return self._gt
        g = self.get_covariant_metric_tensor()
        gt = sp.Matrix(g).inv()
        gt = sp.factor(self.simplify(gt))
        gt = np.array(gt)
        self._gt = gt
        return gt

    def get_christoffel_second(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns Christoffel symbols Γ^k_{ij} (second kind).

            Γ^k_{ij} = ∂b_i/∂q^j · b^k

        Returns:
            3D Numpy object array with shape (dim, dim, dim) containing SymPy
            expressions.
        """
        if self._ct is not None:
            return self._ct
        b = self.get_covariant_basis()
        bt = self.get_contravariant_basis()
        ct = np.zeros((len(self.psi),) * 3, object)
        for i in range(len(self.psi)):
            for j in range(len(self.psi)):
                db = np.array([bij.diff(self.psi[j], 1) for bij in b[i]])
                for k in range(len(self.psi)):
                    ct[k, i, j] = self.simplify(np.dot(db, bt[k]))
        self._ct = ct
        return ct

    @overload
    def simplify(self, expr: DyadicLike) -> DyadicLike: ...
    @overload
    def simplify(self, expr: VectorLike) -> VectorLike: ...
    @overload
    def simplify(self, expr: sp.Expr) -> sp.Expr: ...
    def simplify(self, expr: TensorLike | sp.Expr) -> TensorLike | sp.Expr:  # ty:ignore[invalid-method-override]
        """Simplifies an expression in this coordinate system context.

        Applies:
            1. Recursive simplification of vector / dyadic components.
            2. Substitution of base scalars <-> underlying symbols.
            3. Complexity-guided SymPy simplification using provided measure.

        Args:
            expr: Expression to simplify.

        Returns:
            Simplified SymPy expression (mapped back to BaseScalars).
        """
        if isinstance(expr, VectorAdd):
            return VectorAdd.fromiter(
                self.simplify(f) * b for b, f in expr.components.items()
            )
        elif isinstance(expr, DyadicAdd):
            return DyadicAdd.fromiter(
                self.simplify(f) * b for b, f in expr.components.items()
            )
        return self.expr_psi_to_base_scalar(
            sp.simplify(self.expr_base_scalar_to_psi(expr), measure=self._measure)
        )

    def refine[T: sp.Basic](self, sc: T) -> T:  # ty:ignore[invalid-method-override]
        """Applies SymPy refine with system assumptions.

        Args:
            sc: Expression to refine.

        Returns:
            Refined expression (with base scalars restored).
        """
        sc = self.expr_base_scalar_to_psi(sc)
        sc = sp.refine(sc, self._assumptions)
        return self.expr_psi_to_base_scalar(sc)

    def replace[T: sp.Basic](self, sc: T) -> T:  # ty:ignore[invalid-method-override]
        """Performs pattern replacements then restores base scalars.

        Args:
            sc: Expression to process.

        Returns:
            Mutated expression with replacements applied.
        """
        sc = self.expr_base_scalar_to_psi(sc)
        for a, b in self._replace:
            sc = cast(T, sc.replace(a, b))
        return self.expr_psi_to_base_scalar(sc)

    def refine_replace[T: sp.Basic](self, sc: T) -> T:
        """Runs refine followed by pattern replacements.

        Args:
            sc: Expression to process.

        Returns:
            Refined and replaced expression.
        """
        sc = self.expr_base_scalar_to_psi(sc)
        sc = sp.refine(sc, self._assumptions)
        for a, b in self._replace:
            sc = sc.replace(a, b)
        cast(T, sc)
        return self.expr_psi_to_base_scalar(sc)


class SubCoordSys:
    """One-dimensional sub-coordinate system extracted from a higher-dimensional system.

    A SubCoordSys provides a lightweight view of a single coordinate direction
    (and its associated base vector and Cartesian coordinate) from a parent
    CoordSys. It is useful when operating along one axis (e.g., for separable
    problems, 1D quadrature, or directional derivatives) without constructing
    a brand new independent coordinate system.

    The sub-system reuses:
      * The selected BaseScalar (coordinate)
      * The corresponding BaseVector
      * The associated Cartesian coordinate symbol
      * The position component along that direction

    It sets `sg = 1` (square root of metric determinant) since in the
    reduced 1D view this factor is trivial; metric/Christoffel objects
    are intentionally not recreated.

    Attributes:
        _base_scalars: Tuple containing the single selected BaseScalar.
        _base_vectors: Tuple containing the single selected BaseVector.
        _psi: Tuple with the underlying symbolic parameter.
        _cartesian_xyz: List with the parent Cartesian coordinate symbol.
        _variable_names: List with the variable name (string) of the sub–coordinate.
        _position_vector: The positional expression (component) along this axis.
        _parent: Reference to the parent CoordSys.
        sg: Square root of metric determinant (set to 1 for this 1D view).

    Example:
        >>> R = CoordSys("R", sp.Lambda((x, y), (x, y)))
        >>> subx = SubCoordSys(R, index=0)
        >>> subx.base_scalars()[0]
        x
    """

    def __init__(self, system: CoordSys, index: int = 0) -> None:
        """Initialize a one-dimensional sub-system.

        Args:
            system: The parent coordinate system.
            index: Coordinate / basis index to extract (0-based).

        Raises:
            AssertionError: If the parent system is strictly 1D (no sub-extraction
            needed).
        """
        assert system.dims > 1
        self._base_scalars = (system._base_scalars[index],)
        # print(self._base_scalars)
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
        """Iterate over base vectors (single element iterator)."""
        return iter(self.base_vectors())

    def base_vectors(self) -> tuple[BaseVector]:
        """Return the tuple with the single base vector."""
        return self._base_vectors

    def base_scalars(self) -> tuple[BaseScalar]:
        """Return the tuple with the single base scalar."""
        return self._base_scalars

    @property
    def rv(self) -> Expr:
        """Return the positional expression along this axis."""
        return self._position_vector

    @property
    def position_vector(self) -> Expr:
        """Return the positional expression along this axis."""
        return self._position_vector

    @property
    def psi(self) -> tuple[Symbol]:
        """Return the underlying symbolic parameter (tuple of length 1)."""
        return self._base_scalars


def get_CoordSys(
    name: str,
    transformation: Lambda,
    vector_names: list[str] | None = None,
    assumptions: AssumptionKeys | bool = True,
    replace: Sequence[tuple[Any, Any]] = (),
    measure: Callable[..., Any] = sp.count_ops,
    cartesian_name: str = "R",
) -> CoordSys:
    """Creates a curvilinear coordinate system with a Cartesian parent.

    Convenience wrapper that ensures a Cartesian root system is created and
    passed as parent to `CoordSys`.

    Args:
        name: Name of the new system.
        transformation: SymPy Lambda mapping (psi...) -> (x,y,z).
        vector_names: Optional custom base vector names.
        assumptions: SymPy assumptions (e.g. sp.Q.real & sp.Q.positive).
        replace: (pattern, replacement) pairs for simplification assistance.
        measure: Custom complexity function used during simplification.
        cartesian_name: Name for the automatically created Cartesian parent.

    Returns:
        A fully constructed CoordSys instance.
    """
    position_vector = transformation.args[1]
    assert isinstance(position_vector, Tuple)

    return CoordSys(
        name,
        transformation,
        vector_names=vector_names,
        parent=CartCoordSys(
            cartesian_name,
            {1: (x,), 2: (x, y), 3: (x, y, z)}[len(position_vector)],
        ),
        assumptions=assumptions,
        replace=replace,
        measure=measure,
    )


sp.vector.BaseDyadic = BaseDyadic  # ty:ignore[possibly-missing-attribute]
sp.vector.BaseVector = BaseVector  # ty:ignore[possibly-missing-attribute]
sp.vector.BaseScalar = BaseScalar  # ty:ignore[possibly-missing-attribute]
sp.vector.vector.BaseDyadic = BaseDyadic  # ty:ignore[possibly-missing-attribute]
sp.vector.vector.BaseVector = BaseVector  # ty:ignore[possibly-missing-attribute]
sp.vector.vector.BaseScalar = BaseScalar  # ty:ignore[possibly-missing-attribute]
sp.vector.dyadic.BaseVector = BaseVector  # ty:ignore[possibly-missing-attribute]
sp.vector.dyadic.BaseScalar = BaseScalar  # ty:ignore[possibly-missing-attribute]
sp.vector.dyadic.BaseDyadic = BaseDyadic  # ty:ignore[possibly-missing-attribute]
# sp.vector.Vector._base_func = BaseVector
sp.vector.vector.VectorMul._base_func = BaseVector  # ty:ignore[possibly-missing-attribute]
sp.vector.dyadic.DyadicMul._base_func = BaseDyadic  # ty:ignore[possibly-missing-attribute]
# sp.vector.vector.VectorMul._base_instance = BaseVector
# sp.vector.functions.BaseVector = BaseVector
# sp.vector.functions.BaseScalar = BaseScalar
