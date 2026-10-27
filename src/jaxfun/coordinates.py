"""
Curvilinear coordinate systems.

Much of the theory and notation is taken from:

[1] Kelly, P. A. Mechanics Lecture Notes: Foundations of Continuum Mechanics.
    http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/index.html
"""

from __future__ import annotations

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
from jaxfun.utils import reverse_dict


def _is_tensorlike(obj: Basic) -> TypeGuard[TensorLike]:
    return isinstance(
        obj,
        BaseVector
        | VectorAdd
        | VectorMul
        | VectorZero
        | BaseDyadic
        | DyadicAdd
        | DyadicMul
        | DyadicZero,
    )


def _is_vectorlike(obj: Basic) -> TypeGuard[VectorLike]:
    return isinstance(
        obj,
        BaseVector | VectorAdd | VectorMul | VectorZero,
    )


def _is_dyadiclike(obj: Basic) -> TypeGuard[DyadicLike]:
    return isinstance(
        obj,
        BaseDyadic | DyadicAdd | DyadicMul | DyadicZero,
    )


tensor_product_symbol = "\u2297"


def latex_symbol(name: str) -> str:
    r"""Return the LaTeX form of a symbol name, e.g. ``\theta`` for ``theta``."""
    return sp.latex(sp.Symbol(name))


t, x, y, z = sp.symbols("t,x,y,z", real=True)
t: sp.Symbol
x: sp.Symbol
y: sp.Symbol
z: sp.Symbol


def CartCoordSys(
    name: str | int, s: sp.Symbol | tuple[sp.Symbol, ...] | sp.Tuple
) -> CoordSys:
    return CoordSys(str(name), sp.Lambda(s, s))


def R(dim: int) -> CoordSys:
    """Return Cartesian coordinate system of dimension `dim`

    Args:
        dim: Dimension of the Cartesian space (1, 2, or 3).
    """
    if dim not in (1, 2, 3):
        raise ValueError(f"dim must be 1, 2, or 3; got {dim}")

    return CartCoordSys(f"R{dim}", (x, y, z)[:dim])


def get_system(a: sp.Expr) -> CoordSys:
    for p in sp.core.traversal.iterargs(a):
        if isinstance(p, CoordSys):
            return p
    raise RuntimeError("CoordSys not found")


class BaseTime(Symbol):
    """Symbol representing (physical) time for a coordinate system.

    A SymPy Symbol named ``t``. The time axis itself is the same one whatever
    curvilinear system the spatial dimensions use, and `Symbol` compares and
    hashes on the name, so time symbols from different systems are equal,
    interchangeable under substitution and collapse in a set. Only `_id`
    distinguishes them, and it is not a claim that the axes differ: it is the
    position time occupies among a system's variables, `base_scalars() +
    (base_time(),)`, so it is `sys.dims` -- one past the last spatial index.
    `jaxfun.pinns.loss` reads it as an axis index into a Jacobian. Drop the
    system from here once that index comes from the variable list instead.

    Attributes:
        _id: Tuple holding that position (the system's dimension count).
        _system: Reference back to the CoordSys, for `__getnewargs_ex__`.
        is_commutative: Always True for scalar symbols.
        is_symbol: Marker for SymPy.
        is_Symbol: Marker for SymPy.
        is_real: Time assumed real-valued.
    """

    def __new__(cls, sys: CoordSys) -> Self:
        index: int = _sympify(sys.dims)
        # `Symbol.__new__` is memoized on the name alone, and every time symbol
        # is named "t", so it would hand back one shared object for every
        # system -- and `_id` below, being set after construction, would then
        # hold whichever system's dimension count was set last, retroactively
        # for every holder of an earlier one. That is an observable wrong
        # answer, not merely untidy: `pinns.loss` indexes a Jacobian with it.
        # `__xnew__` is the same constructor without that cache, so each system
        # gets its own instance to annotate.
        #
        # Equality is unaffected and deliberately so: `Symbol` compares and
        # hashes on the name, so time symbols from different systems remain
        # equal, hash alike, collapse in a set and substitute for one another.
        # Only identity, and with it the ownership of `_id`, separates.
        obj = super().__xnew__(cls, "t")
        obj._id = (index,)
        obj._system = sys
        return obj

    def __getnewargs_ex__(self) -> tuple[tuple[CoordSys], dict]:
        """Return the arguments __new__ needs, for copy and pickle.

        Symbol reconstructs itself from its name, but __new__ takes the system
        the time symbol belongs to. Without this, copy.deepcopy of anything
        holding a BaseTime -- a time dependent boundary condition, say -- calls
        __new__ with the string "t". Unlike BaseScalar, the system is not among
        the SymPy args, so it has to be reported here.
        """
        return ((self._system,), {})

    _id: tuple[int, ...]
    _system: CoordSys

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
    name: str
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
        if latex_str is None:
            latex_str = f"x_{index}"

        index = _sympify(index)
        system = _sympify(system)
        obj = super().__new__(cls, index, system)
        if index not in range(0, 3):
            raise ValueError("Invalid index specified.")
        # The _id is used for equating purposes, and for hashing
        obj._id = (index, system)
        obj._name = obj.name = system._variable_names[index]
        obj._pretty_form = pretty_str
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


class MetricSymbol(sp.Function):
    """A metric quantity of a coordinate system, kept as a symbol.

    A coordinate system created with ``symbolic_metric=True`` writes its metric
    in terms of symbols instead of expanding it in the coordinates: the
    covariant components g_ij (`MetricComponent`) and their determinant g
    (`MetricDeterminant`). The symbols are functions of the coordinates. Their
    derivatives are those of the explicit expressions, and generated code
    (``lambdify``) evaluates the explicit expressions. Use
    :meth:`CoordSys.expand_metric` to replace them by the explicit expressions.

    Each coordinate system has its own subclass per symbol, applied to its base
    scalars.
    """

    _system: CoordSys
    _key: tuple[int, int] | None  # (i, j) for g_ij, None for g
    _label: str
    _latex_label: str
    is_real = True

    @property
    def system(self) -> CoordSys:
        return self._system

    @property
    def explicit(self) -> Expr:
        """Return the expression the symbol stands for, at its arguments."""
        return self.at_args(self._system._explicit_metric(self._key))

    def fdiff(self, argindex: int = 1) -> Expr:
        d = self._system._diff_explicit_metric(self._key, argindex - 1)
        return self.at_args(d)

    def at_args(self, expr: Expr) -> Expr:
        """Return `expr`, given in the base scalars, at the arguments of the symbol.

        The arguments are the base scalars unless the symbol has been
        substituted into, e.g. ``g.subs(C.r, 2 * C.r)``.
        """
        base = self._system.base_scalars()
        if self.args == tuple(base):
            return expr
        return expr.xreplace(dict(zip(base, self.args, strict=True)))

    def _sympystr(self, printer: Any) -> str:
        return self._label

    def _latex(self, printer: Any = None) -> str:
        return self._latex_label

    def _pretty(self, printer: Any = None) -> prettyForm:
        return prettyForm(self._label)

    def _jaxcode(self, printer: Any) -> str:
        return printer._print(self.explicit)


class MetricDeterminant(MetricSymbol):
    """Determinant g = det(g_ij) of the covariant metric tensor, as a symbol.

    g is positive, so powers of it combine as they do for a positive symbol
    (``sqrt(g)**2 == g``). A ``symbolic_metric`` system writes
    ``g^ij = adj(g_ij) / g`` and ``sqrt(g)`` for the volume element.
    """

    is_positive = True


class MetricComponent(MetricSymbol):
    """Component g_ij of the covariant metric tensor, as a symbol.

    The diagonal components are positive. Components that vanish identically
    are kept as zero rather than as symbols.
    """


class BaseVector(Vector, AtomicExpr):  # ty:ignore[invalid-method-override]
    """Covariant base vector of a coordinate system

        b_j = ∂r/∂q^j

    where r is the position vector and q^j the curvilinear coordinates.
    For Cartesian systems, these are the regular constant unit vectors.

    Wraps a SymPy Vector basis element with additional metadata and pretty/LaTeX
    formats. The contravariant base vectors are the subclass
    `ContravariantBaseVector`.

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
        _sys: Reference to the parent system, as on SymPy's VectorMul and
            VectorAdd, which copy it from their base vectors.
        _id: (index, system) tuple for hashing.
        _assumptions: SymPy assumptions (commutative=True).
    """

    _id: tuple[int, CoordSys]
    _base_instance: BaseVector
    _components: dict[BaseVector, Expr]
    _measure_number: Expr
    _assumptions: StdFactKB
    _sys: CoordSys
    _name: str
    _pretty_form: str
    _latex_form: str
    args: tuple[sp.Integer, CoordSys]

    # True for the contravariant subclass
    is_contravariant: bool = False
    # Name of the CoordSys attribute holding the names of this kind of vector
    _names_attr: str = "_vector_names"

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
        name = getattr(system, cls._names_attr)[index]
        # Initialize an object
        obj: Self = super().__new__(cls, sp.sympify(index), system)
        # Assign important attributes
        obj._base_instance = obj
        obj._components = {obj: sp.S.One}
        obj._measure_number = sp.S.One
        obj._name = system._name + "." + name
        obj._pretty_form = "" + pretty_str
        obj._latex_form = latex_str
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
        return self._sys

    @property
    def index(self) -> int:
        return int(self._id[0])

    def _sympystr(self, printer: Any) -> str:
        return self._name

    def _sympyrepr(self, printer: Any) -> str:
        index, system = self._id
        return f"{printer._print(system)}.{getattr(system, self._names_attr)[index]}"

    @property
    def free_symbols(self) -> set[Basic]:
        return {self}

    def to_cartesian(self) -> Vector:
        return self._sys.to_cartesian(self)


class ContravariantBaseVector(BaseVector):
    """Contravariant base vector of a coordinate system

        b^j = ∇q^j = g^jk b_k

    where q^j are the curvilinear coordinates, g^jk the contravariant metric
    tensor and b_k the covariant base vectors. The two sets are dual,
    b^i·b_j = δ^i_j, which is what makes contractions with them cheap.

    Cartesian systems have no separate contravariant base vectors, since there
    b^j = b_j.

    Args:
        index: Basis vector index (0, 1 or 2).
        system: The parent coordinate system.
        pretty_str: Optional pretty (ASCII) representation.
        latex_str: Optional LaTeX representation.
    """

    is_contravariant = True
    _names_attr = "_contravariant_vector_names"


class BaseDyadic(Dyadic, AtomicExpr):  # ty:ignore[invalid-method-override]
    """Dyadic (tensor product) of two base vectors.

    Represents a rank-2 basis tensor constructed from two base vectors, each
    either covariant or contravariant

        b_i ⊗ b_j,  b_i ⊗ b^j,  b^i ⊗ b_j  or  b^i ⊗ b^j

    Args:
        vector1: First base vector.
        vector2: Second base vector.

    Raises:
        TypeError: If either operand is not a base vector (or zero).
    """

    _sys: CoordSys
    _base_instance: BaseDyadic
    _measure_number: int
    _components: dict[BaseDyadic, Expr]
    _pretty_form: str
    _latex_form: str
    args: tuple[BaseVector, BaseVector]

    def __new__(cls, vector1: BaseVector, vector2: BaseVector) -> Self:
        v1, v2 = vector1, vector2
        # Verify arguments
        if not isinstance(v1, BaseVector) or not isinstance(v2, BaseVector):
            raise TypeError("BaseDyadic cannot be composed of non-base vectors")
        # Initialize instance
        obj: Self = super().__new__(cls, v1, v2)
        obj._base_instance = obj
        obj._measure_number = 1
        obj._components = {obj: sp.S.One}
        obj._sys = v1._sys
        tp_symb = tensor_product_symbol
        obj._pretty_form = f"({v1._pretty_form}{tp_symb}{v2._pretty_form})"
        obj._latex_form = rf"\left({v1._latex_form}{tp_symb}{v2._latex_form}\right)"

        return obj

    def _sympystr(self, printer) -> str:
        arg0 = printer._print(self.args[0])
        arg1 = printer._print(self.args[1])
        return f"({arg0}{tensor_product_symbol}{arg1})"

    def _sympyrepr(self, printer) -> str:
        arg0 = printer._print(self.args[0])
        arg1 = printer._print(self.args[1])
        return f"BaseDyadic({arg0}, {arg1})"

    def to_cartesian(self) -> Dyadic:
        return self._sys.to_cartesian(self)


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
        _contravariant_base_vectors: Tuple of ContravariantBaseVector objects
            (contravariant basis). Equal to _base_vectors for Cartesian systems.
        _base_dyadics: Tuple of BaseDyadic objects (covariant b_i ⊗ b_j).
        _position_vector: Position vector expressed in parent Cartesian coordinates.
        _is_cartesian: True if transformation is identity.
        _parent: Parent coordinate system (Cartesian root) or None.
        _psi: Tuple of underlying symbolic parameters.
        _map_base_scalar_to_symbol: Mapping BaseScalar -> underlying Symbol.
        _map_base_scalar_to_index: Mapping BaseScalar -> coordinate index.
        _map_symbol_to_base_scalar: Inverse of the above.
        _measure: Complexity metric used during simplification.
    """

    _name: str
    _vector_names: tuple[str, ...]
    _contravariant_vector_names: tuple[str, ...]
    _variable_names: tuple[str, ...]
    _base_scalars: Tuple
    _base_vectors: Tuple
    _contravariant_base_vectors: Tuple
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
    _b: np.ndarray[Any, np.dtype[np.object_]] | None
    _bt: np.ndarray[Any, np.dtype[np.object_]] | None
    _g: np.ndarray[Any, np.dtype[np.object_]] | None
    _gt: np.ndarray[Any, np.dtype[np.object_]] | None
    _ct: np.ndarray[Any, np.dtype[np.object_]] | None
    _det_g: Expr | None
    _sqrt_det_g: Expr | None
    _symbolic_metric: bool
    _g_symbol: MetricDeterminant | None
    _g_sym: np.ndarray[Any, np.dtype[np.object_]] | None
    _adj: np.ndarray[Any, np.dtype[np.object_]] | None
    _d_metric: dict[tuple[tuple[int, int] | None, int], Expr]
    _dbase: dict[tuple[BaseVector, int], VectorLike]
    _map_base_scalar_to_index: dict[BaseScalar, int]

    if TYPE_CHECKING:
        # Cartesian coordinates and base vectors, named by jaxfun
        x: BaseScalar
        y: BaseScalar
        z: BaseScalar
        i: BaseVector
        j: BaseVector
        k: BaseVector

        # Coordinates and base vectors named after the user's symbols, such as
        # C.r, C.b_r and C.bt_r, exist only at runtime.
        def __getattr__(self, name: str) -> Any: ...

    def __new__(
        cls,
        name: str,
        transformation: Lambda,
        vector_names: list[str] | None = None,
        parent: CoordSys | None = None,
        assumptions: AssumptionKeys | bool = True,
        replace: Sequence[tuple[Any, Any]] = (),
        measure: Callable[..., Any] = sp.count_ops,
        symbolic_metric: bool = False,
    ):
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
            symbolic_metric: If True, keep the covariant metric components g_ij
                and their determinant g as symbols (see `MetricSymbol`), so that
                `sg` is sqrt(g) and the contravariant metric tensor and the
                Christoffel symbols are written over g in terms of g_ij. Use
                `expand_metric` for the explicit expressions.

        Returns:
            The constructed coordinate system instance.
        """
        name = str(name)
        psi, position_vector = transformation.args
        assert isinstance(psi, Tuple)
        assert isinstance(position_vector, Tuple)

        variable_names = tuple(s.name for s in psi)
        is_cartesian = position_vector == psi
        if vector_names is None:
            vector_names = (
                ["i", "j", "k"] if is_cartesian else [f"b_{s}" for s in variable_names]
            )

        obj = cast("CoordSys", super().__new__(cls, Str(name), transformation))
        obj._name = name
        obj._variable_names = variable_names
        obj._vector_names = tuple(vector_names)

        # Covariant base vectors
        if is_cartesian:
            latex_vects = [r"\mathbf{{%s}}" % (s,) for s in vector_names]  # noqa: UP031
        else:
            latex_vects = [
                r"\mathbf{b_{%s}}" % (latex_symbol(s),)  # noqa: UP031
                for s in variable_names
            ]
        v: list[BaseVector] = [
            BaseVector(i, obj, vector_names[i], latex_vects[i]) for i in range(len(psi))
        ]
        obj._base_vectors = Tuple(*v)

        # Contravariant base vectors. In Cartesian systems b^i = b_i.
        vt: list[BaseVector] = v
        obj._contravariant_vector_names = obj._vector_names
        if not is_cartesian:
            obj._contravariant_vector_names = tuple(f"bt_{s}" for s in variable_names)
            vt = [
                ContravariantBaseVector(
                    i,
                    obj,
                    f"b^{s}",
                    r"\mathbf{b}^{%s}" % (latex_symbol(s),),  # noqa: UP031
                )
                for i, s in enumerate(variable_names)
            ]
        obj._contravariant_base_vectors = Tuple(*vt)

        base_scalars = [
            BaseScalar(i, obj, s, latex_symbol(s)) for i, s in enumerate(variable_names)
        ]
        obj._base_scalars = Tuple(*base_scalars)
        obj._psi = psi
        obj._cartesian_xyz = (
            obj._base_scalars if parent is None else parent._cartesian_xyz
        )
        bs_to_symb: dict[BaseScalar, Symbol] = dict(
            zip(base_scalars, cast(tuple[Symbol, ...], psi))
        )
        obj._map_base_scalar_to_symbol = bs_to_symb
        obj._map_symbol_to_base_scalar = reverse_dict(bs_to_symb)
        obj._map_base_scalar_to_index = {s: i for i, s in enumerate(base_scalars)}

        position_vector = position_vector.xreplace(obj._map_symbol_to_base_scalar)
        assert isinstance(position_vector, Tuple)
        obj._map_xyz_to_base_scalar = dict(zip(obj._cartesian_xyz, position_vector))

        # Add doit to Cartesian coordinates, such that x, y, z are evaluated in
        # computational space as x(psi), y(psi), z(psi)
        if not is_cartesian:
            for s in obj._cartesian_xyz:
                s.doit = MethodType(
                    lambda self, **hints: obj._map_xyz_to_base_scalar[self], s
                )

        obj._base_dyadics = Tuple(*[a | b for a, b in product(v, repeat=2)])
        obj._position_vector = position_vector
        obj._is_cartesian = is_cartesian
        obj._transformation = transformation
        obj._measure = measure
        obj._assumptions = assumptions
        obj._replace = replace
        obj._parent = parent
        obj._root = obj if parent is None else parent._root

        # Metric quantities, computed when first asked for
        obj._b = obj._bt = obj._g = obj._gt = obj._ct = None
        obj._det_g = obj._sqrt_det_g = None
        obj._g_sym = obj._adj = None
        obj._dbase = {}
        obj._d_metric = {}
        obj._symbolic_metric = symbolic_metric and not is_cartesian
        obj._g_symbol = None
        if obj._symbolic_metric:
            obj._g_symbol = cast(
                MetricDeterminant, obj._metric_symbol(MetricDeterminant, None, "g")
            )

        for s in obj._cartesian_xyz:
            setattr(obj, s.name, s)
        for i, s in enumerate(base_scalars):
            setattr(obj, variable_names[i], s)
            setattr(obj, vector_names[i], v[i])
            setattr(obj, obj._contravariant_vector_names[i], vt[i])

        return obj

    def __deepcopy__(self, memo: dict) -> Self:
        memo[id(self)] = self
        return self

    def __copy__(self) -> Self:
        return self

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

    def contravariant_base_vectors(self) -> Tuple:
        """Return the contravariant base vectors b^i.

        For a Cartesian system these are the base vectors themselves.
        """
        return self._contravariant_base_vectors

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
    def position_vector(self, as_Vector: Literal[True]) -> VectorLike: ...
    def position_vector(self, as_Vector: bool = False) -> Tuple | VectorLike:
        r = self.refine_replace(self.rv)
        if as_Vector:
            return np.array(r) @ self.get_cartesian_basis_vectors()
        return r

    @property
    def b(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        return self.get_covariant_basis()

    @property
    def bt(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        return self.get_contravariant_basis()

    @property
    def sg(self) -> Expr:
        if self.is_cartesian:
            return sp.Integer(1)
        return self.get_sqrt_det_g(True)

    @property
    def symbolic_metric(self) -> bool:
        """Whether the metric determinant is kept as a symbol g."""
        return self._symbolic_metric

    def expand_metric[T: sp.Basic](self, expr: T) -> T:
        """Return `expr` with the metric symbols made explicit.

        For a system created with ``symbolic_metric=True``, the symbols g_ij and
        every power of g are replaced by their explicit expressions in the
        coordinates; in particular sqrt(g) becomes the explicit (refined) square
        root. Other systems return `expr` unchanged.

        Args:
            expr: Expression that may contain the metric symbols.

        Returns:
            The expression without metric symbols.
        """
        if not self._symbolic_metric:
            return expr

        def is_g(e: Basic) -> bool:
            return isinstance(e, MetricDeterminant) and e._system is self

        sg = self.get_sqrt_det_g(True, explicit=True)
        # Powers first: g**e is sg**(2e), where replacing g alone would leave
        # sqrt(sg**2), which SymPy does not reduce to sg.
        out = sp.sympify(expr).replace(
            lambda e: e.is_Pow and is_g(e.base),
            lambda e: e.base.at_args(sg) ** (2 * e.exp),
        )
        rule: dict[Basic, Expr] = {
            s: s.at_args(sg) ** 2 if is_g(s) else s.explicit
            for s in out.atoms(MetricSymbol)
            if s._system is self
        }
        return cast(T, out.xreplace(rule))

    def _metric_symbol(
        self,
        cls: type[MetricSymbol],
        key: tuple[int, int] | None,
        label: str,
        latex_label: str | None = None,
        **assumptions: bool,
    ) -> MetricSymbol:
        """Return a metric symbol of this system, applied to its base scalars.

        Each symbol gets a subclass of its own that carries the system, so it
        knows its system whatever its arguments, e.g. after
        expr_base_scalar_to_psi.
        """
        attrs = {
            "_system": self,
            "_key": key,
            "_label": label,
            "_latex_label": label if latex_label is None else latex_label,
            **{f"is_{k}": v for k, v in assumptions.items()},
        }
        symbol = cast(type[MetricSymbol], type(label, (cls,), attrs))
        return symbol(*self._base_scalars)

    def _explicit_metric(self, key: tuple[int, int] | None) -> Expr:
        """Return the explicit g_ij for key (i, j), or det(g_ij) for key None."""
        if key is None:
            return self.get_det_g(True, explicit=True)
        return self.get_covariant_metric_tensor(explicit=True)[key]

    def _diff_explicit_metric(self, key: tuple[int, int] | None, i: int) -> Expr:
        """Return the derivative of `_explicit_metric(key)` with respect to q^i."""
        if (key, i) not in self._d_metric:
            d = self._explicit_metric(key).diff(self._base_scalars[i])
            self._d_metric[key, i] = self.simplify(d)
        return self._d_metric[key, i]

    def _metric_adjugate(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Return the adjugate of the covariant metric tensor, adj(g_ij) = g g^ij."""
        if self._adj is None:
            adj = sp.Matrix(self.get_covariant_metric_tensor()).adjugate()
            self._adj = np.array(adj.applyfunc(self.simplify))
        return self._adj

    def _over_det(self, num: Expr) -> Expr:
        """Return num / det(g_ij), simplified.

        With a symbolic metric only the numerator is simplified, over the symbol g.
        """
        if self._symbolic_metric:
            return self.simplify(num) / cast(MetricDeterminant, self._g_symbol)
        return self.simplify(num / self.get_det_g())

    def _as_vectors(
        self, a: np.ndarray[Any, np.dtype[np.object_]]
    ) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Return the rows of `a`, holding Cartesian components, as vectors."""
        return a @ np.array(self.get_cartesian_basis_vectors())[: a.shape[1]]

    @property
    def is_orthogonal(self) -> bool:
        return sp.Matrix(self.get_covariant_metric_tensor(True)).is_diagonal()

    @property
    def is_cartesian(self) -> bool:
        return self._is_cartesian

    def change_basis(
        self,
        v: TensorLike,
        basis: Sequence[BaseVector] | Tuple,
        basis2: Sequence[BaseVector] | Tuple | None = None,
        simplify: bool = False,
    ) -> TensorLike:
        """Return v written in another basis of this coordinate system.

        Every base vector s of v is expanded in the target basis {t} as

            s = (s · t*) t   (summed over t)

        where t* is the dual of t: b^j for b_j, b_j for b^j, and e_k for a
        Cartesian e_k. The coefficients are δ, the metric tensors g_ij and g^ij,
        or the Cartesian components of b_i and b^i. A Cartesian vector written
        in the basis of a surface embedded in 3D is projected onto the tangent
        plane.

        Args:
            v: Vector or dyadic with base vectors of this system or of its
                Cartesian system.
            basis: The target basis: ``base_vectors()``,
                ``contravariant_base_vectors()`` or
                ``get_cartesian_basis_vectors()``.
            basis2: The target basis of the second leg of a dyadic, if it differs
                from `basis`. For example the natural basis b_i ⊗ b^j.
            simplify: Whether to simplify the components of the result.

        Returns:
            The same tensor in the target basis.

        Raises:
            ValueError: If v has a base vector of another system.
        """
        from jaxfun.operators import outer

        def expand(s: BaseVector, target: Sequence[BaseVector] | Tuple) -> VectorLike:
            if s in target:
                return s
            ts = cast(Sequence[BaseVector], target)
            return VectorAdd.fromiter(self._basis_coefficient(s, t) * t for t in ts)

        if _is_dyadiclike(v):
            second = basis if basis2 is None else basis2
            out: TensorLike = DyadicAdd.fromiter(
                c * outer(expand(d.args[0], basis), expand(d.args[1], second))
                for d, c in v.components.items()
            )
        else:
            out = VectorAdd.fromiter(
                c * expand(s, basis) for s, c in v.components.items()
            )
        return self.simplify(out) if simplify else out

    def _basis_kind(self, s: BaseVector) -> str:
        """Return whether s is a Cartesian, covariant or contravariant base vector."""
        if s in self.get_cartesian_basis_vectors():
            return "cartesian"
        if s in self._base_vectors:
            return "covariant"
        if s in self._contravariant_base_vectors:
            return "contravariant"
        raise ValueError(f"{s} is not a base vector of {self} or its Cartesian system")

    def _basis_coefficient(self, s: BaseVector, t: BaseVector) -> Expr:
        """Return s · t*, the coefficient of t when s is written in the basis of t."""
        i, j = s.index, t.index
        match self._basis_kind(s), self._basis_kind(t):
            case "covariant", "contravariant":
                return self.get_covariant_metric_tensor()[i, j]
            case "contravariant", "covariant":
                return self.get_contravariant_metric_tensor()[i, j]
            case "covariant", "cartesian":
                return self.b[i, j]
            case "contravariant", "cartesian":
                return self.bt[i, j]
            case "cartesian", "covariant":
                return self.bt[j, i]
            case "cartesian", "contravariant":
                return self.b[j, i]
        return sp.S.One if i == j else sp.S.Zero

    @overload
    def to_cartesian(self, v: VectorLike) -> VectorLike: ...
    @overload
    def to_cartesian(self, v: DyadicLike) -> DyadicLike: ...
    def to_cartesian(self, v: TensorLike) -> TensorLike:
        """Return v written in the Cartesian base vectors (see `change_basis`).

        The components stay in the coordinates of this system.
        """
        return self.change_basis(v, self.get_cartesian_basis_vectors())

    def to_covariant_basis(self, v: TensorLike) -> TensorLike:
        """Return v written in the covariant base vectors (see `change_basis`).

        Every contravariant base vector becomes b^i = g^ij b_j, also in both legs
        of a dyadic.
        """
        return self.change_basis(v, self.base_vectors())

    def to_contravariant_basis(self, v: TensorLike) -> TensorLike:
        """Return v written in the contravariant base vectors (see `change_basis`).

        Every covariant base vector becomes b_i = g_ij b^j, also in both legs of
        a dyadic.
        """
        return self.change_basis(v, self.contravariant_base_vectors())

    @overload
    def from_cartesian(self, v: VectorLike) -> VectorLike: ...
    @overload
    def from_cartesian(self, v: DyadicLike) -> DyadicLike: ...
    def from_cartesian(self, v: TensorLike) -> TensorLike:
        """Return a Cartesian vector or dyadic written in the covariant basis.

        Cartesian coordinates in the components are written in the coordinates
        of this system, and the components are simplified (see `change_basis`).
        """
        from jaxfun.operators import express

        if self.is_cartesian:
            return v
        out = self.change_basis(v.doit(), self.base_vectors(), simplify=True)
        return express(out, self)

    def expr_base_scalar_to_psi[T: sp.Basic](self, v: T) -> T:
        return sp.sympify(v).xreplace(self._map_base_scalar_to_symbol)

    def expr_psi_to_base_scalar[T: sp.Basic](self, v: T) -> T:
        return sp.sympify(v).xreplace(self._map_symbol_to_base_scalar)

    def get_contravariant_component(
        self, v: TensorLike, k: int, j: int | None = None
    ) -> Expr:
        """Return a contravariant component of a vector or dyadic.

        The contravariant components are obtained by contracting with the
        contravariant basis vectors:
            v^k = v · b^k
            v^{kj} = b^k · v · b^j

        For a vector in the covariant basis, v = v^i b_i, this is the
        coefficient of b_k, since b_i · b^k = δ_i^k.

        Args:
            v: A SymPy Vector or Dyadic expressed in this coordinate system,
                in any mix of co- and contravariant base vectors.
            k: Index of the (first) basis direction.
            j: (Optional) Second index for a Dyadic component. Must be
                provided when v is a Dyadic.

        Returns:
            The scalar SymPy expression representing v^k (if v is a Vector) or
            v^{kj} (if v is a Dyadic).

        Raises:
            ValueError: If j is None while querying a Dyadic component.
        """
        from jaxfun.operators import dot

        if isinstance(v, VectorZero | DyadicZero):
            return sp.S.Zero
        bt = self.contravariant_base_vectors()
        if _is_vectorlike(v):
            return dot(v, bt[k])
        if j is None:
            raise ValueError("Second index j must be provided for Dyadic components.")
        return dot(dot(bt[k], v), bt[j])

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

    def get_det_g(self, covariant: bool = True, explicit: bool = False) -> Expr:
        """Returns determinant of the metric tensor.

        Args:
            covariant: If True, returns det(g_ij); otherwise det(g^ij).
            explicit: If True, returns the expression in the coordinates also for
                a system with ``symbolic_metric=True``, which otherwise returns
                the symbol g (or 1/g).

        Returns:
            SymPy expression representing the determinant.
        """
        if not covariant:
            return 1 / self.get_det_g(True, explicit)
        if self._symbolic_metric and not explicit:
            return cast(MetricDeterminant, self._g_symbol)
        if self._det_g is None:
            g = sp.Matrix(self.get_covariant_metric_tensor(True)).det()
            self._det_g = cast(Expr, self.refine(sp.factor(g)))
        return self._det_g

    def get_sqrt_det_g(self, covariant: bool = True, explicit: bool = False) -> Expr:
        """Returns square root of the metric determinant.

        Args:
            covariant: If True, uses det(g_ij); else uses det(g^ij).
            explicit: If True, returns the expression in the coordinates also for
                a system with ``symbolic_metric=True``, which otherwise returns
                sqrt(g) (or 1/sqrt(g)) of the symbol g.

        Returns:
            SymPy expression (possibly simplified) for sqrt(det(g)).
        """
        if not covariant:
            return 1 / self.get_sqrt_det_g(True, explicit)
        if self._symbolic_metric and not explicit:
            return sp.sqrt(cast(MetricDeterminant, self._g_symbol))
        if self._sqrt_det_g is None:
            # The square root, refined with the assumptions, may leave the bases
            # of a power differing only by sign, e.g. (a - b)/(b - a)**4.
            g = self.get_det_g(True, explicit=True)
            self._sqrt_det_g = self.simplify(self.refine(sp.sqrt(g)))
        return self._sqrt_det_g

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
        if self._b is None:
            # Simplified here, not only in g_ij: a derivative like that of
            # atan(k tan(x/2)) is smooth only after simplification.
            q, r = self.base_scalars(), self.rv
            self._b = np.array(
                [[self.simplify(rj.diff(qi)) for rj in r] for qi in q], dtype=object
            )
        return self._as_vectors(self._b) if as_Vector else self._b

    def get_contravariant_basis_vector(self, i: int) -> BaseVector:
        """Returns contravariant basis vector i.

            b^i = grad(q^i) = g^ij b_j

        where g^ij is the contravariant metric tensor and b_j the covariant
        basis vector. b^i is returned as a base vector of its own (the
        covariant b_i in a Cartesian system). Use `to_covariant_basis` to
        expand it as g^ij b_j.

        Args:
            i: Basis index.

        Returns:
            The contravariant base vector b^i.
        """
        return self._contravariant_base_vectors[i]

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
        if self._bt is None:
            # b^k = g^kl b_l = adj(g_ij)_kl b_l / g
            num = self._metric_adjugate() @ self.b
            self._bt = np.array([[self._over_det(c) for c in row] for row in num])
        return self._as_vectors(self._bt) if as_Vector else self._bt

    def get_covariant_metric_tensor(
        self, explicit: bool = False
    ) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns covariant metric tensor g_ij.

            g_ij = b_i · b_j

        Args:
            explicit: If True, returns the expressions in the coordinates also for
                a system with ``symbolic_metric=True``, which otherwise returns
                the symbols g_ij (zero where g_ij vanishes identically).

        Returns:
            Numpy object array (square matrix) of SymPy expressions.
        """
        if self._g is None:
            b = self.b
            # The replacement patterns act here, where they can help simplify
            # all that follows from the metric.
            self._g = np.array(self.simplify(self.replace(b @ b.T)))
        if not self._symbolic_metric or explicit:
            return self._g
        if self._g_sym is None:
            n = self.dims
            G = np.full((n, n), sp.S.Zero, dtype=object)
            for i in range(n):
                for j in range(i, n):
                    if self._g[i, j] != 0:
                        G[i, j] = G[j, i] = self._metric_symbol(
                            MetricComponent,
                            (i, j),
                            f"g_{i}{j}",
                            f"g_{{{i}{j}}}",
                            **({"positive": True} if i == j else {}),
                        )
            self._g_sym = G
        return self._g_sym

    def get_contravariant_metric_tensor(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns contravariant metric tensor g^ij (inverse of g_ij).

            g^ij = b^i · b^j = adj(g_ij) / det(g_ij)

        For a system with ``symbolic_metric=True`` this is written in the symbols
        g_ij and g.

        Returns:
            Numpy object array of SymPy expressions.
        """
        if self._gt is None:
            det = self.get_det_g()
            adj = self._metric_adjugate()
            self._gt = np.array([[a / det for a in row] for row in adj])
        return self._gt

    def get_christoffel_second(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        """Returns Christoffel symbols Γ^k_{ij} (second kind).

            Γ^k_{ij} = ∂b_i/∂q^j · b^k

        Returns:
            3D Numpy object array with shape (dim, dim, dim) containing SymPy
            expressions.
        """
        if self._ct is None:
            # Γ^k_ij = adj(g_ij)_kl [ij,l] / g with the first kind symbols
            # [ij,l] = ∂b_i/∂q^j · b_l
            b, adj, q = self.b, self._metric_adjugate(), self.base_scalars()
            n = range(self.dims)
            ct = np.zeros((self.dims,) * 3, dtype=object)
            for i, j in product(n, n):
                db = np.array([bij.diff(q[j]) for bij in b[i]])
                first = [np.dot(db, b[l]) for l in n]
                for k in n:
                    num = sp.Add(*(adj[k, l] * first[l] for l in n))
                    ct[k, i, j] = self._over_det(num)
            self._ct = ct
        return self._ct

    def diff_base_vector(self, bv: BaseVector, j: int) -> VectorLike:
        """Returns the derivative of a base vector with respect to q^j.

            ∂b_i/∂q^j = Γ^k_ij b_k
            ∂b^i/∂q^j = -Γ^i_jk b^k

        where Γ^k_ij are the Christoffel symbols of the second kind.

        Args:
            bv: Covariant or contravariant base vector of this system.
            j: Index of the coordinate q^j.

        Returns:
            SymPy Vector, in the same basis as bv.
        """
        if self.is_cartesian:
            return Vector.zero
        key = (bv, j)
        if key in self._dbase:
            return self._dbase[key]
        ct = self.get_christoffel_second()
        i = bv.index
        n = range(self.dims)
        if bv.is_contravariant:
            bt = self.contravariant_base_vectors()
            dbv = VectorAdd.fromiter(-ct[i, j, k] * bt[k] for k in n)
        else:
            b = self.base_vectors()
            dbv = VectorAdd.fromiter(ct[k, i, j] * b[k] for k in n)
        self._dbase[key] = dbv
        return dbv

    @overload
    def simplify(self, expr: TensorLike) -> TensorLike: ...
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
        elif isinstance(expr, VectorMul):
            return self.simplify(expr.measure_number) * expr.base_vector
        elif isinstance(expr, DyadicMul):
            return self.simplify(expr.measure_number) * expr.base_dyadic
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

    def replace[T: sp.Basic](self, sc: T, *args: Any, **kwargs: Any) -> T:
        """Performs pattern replacements then restores base scalars.

        Applies the (pattern, replacement) pairs given as ``replace`` when the
        system was created. Called with more arguments, as SymPy's
        ``Basic.replace(query, value, ...)``, it acts on the system itself.

        Args:
            sc: Expression to process.

        Returns:
            Mutated expression with replacements applied.
        """
        if args or kwargs:
            # SymPy walks into the system held by a base scalar, e.g. in
            # sp.simplify, and calls Basic.replace on it.
            return cast(T, super().replace(sc, *args, **kwargs))
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
        return self.replace(self.refine(sc))


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
        assert system.dims > 1, (
            f"cannot take sub-system {index} of coordinate system '{system}': "
            "it is already one-dimensional, so it has no axis to extract."
        )
        self._base_scalars = (system._base_scalars[index],)
        self._base_vectors = (system._base_vectors[index],)
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

    def base_time(self) -> BaseTime:
        return self._parent.base_time()


def get_CoordSys(
    name: str,
    transformation: Lambda,
    vector_names: list[str] | None = None,
    assumptions: AssumptionKeys | bool = True,
    replace: Sequence[tuple[Any, Any]] = (),
    measure: Callable[..., Any] = sp.count_ops,
    symbolic_metric: bool = False,
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
        symbolic_metric: If True, keep the metric components g_ij and their
            determinant g as symbols (see `CoordSys`).

    Returns:
        A fully constructed CoordSys instance.
    """
    position_vector = transformation.args[1]
    assert isinstance(position_vector, Tuple)

    return CoordSys(
        name,
        transformation,
        vector_names=vector_names,
        parent=R(len(position_vector)),
        assumptions=assumptions,
        replace=replace,
        measure=measure,
        symbolic_metric=symbolic_metric,
    )


def _vectoradd_sympystr(self: VectorAdd, printer: Any) -> str:
    # SymPy's version prints only the components along system.base_vectors(),
    # which would drop every contravariant component.
    items = sorted(
        self.components.items(),
        key=lambda x: (
            str(x[0].system),
            getattr(x[0], "is_contravariant", False),
            int(x[0]._id[0]),
        ),
    )
    return " + ".join(printer._print(m * k) for k, m in items)


VectorAdd._sympystr = _vectoradd_sympystr
sp.vector.BaseDyadic = BaseDyadic  # ty:ignore[possibly-missing-submodule]
sp.vector.BaseVector = BaseVector  # ty:ignore[possibly-missing-submodule]
sp.vector.BaseScalar = BaseScalar  # ty:ignore[possibly-missing-submodule]
sp.vector.vector.BaseDyadic = BaseDyadic  # ty:ignore[possibly-missing-submodule]
sp.vector.vector.BaseVector = BaseVector  # ty:ignore[possibly-missing-submodule]
sp.vector.vector.BaseScalar = BaseScalar  # ty:ignore[possibly-missing-submodule]
sp.vector.dyadic.BaseVector = BaseVector  # ty:ignore[possibly-missing-submodule]
sp.vector.dyadic.BaseScalar = BaseScalar  # ty:ignore[possibly-missing-submodule]
sp.vector.dyadic.BaseDyadic = BaseDyadic  # ty:ignore[possibly-missing-submodule]
sp.vector.vector.VectorMul._base_func = BaseVector  # ty:ignore[possibly-missing-submodule]
sp.vector.dyadic.DyadicMul._base_func = BaseDyadic  # ty:ignore[possibly-missing-submodule]
