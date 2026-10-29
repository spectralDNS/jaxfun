"""
SymPy symbols for coordinate systems: time, base scalars, metric symbols,
and co- and contravariant base vectors and dyadics.

The module also adapts SymPy's vector module to these classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, TypeGuard

import sympy as sp
from sympy.core import AtomicExpr, Expr, Symbol
from sympy.core.assumptions import StdFactKB
from sympy.core.basic import Basic
from sympy.core.sympify import _sympify
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector.dyadic import Dyadic, DyadicAdd, DyadicMul, DyadicZero
from sympy.vector.vector import Vector, VectorAdd, VectorMul, VectorZero

from jaxfun.typing import DyadicLike, TensorLike, VectorLike

if TYPE_CHECKING:
    from jaxfun.coordinates.system import CoordSys


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
