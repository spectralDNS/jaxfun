"""Symbolic argument (test/trial) and basis function utilities for Galerkin.

This module builds SymPy Function objects that encode basis functions,
test/trial functions and JAX-backed coefficient arrays over different
function space types (orthogonal, tensor product, vector, direct sums).

Key constructs:
    * TestFunction / TrialFunction: Weak form symbolic arguments.
    * ScalarFunction / VectorFunction: Physical-domain symbolic fields. No basis.
    * JAXFunction: Galerkin functions with JAX-backed coefficients.
"""

from __future__ import annotations

import itertools
import numbers
from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum, unique
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast, overload

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from sympy import Basic, Expr, Function, Tuple
from sympy.core.function import AppliedUndef
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import Vector, VectorAdd

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, CoordSys, latex_symbol
from jaxfun.galerkin.cartesianproductspace import (
    CartesianBaseSpace,
    CartesianProductSpace,
    CartesianTensorProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.operators import Dot
from jaxfun.typing import (
    ComputationalSpaceType,
    FunctionSpaceType,
    MeshKind,
    Padding,
    RankTag,
    ScalarSpaceType,
    TestSpaceType,
)

from .composite import DirectSum
from .orthogonal import OrthogonalSpace
from .tensorproductspace import DirectSumTPS, TensorProductSpace

if TYPE_CHECKING:
    from .expansion import OrthogonalExpansion

t, x, y, z = sp.symbols("t,x,y,z", real=True)

indices = "ijklmn"

# Default names of unnamed JAXFunctions, so that their printed forms differ
_jaxfunction_counter = itertools.count()

_CompositeSpaceT = TypeVar(
    "_CompositeSpaceT", bound=CartesianTensorProductSpace | CartesianProductSpace
)
_ScalarSpaceT = TypeVar("_ScalarSpaceT", bound=ScalarSpaceType)


@unique
class ArgumentTag(Enum):
    TEST = 0
    TRIAL = 1
    JAXFUNC = 2
    NONE = -1


def get_arg(p: Any) -> ArgumentTag:
    return getattr(p, "argument", ArgumentTag.NONE)


def is_jaxfunc_leaf(expr: Basic) -> bool:
    """Return True when ``expr`` is itself a bare node tagged ArgumentTag.JAXFUNC.

    Applies equally to a Galerkin JAXFunction and a PINN FlaxFunction: both are
    tagged the same way, only through get_arg / ArgumentTag.
    """
    return get_arg(expr) is ArgumentTag.JAXFUNC


def is_derivative_of_jaxfunc_leaf(expr: Basic) -> bool:
    """Return True for a Derivative whose differentiated expression is itself a bare
    JAXFUNC-tagged leaf, e.g. Derivative(u, x) but not Derivative(sin(u), x)."""
    return isinstance(expr, sp.Derivative) and is_jaxfunc_leaf(expr.expr)


def is_jaxfunc_primitive(expr: Basic) -> bool:
    """Return True for a bare JAXFUNC-tagged leaf or a direct Derivative of one."""
    return is_jaxfunc_leaf(expr) or is_derivative_of_jaxfunc_leaf(expr)


def get_BasisFunction(
    name: str,
    *,
    vector_index: int,
    local_index: int,
    global_index: int,
    rank: RankTag,
    offset: int,
    functionspace: BaseSpace,
    argument: ArgumentTag,
    arg: BaseScalar,
) -> AppliedUndef:
    """Create a symbolic basis function with enriched printing.

    Adds (monkey-patches) __str__, _pretty and _latex methods so tensor
    product / vector indices appear compactly (e.g. phi_i^{(k)}(x)).

    Args:
        name: Base name (usually space.fun_str).
        vector_index: Component index for vector-valued spaces.
        local_index: Local index within a 1D factor space.
        global_index: Global index for scalar functionspace.
        rank: Tensor rank of the overall function space (0 or 1).
        offset: Shift applied to index selection (test vs trial).
        functionspace: Parent function space object.
        argument: ArgumentTag.TEST for test, ArgumentTag.TRIAL for trial (stored on
            Function).
        arg: Underlying BaseScalar symbol (coordinate).

    Returns:
        SymPy Function instance representing one factor basis function.
    """

    # Need additional printing because of the tensor product structure of the basis
    # functions

    def __str__(cls) -> str:
        index = indices[cls.local_index + cls.offset]
        lhs = f"{cls.name}_{index}"
        rhs = f"({cls.args[0].name})"
        if cls.rank == RankTag.SCALAR:
            return lhs + rhs
        elif cls.rank == RankTag.VECTOR:
            sup = "^{(" + str(cls.vector_index) + ")}"
            return lhs + sup + rhs
        raise NotImplementedError("Rank > 1 basis functions not supported.")

    def _pretty(cls, printer: Any = None) -> prettyForm:
        return prettyForm(cls.__str__())

    def _sympystr(cls, printer: Any) -> str:
        return cls.__str__()

    def _latex(cls, printer: Any = None, exp: float | None = None) -> str:
        index = indices[cls.local_index + cls.offset]
        lhs = f"{latex_symbol(cls.name)}_{index}"
        rhs = f"({latex_symbol(cls.args[0].name)})"
        if cls.rank == RankTag.SCALAR:
            s = lhs + rhs
        elif cls.rank == RankTag.VECTOR:
            sup = "^{(" + str(cls.vector_index) + ")}"
            s = lhs + sup + rhs
        else:
            raise NotImplementedError("Rank > 1 basis functions not supported.")
        return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"

    b: AppliedUndef = sp.Function(
        name,
        vector_index=vector_index,
        local_index=local_index,
        global_index=global_index,
        rank=rank,
        offset=offset,
        functionspace=functionspace,
        argument=argument,
    )(arg)  # ty:ignore[call-non-callable]

    b.__class__.__str__ = __str__  # ty:ignore[invalid-assignment]
    b.__class__._pretty = _pretty  # ty:ignore[unresolved-attribute]
    b.__class__._sympystr = _sympystr  # ty:ignore[unresolved-attribute]
    b.__class__._latex = _latex  # ty:ignore[unresolved-attribute]
    return b


def _get_computational_function(
    arg: str, V: ComputationalSpaceType
) -> Expr | AppliedUndef:
    """Return symbolic test or trial function in computational coordinates.

    Dispatches on space type and builds a product (scalar) or VectorAdd
    (vector space) of factor basis functions.

    Args:
        arg: "test" or "trial".
        V: Function space instance.

    Returns:
        SymPy Expr representing assembled test/trial function.
    """
    base_scalars = V.system.base_scalars()
    offset = V.dims if arg == "trial" else 0
    if isinstance(V, OrthogonalSpace):
        assert base_scalars[0].is_Symbol
        return get_BasisFunction(
            V.fun_str,
            vector_index=0,
            local_index=0,
            global_index=getattr(V, "global_index", 0),
            rank=RankTag.SCALAR,
            offset=offset,
            functionspace=V,
            argument=ArgumentTag.TEST if arg == "test" else ArgumentTag.TRIAL,
            arg=base_scalars[0],
        )

    elif isinstance(V, TensorProductSpace):
        return sp.Mul.fromiter(
            get_BasisFunction(
                v.fun_str,
                vector_index=0,
                local_index=getattr(a, "_id", [0])[0],
                global_index=getattr(V, "global_index", 0),
                rank=RankTag.SCALAR,
                offset=offset,
                functionspace=v,
                argument=ArgumentTag.TEST if arg == "test" else ArgumentTag.TRIAL,
                arg=a,
            )
            for a, v in zip(base_scalars, V.basespaces, strict=False)
        )

    assert isinstance(V, VectorTensorProductSpace), (
        f"expected VectorTensorProductSpace, got {type(V).__name__}"
    )
    b = V.system.base_vectors()
    return VectorAdd.fromiter(
        sp.Mul.fromiter(
            get_BasisFunction(
                v.fun_str,
                vector_index=i,
                local_index=getattr(a, "_id", [0])[0],
                global_index=getattr(Vi, "global_index", 0),
                rank=RankTag.VECTOR,
                offset=offset,
                functionspace=v,
                argument=ArgumentTag.TEST if arg == "test" else ArgumentTag.TRIAL,
                arg=a,
            )
            for a, v in zip(base_scalars, Vi, strict=False)
        )
        * b[i]
        for i, Vi in enumerate(V)
    )


class BaseFunction(Function):
    """Abstract base class for symbolic functions with enriched printing."""

    name: str

    @abstractmethod
    def doit(self, **hints: Any) -> Any:
        pass

    def _pretty(self, printer: Any = None) -> prettyForm:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any = None) -> str:
        return self.__str__()

    @abstractmethod
    def _latex(self, printer: Any = None) -> str:
        pass

    def _applied(self, arguments: tuple[sp.Expr, ...]) -> Any:
        """Return the function applied at `arguments`, for SymPy's rebuild."""
        raise NotImplementedError


class _FunctionTag(sp.Dummy):
    """The last argument of a BaseFunction. It makes each function unique and
    links back to it, so that SymPy can rebuild it from its args."""

    owner: BaseFunction

    @property
    def free_symbols(self) -> set[sp.Basic]:
        # Not a variable: SymPy must never rename or substitute it
        return set()


def _rebuilt(*given: Any) -> BaseFunction | sp.Expr | None:
    """Return the function SymPy rebuilds from `args` in ``u.func(*u.args)``,
    or None for an ordinary constructor call.

    The constructor's positional parameters receive the args, the tag last;
    the parameters after them keep their defaults (None or False), which no
    SymPy arg can be.
    """
    args = tuple(a for a in given if a is not None and a is not False)
    if args and isinstance(args[-1], _FunctionTag):
        return args[-1].owner._applied(cast(tuple[sp.Expr, ...], args[:-1]))
    return None


# Names of SymPy's own API that `dir` keeps for an ExpansionFunction. The rest of
# what it inherits from SymPy still works but is not listed, so that tab
# completion shows the jaxfun API.
_SYMPY_NAMES = ("args", "diff", "doit", "free_symbols", "func", "has", "subs")
# Internal names, typed accessors meant for code rather than interactive use
# (`get_array` is `array` with the type narrowed by the space), and class
# attributes SymPy's metaclass sets on each class
_HIDDEN_NAMES = ("c_names", "default_assumptions", "get_array", "nargs", "own_name")


class ExpansionFunction[SpaceT: FunctionSpaceType](BaseFunction):
    r"""Base class for functions that represent expansions in a given function
    space.

    For example, a TestFunction T(x; V) represents the expansion of the test
    function in the basis of the function space
    V = :math:`\text{span}\{\phi_i\}_{i=0}^N`.

    .. math::
        T(x; V) = \sum_i T_i \phi_i(x)

    For the tensor product space W = V \otimes U, the expansion is given by

    .. math::
        T(x, y; W) = \sum_{i,j} T_{i,j} \phi_i(x) \phi_j(y)

    """

    functionspace: SpaceT
    own_name: str

    @property
    def c_names(self) -> str:
        return ", ".join([i.name for i in self.functionspace.system._cartesian_xyz])

    def __str__(self) -> str:
        V = self.functionspace
        name = (
            "\033[1m%s\033[0m" % (self.name,) if V.rank == RankTag.VECTOR else self.name  # noqa: UP031
        )
        return f"{name}({self.c_names}; {V.name})"

    def _latex(self, printer: Any = None) -> str:
        name = self.name
        if name != self.own_name:
            assert not isinstance(self.functionspace, DirectSum)
            if self.functionspace.rank == RankTag.VECTOR:
                name = r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
        return f"{name}({self.c_names}; {self.functionspace.name})"

    def _applied(self, arguments: tuple[sp.Expr, ...]) -> Any:
        """Return the function applied at `arguments`, for SymPy's rebuild.

        A test or trial function stands for basis functions, so it only exists
        at its own arguments.
        """
        if tuple(arguments) == self.args[:-1]:
            return self
        raise NotImplementedError(
            f"{self.name} stands for the basis functions of "
            f"{self.functionspace.name}; it cannot be evaluated at, or have its "
            f"coordinates replaced by, {tuple(arguments)}"
        )

    def _eval_simplify(self, **kwargs: Any) -> Self:
        return self

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        # The tag's link to its owner does not survive pickling, so the function
        # is rebuilt from its args and attributes rather than its constructor.
        return (_restore_expansion, (type(self), self.args, dict(vars(self))))

    def __dir__(self) -> list[str]:
        names = set(_SYMPY_NAMES)
        for cls in type(self).__mro__:
            if cls.__module__.startswith("jaxfun"):
                names.update(cls.__dict__)
        names.update(self.__dict__)
        names.difference_update(_HIDDEN_NAMES)
        return sorted(n for n in names if not n.startswith("_"))

    @property
    def _iterable(self) -> bool:
        # SymPy's `iterable` reads this before trying `iter`, which succeeds for
        # any class with `__getitem__`. Only fields of product spaces have items.
        return isinstance(
            self.functionspace, CartesianTensorProductSpace | CartesianProductSpace
        )

    @overload
    def __getitem__(
        self: ExpansionFunction[VectorTensorProductSpace], i: int
    ) -> Dot: ...
    @overload
    def __getitem__(self, i: int) -> Self: ...
    def __getitem__(self, i: int) -> Self | Dot:
        """Return field i of a mixed space, or component i of a vector field.

        A component of a vector field ``u`` is not a field of its own but the
        projection ``u·b^i`` onto the contravariant basis vector ``b^i``.
        """
        assert isinstance(
            self.functionspace, CartesianTensorProductSpace | CartesianProductSpace
        )
        V = self.functionspace
        if isinstance(V, VectorTensorProductSpace):
            # IndexError, not a flat index into the metric, ends unpacking
            k = range(len(V))[i]
            return Dot(self, V.system.get_contravariant_basis_vector(k))
        name = self.name
        if len(name) == len(self.functionspace):
            name = name[i]
        else:
            raise ValueError(
                f"Length of self.name ({name}) != {len(self.functionspace) = }"
            )
        # A component of a transient trial function is itself transient.
        transient = {"transient": True} if getattr(self, "transient", False) else {}
        result = self.__class__(self.functionspace[i], name=name, **transient)
        parent = self.functionspace
        if isinstance(result.functionspace, CartesianTensorProductSpace):
            result.functionspace.leaf = cast(CartesianTensorProductSpace, parent)
            for space in result.functionspace.flatten():
                space.leaf = cast(CartesianTensorProductSpace, parent)
        elif isinstance(result.functionspace, CartesianProductSpace):
            result.functionspace.leaf = cast(CartesianProductSpace, parent)
            for space in result.functionspace.flatten():
                space.leaf = cast(CartesianProductSpace, parent)
        elif isinstance(result.functionspace, TensorProductSpace):
            result.functionspace.leaf = cast(CartesianTensorProductSpace, parent)
        elif isinstance(result.functionspace, OrthogonalSpace | DirectSum):
            result.functionspace.leaf = cast(CartesianProductSpace, parent)
        return result


# NOTE
# Need a unique Symbol in order to create a new TestFunction/TrialFunction for a new
# space. Without it all TestFunctions/TrialFunctions created with the same Cartesian
# coordinates will be the same object.
class TestFunction[SpaceT: FunctionSpaceType](ExpansionFunction[SpaceT]):
    """Symbolic test function T(x; V) for weak form assembly.

    Holds a reference to its functionspace and lazily expands (via doit)
    into computational (tensor product) factor basis functions.
    """

    __test__ = False  # prevent pytest from considering this a test.
    argument: Literal[ArgumentTag.TEST]
    functionspace: TestSpaceType

    def __new__(
        cls,
        V: SpaceT,
        name: str | None = None,
        *rebuild: sp.Expr,
    ) -> Self:
        if (rebuilt := _rebuilt(V, name, *rebuild)) is not None:
            return cast(Self, rebuilt)
        coors = V.system
        tag = _FunctionTag()
        obj: Self = Function.__new__(cls, *coors._cartesian_xyz, tag)
        tag.owner = obj
        obj.argument = ArgumentTag.TEST

        _composite = (
            DirectSum
            | DirectSumTPS
            | CartesianProductSpace
            | CartesianTensorProductSpace
        )
        if isinstance(V, _composite):
            obj.functionspace = V.get_homogeneous()
        elif isinstance(V, OrthogonalSpace | TensorProductSpace):
            obj.functionspace = V
        else:
            raise ValueError("Unknown test space")

        if isinstance(V, CartesianProductSpace | CartesianTensorProductSpace):
            if name is None:
                name = "hijklmn"[: len(V)]
            elif len(name) != len(V) and not (
                len(name) == 1 and V.rank == RankTag.VECTOR
            ):
                raise ValueError(
                    f"Length of name str must equal the length of basespaces in "
                    f"CartesianProductSpace. Got {len(V)} basespaces and "
                    f"{len({name}) = }"
                )
        obj.name = name if name is not None else "TestFunction"
        obj.own_name = "TestFunction"
        return obj

    def doit(self, **hints: Any) -> Expr | AppliedUndef:
        if self.functionspace.rank == RankTag.NONE:
            raise ValueError(
                "TestFunction expansion not possible for Cartesian products that are "
                "not ranked tensors"
            )
        return _get_computational_function(
            "test", cast(ComputationalSpaceType, self.functionspace)
        )


class TrialFunction[SpaceT: FunctionSpaceType](ExpansionFunction[SpaceT]):
    """Symbolic trial function U(x; V) for weak form assembly.

    Direct sums expand to sum of component spaces. Tensor product spaces
    with direct-sum factors expand into additive combinations of products.
    """

    argument: Literal[ArgumentTag.TRIAL]
    transient: bool

    def __new__(
        cls,
        V: SpaceT,
        name: str | None = None,
        transient: bool = False,
        *rebuild: sp.Expr,
    ) -> Self:
        if (rebuilt := _rebuilt(V, name, transient, *rebuild)) is not None:
            return cast(Self, rebuilt)
        coors = V.system
        time_arg = [coors.base_time()] if transient else []
        tag = _FunctionTag()
        obj: Self = Function.__new__(cls, *coors._cartesian_xyz, *time_arg, tag)
        tag.owner = obj
        obj.functionspace = V
        if isinstance(V, CartesianProductSpace | CartesianTensorProductSpace):
            if name is None:
                name = "abcdefg"[: len(V)]
            elif len(name) != len(V) and not (
                len(name) == 1 and V.rank == RankTag.VECTOR
            ):
                raise ValueError(
                    f"Length of name str must equal the length of basespaces in "
                    f"CartesianProductSpace. Got {len(V)} basespaces and "
                    f"{len({name}) = }"
                )
        obj.name = name if name is not None else "TrialFunction"
        obj.own_name = "TrialFunction"
        obj.argument = ArgumentTag.TRIAL
        obj.transient = transient
        return obj

    @property
    def c_names(self) -> str:
        t = self.functionspace.system.base_time().name
        return super().c_names + f", {t}" * self.transient

    def doit(self, **hints: dict) -> Expr | AppliedUndef:
        fspace = self.functionspace
        comp_fun = partial(_get_computational_function, "trial")
        if isinstance(fspace, DirectSum):
            return sp.Add.fromiter(comp_fun(f) for f in fspace.basespaces)

        elif isinstance(fspace, DirectSumTPS):
            return sp.Add.fromiter(comp_fun(f) for f in fspace.tpspaces.values())

        elif isinstance(fspace, VectorTensorProductSpace):
            vector = []
            for i, (bi, tpspace) in enumerate(
                zip(
                    fspace.system.base_vectors(),
                    fspace.flatten(),
                    strict=True,
                )
            ):
                tpspaces = (
                    tpspace.tpspaces.values()
                    if isinstance(tpspace, DirectSumTPS)
                    else [tpspace]
                )

                for Vi in tpspaces:
                    vector.append(
                        sp.Mul.fromiter(
                            get_BasisFunction(
                                v.fun_str,
                                vector_index=i,
                                local_index=getattr(a, "_id", [0])[0],
                                global_index=getattr(Vi, "global_index", 0),
                                rank=RankTag.VECTOR,
                                offset=fspace.dims,
                                functionspace=v,
                                argument=ArgumentTag.TRIAL,
                                arg=a,
                            )
                            for a, v in zip(
                                fspace.system.base_scalars(), Vi, strict=True
                            )
                        )
                        * bi
                    )
            return VectorAdd.fromiter(vector)

        return comp_fun(cast(ComputationalSpaceType, fspace))


def _restore_expansion(
    cls: type[ExpansionFunction], args: tuple[sp.Basic, ...], state: dict[str, Any]
) -> ExpansionFunction:
    obj = Function.__new__(cls, *args)
    if _is_new(obj):
        obj.__dict__.update(state)
        cast(_FunctionTag, args[-1]).owner = obj
    return obj


def _is_new(obj: BaseFunction) -> bool:
    """Return True unless SymPy's cache returned `obj` already built, e.g., to
    a restore in the process that created it, which must not overwrite it."""
    return "name" not in vars(obj)


class _PlaceholderFunction(BaseFunction):
    """Physical-domain field placeholder u(x, y, ..) of a coordinate system.

    SymPy rebuilds it from its args through the `_FunctionTag` last among them,
    as for ExpansionFunction.
    """

    system: CoordSys

    def __new__(cls, name: str, system: CoordSys, *rebuild: sp.Expr) -> Self:
        if (rebuilt := _rebuilt(name, system, *rebuild)) is not None:
            return cast(Self, rebuilt)
        return cls._from_parts(name.lower(), system, _FunctionTag())

    @classmethod
    def _from_parts(
        cls,
        name: str,
        system: CoordSys,
        tag: _FunctionTag,
        arguments: Sequence[sp.Expr] | None = None,
    ) -> Self:
        """Build the placeholder from its name, system and the tag that makes
        it unique, applied at `arguments` (by default its own coordinates)."""
        own = tuple(system._cartesian_xyz)
        arguments = own if arguments is None else tuple(arguments)
        obj: Self = Function.__new__(cls, *arguments, tag)
        if not _is_new(obj):
            return obj
        obj.system = system
        obj.name = name
        if arguments == own:
            tag.owner = obj
        return obj

    def _applied(self, arguments: tuple[sp.Expr, ...]) -> Self:
        """Return the placeholder applied at `arguments`, as for
        ``u.subs(x, 1)``."""
        arguments = tuple(sp.sympify(a) for a in arguments)
        if arguments == self.args[:-1]:
            return self
        if len(arguments) != len(self.args) - 1:
            raise ValueError(f"{self.name} takes {len(self.args) - 1} arguments")
        tag = cast(_FunctionTag, self.args[-1])
        return type(self)._from_parts(self.name, self.system, tag, arguments)

    def _eval_simplify(self, **kwargs: Any) -> Self:
        return self

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        # The tag's link to its owner does not survive pickling, so pass the
        # name and system explicitly.
        return (
            _restore_placeholder,
            (type(self), self.name, self.system, self.args[-1], self.args[:-1]),
        )


def _restore_placeholder(
    cls: type[_PlaceholderFunction],
    name: str,
    system: CoordSys,
    tag: _FunctionTag,
    arguments: Sequence[sp.Expr],
) -> _PlaceholderFunction:
    owner = cls._from_parts(name, system, tag)
    return owner._applied(tuple(arguments))


class ScalarFunction(_PlaceholderFunction):
    """Physical-domain scalar field placeholder u(x, y, ..) with mapping support.

    Calling .doit() returns the computational-domain representation U(X, Y, ..)
    """

    def doit(self, **hints: Any) -> AppliedUndef:
        """Return function in computational domain"""
        return Function(self.name.upper())(*self.system.base_scalars())  # ty:ignore[call-non-callable]

    @property
    def arg_str(self) -> str:
        return f"({self.args[0]})" if len(self.args) == 2 else str(self.args[:-1])

    def __str__(self) -> str:
        return f"{self.name}{self.arg_str}"

    def _latex(self, printer: Any = None) -> str:  # prob always want latex here?
        return f"{latex_symbol(self.name)}{self.arg_str}"


class VectorFunction(_PlaceholderFunction):
    """Physical-domain vector field placeholder v(x, y, ..).

    The .doit() method builds a VectorAdd of component scalar Functions
    times basis vectors in the computational domain.
    """

    def __str__(self) -> str:
        return "\033[1m%s\033[0m" % (self.name,) + str(self.args[:-1])  # noqa: UP031

    def doit(self, **hints: Any) -> VectorAdd:
        """Return function in computational domain"""
        vn = self.system._variable_names
        return VectorAdd.fromiter(
            Function(f"{self.name.upper()}_{latex_symbol(vn[i])}")(
                *self.system.base_scalars()
            )  # ty:ignore[call-non-callable]
            * bi
            for i, bi in enumerate(self.system.base_vectors())
        )

    def _latex(self, printer: Any = None) -> str:
        return r"\mathbf{{%s}}" % (latex_symbol(self.name),) + str(self.args[:-1])  # noqa: UP031


class Jaxc(sp.Dummy):
    r"""The coefficients of a JAXFunction.

    A JAXFunction represents a complete function in a given function space,
    backed by a JAX array of coefficients. That is, in 1D it represents the
    function u(x) defined by
    .. math::

        u(x) = \sum_i \hat{u}_{i} \phi_i(x)

    where the coefficients :math:`\hat{u}_{i}` are stored in the JAXFunction array,
    and :math:`\phi_i(x)` are the basis functions of the function space.
    The higher dimensional cases (tensor product spaces, vector-valued spaces)
    are handled similarly with different definitions of the basis functions.

    When assembling weak forms, the JAXFunction is expanded into a TrialFunction
    multiplied by the Jaxc symbolic coefficient array. The Jaxc object holds the
    reference to the coefficient array and function space, allowing for assembly
    of the system matrices and vectors.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jaxfun.galerkin import Chebyshev, inner
        >>> from jaxfun.galerkin.arguments import Jaxc, JAXFunction, TestFunction, \
        ... TrialFunction
        >>> V = Chebyshev.Chebyshev(4, name="V")
        >>> w = JAXFunction(jnp.ones(V.dim), V, name="w")
        >>> v = TestFunction(V, name="v")
        >>> b = inner(v * w)
        >>> assert jnp.all(b == jnp.array([3.1415927, 1.5707964, 1.5707964, 1.5707964]))
        >>> u = TrialFunction(V, name="u")
        >>> a = inner(u * v)
        >>> assert jnp.all(b == a @ w.array)
        >>> assert w.doit().__str__() == 'w(x)'
        >>> assert w.doit(linear=True).__str__() == '_\\hat{w}_{j}*T_j(x)'

    Args:
        array: JAX array of coefficients.
        V: Function space instance.

    """

    array: Array
    functionspace: FunctionSpaceType
    argument: Literal[ArgumentTag.JAXFUNC]
    _is_Symbol: bool = True

    def __new__(cls, array: Array, V: FunctionSpaceType, name: str) -> Self:
        obj = super().__new__(cls, name)
        obj.array = array
        obj.functionspace = V
        obj.argument = ArgumentTag.JAXFUNC
        return obj

    def doit(self, **hints: dict) -> Self:
        return self

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        return latex_symbol(self.name)


def get_JAXFunction(
    name: str,
    *,
    array: Array,
    jaxfunction: JAXFunction,
    global_index: int,
    rank: RankTag,
    functionspace: FunctionSpaceType,
    argument: ArgumentTag,
    args: Tuple,
) -> AppliedUndef:
    """Create a symbolic basis function with enriched printing.

    Adds (monkey-patches) __str__, _pretty and _latex methods so tensor
    product / vector indices appear compactly .

    Args:
        name: Base name (usually space.fun_str).
        array: JAXFunction array of coefficients.
        jaxfunction: The JAXFunction the node stands for. It is part of the
            node's identity, so nodes of different JAXFunctions never compare
            equal, even with equal names and spaces.
        global_index: Global component index for vector-valued spaces.
        rank: Tensor rank of the overall function space (0 or 1).
        functionspace: Parent function space object.
        argument: ArgumentTag.TEST for test, ArgumentTag.TRIAL for trial (stored on
            Function).
        args: Underlying BaseScalar symbols (coordinates).

    Returns:
        SymPy Function instance representing one factor basis function.
    """

    # Need additional printing because of the tensor product structure of the basis
    # functions

    def __str__(cls) -> str:
        lhs = f"{cls.name}"

        if cls.rank == RankTag.SCALAR:
            rhs = f"{cls.args}" if len(cls.args) > 1 else f"({cls.args[0].name})"
            return lhs + rhs
        elif cls.rank == RankTag.VECTOR:
            rhs = f"{cls.args}"
            sup = "^{(" + str(cls.global_index) + ")}"
            return lhs + sup + rhs
        raise NotImplementedError("Rank > 1 basis functions not supported.")

    def _pretty(cls, printer: Any = None) -> prettyForm:
        return prettyForm(cls.__str__())

    def _sympystr(cls, printer: Any) -> str:
        return cls.__str__()

    def _latex(cls, printer: Any = None, exp: float | None = None) -> str:
        lhs = f"{latex_symbol(cls.name)}"
        if cls.rank == RankTag.SCALAR:
            rhs = f"({cls.args[0]})" if len(cls.args) == 1 else f"{cls.args}"
            s = lhs + rhs
        elif cls.rank == RankTag.VECTOR:
            rhs = f"{cls.args}"
            sup = "^{(" + str(cls.global_index) + ")}"
            s = lhs + sup + rhs
        else:
            raise NotImplementedError("Rank > 1 basis functions not supported.")
        return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"

    b: AppliedUndef = sp.Function(
        name,
        array=array,
        jaxfunction=jaxfunction,
        global_index=global_index,
        rank=rank,
        functionspace=functionspace,
        argument=argument,
    )(*args)  # ty:ignore[call-non-callable]

    b.__class__.__str__ = __str__  # ty:ignore[invalid-assignment]
    b.__class__._pretty = _pretty  # ty:ignore[unresolved-attribute]
    b.__class__._sympystr = _sympystr  # ty:ignore[unresolved-attribute]
    b.__class__._latex = _latex  # ty:ignore[unresolved-attribute]
    # prevent printing of the raw array in the function args
    del b._kwargs["array"]  # ty:ignore[unresolved-attribute]
    return b


class JAXFunction[SpaceT: FunctionSpaceType](ExpansionFunction[SpaceT]):
    r"""A Galerkin function with explicit expansion coefficients.

    A JAXFunction represents a complete function in a given function space,
    backed by a JAX array of coefficients. That is, in 1D it represents the
    function u(x) defined by

    .. math::

        u(x) = \sum_i \hat{u}_{i} \phi_i(x)

    where the coefficients :math:`\hat{u}_{i}` are stored in the JAXFunction array,
    and :math:`\phi_i(x)` are the basis functions of the function space.

    The higher dimensional cases (tensor product spaces, vector-valued spaces)
    are handled similarly with different definitions of the basis functions.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jaxfun.galerkin import Chebyshev, inner
        >>> from jaxfun.galerkin.arguments import JAXFunction, TestFunction, \
        ... TrialFunction
        >>> V = Chebyshev.Chebyshev(4, name="V")
        >>> w = JAXFunction(jnp.ones(V.dim), V, name="w")
        >>> v = TestFunction(V, name="v")
        >>> b = inner(v * w)
        >>> assert jnp.all(b == jnp.array([3.1415927, 1.5707964, 1.5707964, 1.5707964]))
        >>> u = TrialFunction(V, name="u")
        >>> a = inner(u * v)
        >>> assert jnp.all(b == a @ w.array)

    The sum or difference of two different JAXFunctions is computed on the
    coefficients and is a new JAXFunction of the same size, as is
    ``u.scale(c)``. It lives in the space of the operands when they share one,
    and otherwise in their common orthogonal space; a space with inhomogeneous
    boundary conditions (a DirectSum) is always moved to its orthogonal space
    first. Any other operation stays symbolic, including ``2 * u``, ``u + u``
    and products of JAXFunctions.

        >>> w2 = w.scale(2) - w
        >>> assert jnp.all(w2.array == w.array)

    A JAXFunction can't live in a space with time-dependent boundary
    conditions, because its value would depend on a time it doesn't carry.

    Args:
        array: JAX array of coefficients.
        V: Function space instance.
        name: Optional name for the JAXFunction object. Unnamed JAXFunctions
            are named jf0, jf1, ... in order of creation.
    """

    array: Array | tuple[Array, ...]
    argument: Literal[ArgumentTag.JAXFUNC]
    functionspace: SpaceT

    def __new__(
        cls,
        array: Array | tuple[Array, ...] | sp.Expr | sp.Tuple,
        V: SpaceT,
        name: str | None = None,
        *rebuild: sp.Expr,
    ) -> Self:
        from .inner import project

        if (rebuilt := _rebuilt(array, V, name, *rebuild)) is not None:
            return cast(Self, rebuilt)

        if has_time_dependent_bcs(cast(FunctionSpaceType, V)):
            raise ValueError(
                f"{V.name} has time-dependent boundary conditions, so a function "
                "in it is only defined at a given time. Use its orthogonal space, "
                "V.get_orthogonal(), with the coefficients at that time, "
                "V.to_orthogonal(c, t)."
            )
        if isinstance(array, sp.Expr | sp.Tuple):
            if cast(FunctionSpaceType, V).rank == RankTag.VECTOR:
                assert isinstance(array, sp.Expr)
                array = project(array, cast(VectorTensorProductSpace, V))
            else:
                if isinstance(array, sp.Expr):
                    array = project(array, cast(ScalarSpaceType, V))
                else:
                    array = project(
                        array,
                        cast(CartesianTensorProductSpace | CartesianProductSpace, V),
                    )

        name = name if name is not None else f"jf{next(_jaxfunction_counter)}"
        return cls._from_parts(array, V, name, _FunctionTag())

    @classmethod
    def _from_parts(
        cls,
        array: Array | tuple[Array, ...],
        V: SpaceT,
        name: str,
        tag: _FunctionTag,
        arguments: Sequence[sp.Expr] | None = None,
    ) -> Self:
        """Build the JAXFunction from its coefficients, space, name and the tag
        that makes it unique, applied at `arguments` (by default its own
        coordinates)."""
        coors: CoordSys = V.system
        own = tuple(coors._cartesian_xyz)
        arguments = own if arguments is None else tuple(arguments)
        obj: Self = Function.__new__(cls, *arguments, tag)
        if not _is_new(obj):
            return obj
        obj.array = array
        obj.functionspace = V
        obj.argument = ArgumentTag.JAXFUNC
        obj.name = name
        obj.own_name = "JAXFunction"
        if arguments == own:
            tag.owner = obj
        return obj

    def _applied(self, arguments: tuple[sp.Expr, ...]) -> sp.Expr:
        """Return the function applied at `arguments`: its value when they are
        numbers, as for ``u.subs(x, 0.5)``, and otherwise the function of the
        new arguments, as for ``u.subs(x, 2*y)``."""
        arguments = tuple(sp.sympify(a) for a in arguments)
        if arguments == self.args[:-1]:
            return self
        if len(arguments) != len(self.args) - 1:
            raise ValueError(f"{self.name} takes {len(self.args) - 1} arguments")
        if all(a.is_number for a in arguments):
            return self._value_at(arguments)
        return JAXFunction._from_parts(
            self.array,
            self.functionspace,
            self.name,
            cast(_FunctionTag, self.args[-1]),
            arguments,
        )

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        # The tag's link to its JAXFunction does not survive pickling, so pass
        # the coefficients and space explicitly.
        return (
            _restore_jaxfunction,
            (self.array, self.functionspace, self.name, self.args[-1], self.args[:-1]),
        )

    def evalf(
        self,
        n: int = 15,
        subs: dict | None = None,
        maxn: int = 100,
        chop: bool = False,
        strict: bool = False,
        quad: str | None = None,
        verbose: bool = False,
    ) -> sp.Expr:
        """Return the value at the point given by `subs`, or the function
        itself when no point is given.

        Args:
            n: Number of significant digits.
            subs: Values of all the coordinates, e.g. ``{x: 0.5}``.
            maxn, chop, strict, quad, verbose: As for SymPy's ``evalf``.
        """
        if subs is None:
            return self
        return cast(sp.Expr, self.subs(subs)).evalf(
            n, maxn=maxn, chop=chop, strict=strict, quad=quad, verbose=verbose
        )

    n = evalf

    @property
    def c_names(self) -> str:
        # The arguments it is applied at, its own coordinates unless substituted
        return ", ".join(str(a) for a in self.args[:-1])

    def _sympyrepr(self, printer: Any) -> str:
        return f"JAXFunction({', '.join(printer._print(a) for a in self.args)})"

    @property
    def dim(self) -> int:
        """Dimension of the function space, the number of coefficients."""
        return self.functionspace.dim

    @property
    def num_dofs(self) -> int | tuple[int, ...] | tuple[tuple[int, ...], ...]:
        """Number of coefficients, per axis for a tensor product space."""
        return self.functionspace.num_dofs

    @property
    def dims(self) -> int:
        """Number of spatial dimensions."""
        return self.functionspace.dims

    def _value_at(self, coordinates: Sequence[sp.Expr]) -> sp.Expr:
        """Return the value at a point, as a SymPy number (vector for a vector
        function)."""
        V = cast(FunctionSpaceType, self.functionspace)
        if not V.system._is_cartesian:
            raise NotImplementedError(
                f"{self.name} lives in curvilinear coordinates; evaluate it with "
                "u(points) at computational coordinates instead"
            )
        point = jnp.array([[float(c) for c in coordinates]])
        values = self(point[0, 0] if V.dims == 1 else point)
        values = jnp.ravel(values)
        numbers = [
            sp.Float(float(v)) if not jnp.iscomplexobj(v) else sp.sympify(complex(v))
            for v in values
        ]
        if V.rank == RankTag.VECTOR:
            return VectorAdd.fromiter(
                c * b for c, b in zip(numbers, V.system.base_vectors(), strict=True)
            )
        return numbers[0]

    @overload
    def get_array(self: JAXFunction[_CompositeSpaceT]) -> tuple[Array, ...]: ...
    @overload
    def get_array(self: JAXFunction[_ScalarSpaceT]) -> Array: ...
    def get_array(self) -> Array | tuple[Array, ...]:
        return self.array

    def backward(self, N: Padding = None) -> Array:
        return self.functionspace.backward(self.array, N=N)  # ty: ignore[invalid-argument-type]

    def doit(self, **hints: Any) -> Expr | AppliedUndef:
        hints["linear"] = hints.get("linear", False)
        V = cast(FunctionSpaceType, self.functionspace)

        if hints.get("linear", True):
            trial = TrialFunction(V).doit()
            offset = 1 if isinstance(V, OrthogonalSpace | DirectSum) else V.dims
            local_indices = slice(offset, 2 * offset)
            global_index = 0
            hat = f"\\hat{{{self.name}}}"
            if V.rank == RankTag.SCALAR:
                name = "".join((hat, "_{", indices[local_indices], "}"))
                return Jaxc(cast(Array, self.array), V, name=name) * trial

            elif V.rank == RankTag.VECTOR:
                assert isinstance(V, VectorTensorProductSpace)
                s = []
                for k, v in cast(Vector, trial).components.items():
                    global_index = k._id[0]
                    name = "".join(
                        (
                            hat,
                            "_{",
                            indices[local_indices],
                            "}^{(",
                            str(global_index),
                            ")}",
                        )
                    )
                    s.append(
                        Jaxc(self.array[global_index], V[global_index], name=name)
                        * k
                        * v
                    )
                return trial.func(*s)

            else:
                raise ValueError(
                    "JAXFunction expansion not possible for Cartesian products that "
                    "are not ranked tensors"
                )

        # Nonlinear case, return a multivar function.
        if V.rank == RankTag.SCALAR:
            return get_JAXFunction(
                self.name,
                array=cast(Array, self.array),
                jaxfunction=self,
                global_index=0,
                rank=RankTag.SCALAR,
                functionspace=V,
                argument=ArgumentTag.JAXFUNC,
                args=V.system.base_scalars(),
            )

        elif V.rank == RankTag.VECTOR:
            assert isinstance(V, VectorTensorProductSpace)

            return VectorAdd.fromiter(
                get_JAXFunction(
                    "".join((self.name, "^{(", str(i), ")}")),
                    array=self.array[i],
                    jaxfunction=self,
                    global_index=i,
                    rank=RankTag.SCALAR,
                    functionspace=V[i],
                    argument=ArgumentTag.JAXFUNC,
                    args=V.system.base_scalars(),
                )
                * bi
                for i, bi in enumerate(V.system.base_vectors())
            )

        else:
            raise ValueError(
                "Unranked Composite space not supported for nonlinear expansion."
            )

    def to_orthogonal(self, name: str | None = None) -> JAXFunction:
        """Return the same function as a JAXFunction in the orthogonal basis.

        The boundary conditions of the space, inhomogeneous ones included, are
        folded into the coefficients and no longer imposed.
        """
        V = cast(FunctionSpaceType, self.functionspace)
        return JAXFunction(
            orthogonal_coefficients(V, self.array), V.get_orthogonal(), name=name
        )

    def to_expansion(self) -> OrthogonalExpansion:
        """Return the function as an OrthogonalExpansion, for exact arithmetic
        that changes the number of basis functions (products, ``x * u``).

        The boundary conditions of the space, inhomogeneous ones included, are
        folded into the coefficients. Only 1D spaces are supported.
        """
        from .expansion import OrthogonalExpansion

        assert self.functionspace.dims == 1, "Only 1D spaces supported for expansion"

        return OrthogonalExpansion.from_jaxfunction(self)

    def _linear_combination(
        self, other: JAXFunction, a: complex = 1, b: complex = 1
    ) -> JAXFunction | None:
        """Return ``a*self + b*other`` as a JAXFunction.

        The result lives in the space of the operands when they share one
        without inhomogeneous boundary conditions, and otherwise in their common
        orthogonal space.

        Returns:
            The combination, or None when the operands have no common
            orthogonal space of the same size.
        """
        U = cast(FunctionSpaceType, self.functionspace)
        V = cast(FunctionSpaceType, other.functionspace)
        if U is V and not has_boundary_lifting(U):
            return JAXFunction(_axpby(a, self.array, b, other.array), U)
        W = U.get_orthogonal()
        if orthogonal_signature(W) != orthogonal_signature(V.get_orthogonal()):
            return None
        return JAXFunction(
            _axpby(
                a,
                orthogonal_coefficients(U, self.array),
                b,
                orthogonal_coefficients(V, other.array),
            ),
            W,
        )

    def scale(self, c: complex | Array) -> JAXFunction:
        """Return ``c*self`` as a JAXFunction.

        A space with inhomogeneous boundary conditions is replaced by its
        orthogonal space, since scaling also scales the boundary values.
        """
        V = cast(FunctionSpaceType, self.functionspace)
        if not has_boundary_lifting(V):
            return JAXFunction(jax.tree.map(lambda x: c * x, self.array), V)
        array = orthogonal_coefficients(V, self.array)
        return JAXFunction(jax.tree.map(lambda x: c * x, array), V.get_orthogonal())

    # Only `+` and `-` between two different JAXFunctions are computed on the
    # coefficients. SymPy's own algorithms combine an expression with numbers
    # and with itself through these operators and rely on `1*u` being `u` and
    # `u - u` being 0; a new JAXFunction there makes, e.g., pattern matching in
    # `integrate` run forever.
    def __add__(self, other: Any) -> Any:
        if isinstance(other, JAXFunction) and other is not self:
            result = self._linear_combination(other)
            if result is not None:
                return result
        return super().__add__(other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, JAXFunction) and other is not self:
            result = self._linear_combination(other, 1, -1)
            if result is not None:
                return result
        return super().__sub__(other)

    @overload
    def __matmul__(
        self: JAXFunction[_CompositeSpaceT], a: tuple[Array, ...]
    ) -> tuple[Array, ...]: ...
    @overload
    def __matmul__(self: JAXFunction[_ScalarSpaceT], a: Array) -> Array: ...
    def __matmul__(self, a: Array | tuple[Array, ...]) -> Array | tuple[Array, ...]:
        if isinstance(self.array, tuple):
            assert isinstance(a, tuple) and len(self.array) == len(a)
            return tuple(ai @ ai_ for ai, ai_ in zip(self.array, a))
        assert isinstance(a, Array)
        return self.array @ a

    @overload
    def __rmatmul__(
        self: JAXFunction[_CompositeSpaceT], a: tuple[Array, ...]
    ) -> tuple[Array, ...]: ...
    @overload
    def __rmatmul__(self: JAXFunction[_ScalarSpaceT], a: Array) -> Array: ...
    def __rmatmul__(self, a: Array | tuple[Array, ...]) -> Array | tuple[Array, ...]:
        if isinstance(self.array, tuple):
            assert isinstance(a, tuple) and len(self.array) == len(a)
            return tuple(ai_ @ ai for ai, ai_ in zip(a, self.array))
        assert isinstance(a, Array)
        return a @ self.array

    def __call__(self, x: float | Array) -> Array:
        """Evaluate the JAXFunction at given points x in the physical domain.

        Args:
            x: Coordinates (N, d). Created, e.g., by calling
                self.functionspace.flatmesh().
        """
        V = cast(FunctionSpaceType, self.functionspace)
        if isinstance(V, OrthogonalSpace | DirectSum):
            return V.evaluate(x, cast(Array, self.array))

        x = jnp.asarray(x)
        if isinstance(V, TensorProductSpace):
            z = V.evaluate(x, cast(Array, self.array))
        else:
            z = V.evaluate(x, cast(tuple[Array, ...], self.array))

        if V.rank == RankTag.SCALAR:
            return jnp.expand_dims(z, -1)
        return z

    def evaluate_mesh(
        self, kind: MeshKind | str = MeshKind.QUADRATURE, N: Padding = None
    ) -> Array:
        """Evaluate the JAXFunction at tensor mesh in the physical domain.

        Args:
            kind: Mesh type (quadrature, uniform).
            N: Optional padding for the number of points in each dimension.
        """
        return self.functionspace.evaluate_mesh(self.array, kind=kind, N=N)  # ty:ignore[invalid-argument-type]


def _restore_jaxfunction(
    array: Array | tuple[Array, ...],
    V: Any,
    name: str,
    tag: _FunctionTag,
    arguments: Sequence[sp.Expr],
) -> JAXFunction:
    owner = JAXFunction._from_parts(array, V, name, tag)
    if tuple(arguments) == owner.args[:-1]:
        return owner
    return cast(JAXFunction, owner._applied(tuple(arguments)))


def has_boundary_lifting(V: FunctionSpaceType) -> bool:
    """Return True if V, or a component of V, has inhomogeneous boundary
    conditions (a DirectSum or DirectSumTPS)."""
    if isinstance(V, CartesianBaseSpace):
        return any(has_boundary_lifting(space) for space in V.flatten())
    return isinstance(V, DirectSum | DirectSumTPS)


def has_time_dependent_bcs(V: FunctionSpaceType) -> bool:
    """Return True if V, or a component of V, has boundary values that depend
    on time."""
    if isinstance(V, CartesianBaseSpace):
        return any(has_time_dependent_bcs(space) for space in V.flatten())
    if isinstance(V, DirectSumTPS):
        return V.lifting.is_transient
    if isinstance(V, DirectSum):
        return V.bcs.has_time(V.system.base_time())
    return False


def _check_time_independent(V: FunctionSpaceType) -> None:
    if has_time_dependent_bcs(V):
        raise ValueError(
            f"{V.name} has time-dependent boundary conditions, so its "
            "coefficients have no orthogonal counterpart without a time `t`."
        )


def orthogonal_coefficients(
    V: FunctionSpaceType, c: Array | tuple[Array, ...], t: float | Array | None = None
) -> Array | tuple[Array, ...]:
    """Return the coefficients c of V in V's orthogonal basis, with boundary
    values evaluated at time t.

    Raises:
        ValueError: If V has time-dependent boundary conditions and t is None.
    """
    if t is None:
        _check_time_independent(V)
    if isinstance(V, CartesianBaseSpace):
        return tuple(
            cast(Array, orthogonal_coefficients(space, c[i], t))
            for i, space in enumerate(V.flatten())
        )
    if isinstance(V, DirectSum | DirectSumTPS):
        return V.to_orthogonal(cast(Array, c), t)
    return V.to_orthogonal(cast(Array, c))


def orthogonal_signature(V: FunctionSpaceType) -> tuple[Any, ...]:
    """Return what identifies the orthogonal space V up to its name.

    Two orthogonal spaces with equal signatures have the same basis functions,
    so coefficients in one are coefficients in the other.
    """
    if isinstance(V, CartesianBaseSpace):
        return (V.system, tuple(orthogonal_signature(s) for s in V.flatten()))
    if isinstance(V, TensorProductSpace):
        return (V.system, tuple(orthogonal_signature(s) for s in V.basespaces))
    parameters = getattr(V, "parameters", {})
    return (type(V), tuple(parameters.items()), V.domain, V.system, V.N)


def _is_number(c: Any) -> bool:
    if isinstance(c, bool):
        return False
    if isinstance(c, numbers.Number):
        return True
    if isinstance(c, sp.Basic):
        return bool(c.is_number)
    return isinstance(c, Array) and c.ndim == 0


def _as_number(c: Any) -> Any:
    if isinstance(c, sp.Basic):
        return float(c) if c.is_real else complex(c)
    return c


def _axpby(
    a: complex, x: Array | tuple[Array, ...], b: complex, y: Array | tuple[Array, ...]
) -> Array | tuple[Array, ...]:
    return jax.tree.map(lambda xi, yi: a * xi + b * yi, x, y)


def evaluate_jaxfunction_expr_quad(
    a: Basic,
    N: int | tuple[int | None, ...] | None = None,
    space: ScalarSpaceType | None = None,
) -> Array:
    """Evaluate a symbolic JAXFunction expression on the quadrature mesh.

    Args:
        a: SymPy expression potentially containing JAXFunction objects.
        N: Optional padding for the number of quadrature points in each dimension.
        space: Space whose quadrature mesh to evaluate on, which may belong to
            another family than the JAXFunctions along any axis. Defaults to
            the space of each JAXFunction.
    """
    from jaxfun.integrators.nonlinear import compile_nonlinear_evaluator

    from .forms import get_jaxfunctions

    jaxfs: set[JAXFunction] = get_jaxfunctions(a)

    if len(jaxfs) > 1:
        # Multiple JAXFunction components present.  Split into
        # per-component sub-expressions and evaluate each independently.

        if isinstance(a, sp.Add):
            result = jnp.asarray(0.0)
            for arg in a.args:
                result = result + evaluate_jaxfunction_expr_quad(arg, N, space)
            return result

        if not isinstance(a, sp.Mul):
            raise NotImplementedError(
                f"Multiple JAXFunctions in non-Mul expression not supported: {a}"
            )
        # Partition args by the identity of the single JAXFunction they
        # depend on.  Args that share the same JAXFunction (e.g.
        # U^(0) and d(U^(0))/dx) are grouped together so the nonlinear
        # compiler can handle them as a single sub-expression.
        groups: dict[int, sp.Expr] = {}
        for arg in a.args:
            arg_jaxfs = get_jaxfunctions(arg)
            if len(arg_jaxfs) == 0:
                # Static/numeric factor: attach to an arbitrary group so it
                # is folded in; use key -1 to signal "no JAXFunction".
                key = -1
            elif len(arg_jaxfs) == 1:
                key = id(next(iter(arg_jaxfs)))
            else:
                # arg itself has multiple JAXFunctions — treat as its own
                # group and let the recursion resolve it.
                key = id(arg)
            groups[key] = cast(sp.Expr, groups.get(key, sp.Integer(1)) * arg)

        result = jnp.asarray(1.0)
        for key, sub_expr in groups.items():
            if key == -1:
                # Pure numeric/symbolic constant — evaluate to a scalar.
                result = result * float(sub_expr)
            else:
                result = result * evaluate_jaxfunction_expr_quad(sub_expr, N, space)
        return result

    assert len(jaxfs) == 1, "Single JAXFunction not found in expression."
    jaxf = jaxfs.pop()
    V = jaxf.functionspace if space is None else space
    fun = compile_nonlinear_evaluator(cast(sp.Expr, a), V, cast(AppliedUndef, jaxf))
    return fun(cast(Array, jaxf.array), N)
