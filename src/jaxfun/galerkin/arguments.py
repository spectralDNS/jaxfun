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

from abc import abstractmethod
from enum import Enum, unique
from functools import partial
from typing import Any, Literal, Self, TypeVar, cast, overload

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from sympy import Basic, Expr, Function, Tuple
from sympy.core.function import AppliedUndef
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import VectorAdd

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, CoordSys, Vector, latex_symbols
from jaxfun.galerkin.cartesianproductspace import (
    CartesianProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.typing import (
    ComputationalSpaceType,
    FunctionSpaceType,
    MeshKind,
    Padding,
    ScalarSpaceType,
    TestSpaceType,
    TrialSpaceType,
)

from .composite import DirectSum
from .orthogonal import OrthogonalSpace
from .tensorproductspace import DirectSumTPS, TensorProductSpace

_CompositeSpaceT = TypeVar("_CompositeSpaceT", bound=CartesianProductSpace)
_ScalarSpaceT = TypeVar("_ScalarSpaceT", bound=ScalarSpaceType)

t, x, y, z = sp.symbols("t,x,y,z", real=True)

indices = "ijklmn"


@unique
class ArgumentTag(Enum):
    TEST = 0
    TRIAL = 1
    JAXFUNC = 2
    NONE = -1


def get_arg(p: Any) -> ArgumentTag:
    return getattr(p, "argument", ArgumentTag.NONE)


def get_BasisFunction(
    name: str,
    *,
    vector_index: int,
    local_index: int,
    global_index: int,
    rank: int,
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
        if cls.rank == 0:
            return lhs + rhs
        elif cls.rank == 1:
            sup = "^{(" + str(cls.vector_index) + ")}"
            return lhs + sup + rhs
        raise NotImplementedError("Rank > 1 basis functions not supported.")

    def _pretty(cls, printer: Any = None) -> prettyForm:
        return prettyForm(cls.__str__())

    def _sympystr(cls, printer: Any) -> str:
        return cls.__str__()

    def _latex(cls, printer: Any = None, exp: float | None = None) -> str:
        index = indices[cls.local_index + cls.offset]
        lhs = f"{latex_symbols[cls.name]}_{index}"
        rhs = f"({latex_symbols[cls.args[0].name]})"
        if cls.rank == 0:
            s = lhs + rhs
        elif cls.rank == 1:
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
            rank=0,
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
                rank=0,
                offset=offset,
                functionspace=v,
                argument=ArgumentTag.TEST if arg == "test" else ArgumentTag.TRIAL,
                arg=a,
            )
            for a, v in zip(base_scalars, V.basespaces, strict=False)
        )

    assert isinstance(V, VectorTensorProductSpace)
    b = V.system.base_vectors()
    return VectorAdd.fromiter(
        sp.Mul.fromiter(
            get_BasisFunction(
                v.fun_str,
                vector_index=i,
                local_index=getattr(a, "_id", [0])[0],
                global_index=getattr(Vi, "global_index", 0),
                rank=1,
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


class ExpansionFunction(BaseFunction):
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

    functionspace: FunctionSpaceType
    own_name: str

    @property
    def c_names(self) -> str:
        return ", ".join([i.name for i in self.functionspace.system._cartesian_xyz])

    def __str__(self) -> str:
        V = self.functionspace
        name = "\033[1m%s\033[0m" % (self.name,) if V.rank == 1 else self.name  # noqa: UP031
        return f"{name}({self.c_names}; {V.name})"

    def _latex(self, printer: Any = None) -> str:
        name = self.name
        if name != self.own_name:
            assert not isinstance(self.functionspace, DirectSum)
            if self.functionspace.rank == 1:
                name = r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
        return f"{name}({self.c_names}; {self.functionspace.name})"

    def __getitem__(self, i: int) -> sp.Expr:
        assert isinstance(self.functionspace, CartesianProductSpace)
        return self.__class__(self.functionspace[i], name=self.name[i])


# NOTE
# Need a unique Symbol in order to create a new TestFunction/TrialFunction for a new
# space. Without it all TestFunctions/TrialFunctions created with the same Cartesian
# coordinates will be the same object.
class TestFunction(ExpansionFunction):
    """Symbolic test function T(x; V) for weak form assembly.

    Holds a reference to its functionspace and lazily expands (via doit)
    into computational (tensor product) factor basis functions.
    """

    __test__ = False  # prevent pytest from considering this a test.
    argument: Literal[ArgumentTag.TEST]
    functionspace: TestSpaceType

    def __new__(
        cls,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        coors = V.system
        obj: Self = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Dummy()]))
        obj.argument = ArgumentTag.TEST

        if isinstance(V, DirectSum | DirectSumTPS | CartesianProductSpace):
            obj.functionspace = V.get_homogeneous()
        elif isinstance(V, OrthogonalSpace | TensorProductSpace):
            obj.functionspace = V
        else:
            raise ValueError("Unknown test space")

        obj.name = name if name is not None else "TestFunction"
        obj.own_name = "TestFunction"
        return obj

    def doit(self, **hints: Any) -> Expr | AppliedUndef:
        if self.functionspace.rank < 0:
            raise ValueError(
                "TestFunction expansion not possible for CartesianProductSpace"
            )
        return _get_computational_function(
            "test", cast(ComputationalSpaceType, self.functionspace)
        )


class TrialFunction(ExpansionFunction):
    """Symbolic trial function U(x; V) for weak form assembly.

    Direct sums expand to sum of component spaces. Tensor product spaces
    with direct-sum factors expand into additive combinations of products.
    """

    argument: Literal[ArgumentTag.TRIAL]
    functionspace: TrialSpaceType
    transient: bool

    def __new__(
        cls,
        V: FunctionSpaceType,
        name: str | None = None,
        transient: bool = False,
    ) -> Self:
        coors = V.system
        time_arg = [coors.base_time()] if transient else []
        obj: Self = Function.__new__(
            cls, *(list(coors._cartesian_xyz) + time_arg + [sp.Dummy()])
        )
        obj.functionspace = V
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
                                rank=1,
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


class ScalarFunction(BaseFunction):
    """Physical-domain scalar field placeholder u(x, y, ..) with mapping support.

    Calling .doit() returns the computational-domain representation U(X, Y, ..)
    """

    system: CoordSys

    def __new__(cls, name: str, system: CoordSys) -> Self:
        obj: Self = Function.__new__(cls, *(list(system._cartesian_xyz) + [sp.Dummy()]))
        obj.system = system
        obj.name = name.lower()
        return obj

    def doit(self, **hints: Any) -> AppliedUndef:
        """Return function in computational domain"""
        return Function(self.name.upper())(*self.system.base_scalars())  # ty:ignore[call-non-callable]

    @property
    def arg_str(self) -> str:
        return f"({self.args[0]})" if len(self.args) == 2 else str(self.args[:-1])

    def __str__(self) -> str:
        return f"{self.name}{self.arg_str}"

    def _latex(self, printer: Any = None) -> str:  # prob always want latex here?
        return f"{latex_symbols[self.name]}{self.arg_str}"


class VectorFunction(BaseFunction):
    """Physical-domain vector field placeholder v(x, y, ..).

    The .doit() method builds a VectorAdd of component scalar Functions
    times basis vectors in the computational domain.
    """

    system: CoordSys

    def __new__(cls, name: str, system: CoordSys) -> Self:
        obj: Self = Function.__new__(cls, *(list(system._cartesian_xyz) + [sp.Dummy()]))
        obj.system = system
        obj.name = name.lower()
        return obj

    def __str__(self) -> str:
        return "\033[1m%s\033[0m" % (self.name,) + str(self.args[:-1])  # noqa: UP031

    def doit(self, **hints: Any) -> VectorAdd:
        """Return function in computational domain"""
        vn = self.system._variable_names
        return VectorAdd.fromiter(
            Function(f"{self.name.upper()}_{latex_symbols[vn[i]]}")(
                *self.system.base_scalars()
            )  # ty:ignore[call-non-callable]
            * bi
            for i, bi in enumerate(self.system.base_vectors())
        )

    def _latex(self, printer: Any = None) -> str:
        return r"\mathbf{{%s}}" % (latex_symbols[self.name],) + str(self.args[:-1])  # noqa: UP031


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
        return latex_symbols[self.name]


def get_JAXFunction(
    name: str,
    *,
    array: Array,
    global_index: int,
    rank: int,
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

        if cls.rank == 0:
            rhs = f"{cls.args}" if len(cls.args) > 1 else f"({cls.args[0].name})"
            return lhs + rhs
        elif cls.rank == 1:
            rhs = f"{cls.args}"
            sup = "^{(" + str(cls.global_index) + ")}"
            return lhs + sup + rhs
        raise NotImplementedError("Rank > 1 basis functions not supported.")

    def _pretty(cls, printer: Any = None) -> prettyForm:
        return prettyForm(cls.__str__())

    def _sympystr(cls, printer: Any) -> str:
        return cls.__str__()

    def _latex(cls, printer: Any = None, exp: float | None = None) -> str:
        lhs = f"{latex_symbols[cls.name]}"
        if cls.rank == 0:
            rhs = (
                f"({latex_symbols[cls.args[0]]})"
                if len(cls.args) == 1
                else f"{latex_symbols[cls.args]}"
            )
            s = lhs + rhs
        elif cls.rank == 1:
            rhs = f"{latex_symbols[cls.args]}"
            sup = "^{(" + str(cls.global_index) + ")}"
            s = lhs + sup + rhs
        else:
            raise NotImplementedError("Rank > 1 basis functions not supported.")
        return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"

    b: AppliedUndef = sp.Function(
        name,
        array=array,
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


class JAXFunction[SpaceT: FunctionSpaceType](ExpansionFunction):
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

    Args:
        array: JAX array of coefficients.
        V: Function space instance.
        name: Optional name for the JAXFunction object.
    """

    array: Array | tuple[Array, ...]
    argument: Literal[ArgumentTag.JAXFUNC]
    functionspace: SpaceT

    def __new__(
        cls,
        array: Array | tuple[Array, ...] | sp.Expr | sp.Tuple,
        V: SpaceT,
        name: str | None = None,
    ) -> Self:
        from .inner import project

        coors: CoordSys = V.system
        obj: Self = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Dummy()]))
        if isinstance(array, sp.Expr):
            array = project(array, cast(ScalarSpaceType, V))

        elif isinstance(array, sp.Tuple):
            array = project(array, cast(CartesianProductSpace, V))

        obj.array = array
        obj.functionspace = V
        obj.argument = ArgumentTag.JAXFUNC
        obj.name = name if name is not None else "JAXFunction"
        obj.own_name = "JAXFunction"
        return obj

    @overload
    def backward(
        self: JAXFunction[_CompositeSpaceT], N: Padding = None
    ) -> tuple[Array, ...]: ...
    @overload
    def backward(self: JAXFunction[_ScalarSpaceT], N: Padding = None) -> Array: ...
    def backward(self, N: Padding = None) -> Array | tuple[Array, ...]:
        return self.functionspace.backward(self.array, N=N)  # ty: ignore[invalid-argument-type]

    def doit(self, **hints: Any) -> Expr | AppliedUndef:
        hints["linear"] = hints.get("linear", False)
        V = self.functionspace

        if hints.get("linear", True):
            trial = TrialFunction(V).doit()
            offset = 1 if isinstance(V, OrthogonalSpace | DirectSum) else V.dims
            local_indices = slice(offset, 2 * offset)
            global_index = 0
            hat = f"\\hat{{{self.name}}}"
            if V.rank == 0:
                name = "".join((hat, "_{", indices[local_indices], "}"))
                return Jaxc(cast(Array, self.array), V, name=name) * trial

            elif V.rank == 1:
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
                    "Unranked Composite space not supported for linear expansion."
                )

        # Nonlinear case, return a multivar function.
        if V.rank == 0:
            return get_JAXFunction(
                self.name,
                array=cast(Array, self.array),
                global_index=0,
                rank=0,
                functionspace=V,
                argument=ArgumentTag.JAXFUNC,
                args=V.system.base_scalars(),
            )

        elif V.rank == 1:
            assert isinstance(V, VectorTensorProductSpace)

            return VectorAdd.fromiter(
                get_JAXFunction(
                    "".join((self.name, "^{(", str(i), ")}")),
                    array=self.array[i],
                    global_index=i,
                    rank=0,
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

    @overload
    def __matmul__(
        self: JAXFunction[_CompositeSpaceT], a: tuple[Array, ...]
    ) -> tuple[Array, ...]: ...
    @overload
    def __matmul__(self: JAXFunction[_ScalarSpaceT], a: Array) -> Array: ...
    @jax.jit(static_argnums=0)
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
    @jax.jit(static_argnums=0)
    def __rmatmul__(self, a: Array | tuple[Array, ...]) -> Array | tuple[Array, ...]:
        if isinstance(self.array, tuple):
            assert isinstance(a, tuple) and len(self.array) == len(a)
            return tuple(ai_ @ ai for ai, ai_ in zip(a, self.array))
        assert isinstance(a, Array)
        return a @ self.array

    @jax.jit(static_argnums=0)
    def __call__(self, x: Array) -> Array:
        """Evaluate the JAXFunction at given points x in the physical domain.

        Args:
            x: Coordinates (N, d). Created by calling self.functionspace.flatmesh().
        """
        V = cast(FunctionSpaceType, self.functionspace)
        if isinstance(V, OrthogonalSpace | DirectSum | TensorProductSpace):
            return V.evaluate(x, cast(Array, self.array))

        z = V.evaluate(x, cast(tuple[Array, ...], self.array))
        if V.rank == 0:
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


def evaluate_jaxfunction_expr_quad(
    a: Basic,
    N: int | tuple[int | None, ...] | None = None,
) -> Array:
    """Evaluate a symbolic JAXFunction expression on the quadrature mesh.

    Args:
        a: SymPy expression potentially containing JAXFunction objects.
        N: Optional padding for the number of quadrature points in each dimension.
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
                result = result + evaluate_jaxfunction_expr_quad(arg, N=N)
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
                result = result * evaluate_jaxfunction_expr_quad(sub_expr, N=N)
        return result

    assert len(jaxfs) == 1, "Single JAXFunction not found in expression."
    jaxf = jaxfs.pop()
    V = jaxf.functionspace
    fun = compile_nonlinear_evaluator(cast(sp.Expr, a), V, cast(AppliedUndef, jaxf))
    return fun(cast(Array, jaxf.array), N)
