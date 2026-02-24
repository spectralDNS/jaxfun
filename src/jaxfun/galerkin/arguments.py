"""Symbolic argument (test/trial) and basis function utilities for Galerkin.

This module builds SymPy Function objects that encode basis functions,
test/trial functions and JAX-backed coefficient arrays over different
function space types (orthogonal, tensor product, vector, direct sums).

Key constructs:
    * TestFunction / TrialFunction: Weak form symbolic arguments.
    * ScalarFunction / VectorFunction: Physical-domain symbolic fields. No basis.
    * JAXFunction: Galerkin functions with JAX-backed coefficients.
"""

import itertools
from abc import abstractmethod
from enum import Enum, unique
from functools import partial
from typing import Any, Literal, Self, cast

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
from jaxfun.typing import FunctionSpaceType, TestSpaceType, TrialSpaceType

from .composite import DirectSum
from .orthogonal import OrthogonalSpace
from .tensorproductspace import (
    DirectSumTPS,
    TensorProductSpace,
    VectorTensorProductSpace,
)

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
    global_index: int,
    local_index: int,
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
        global_index: Global component index for vector-valued spaces.
        local_index: Local index within a 1D factor space.
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
            sup = "^{(" + str(cls.global_index) + ")}"
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
            sup = "^{(" + str(cls.global_index) + ")}"
            s = lhs + sup + rhs
        else:
            raise NotImplementedError("Rank > 1 basis functions not supported.")
        return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"

    b: AppliedUndef = sp.Function(
        name,
        global_index=global_index,
        local_index=local_index,
        rank=rank,
        offset=offset,
        functionspace=functionspace,
        argument=argument,
    )(arg)  # ty:ignore[call-non-callable]

    b.__class__.__str__ = __str__
    b.__class__._pretty = _pretty
    b.__class__._sympystr = _sympystr
    b.__class__._latex = _latex
    return b


def _get_computational_function(arg: str, V: TestSpaceType) -> Expr | AppliedUndef:
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
            global_index=0,
            local_index=0,
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
                global_index=0,
                local_index=getattr(a, "_id", [0])[0],
                rank=0,
                offset=offset,
                functionspace=v,
                argument=ArgumentTag.TEST if arg == "test" else ArgumentTag.TRIAL,
                arg=a,
            )
            for a, v in zip(base_scalars, V.basespaces, strict=False)
        )

    elif isinstance(V, VectorTensorProductSpace):
        b = V.system.base_vectors()
        return VectorAdd.fromiter(
            sp.Mul.fromiter(
                get_BasisFunction(
                    v.fun_str,
                    global_index=i,
                    local_index=getattr(a, "_id", [0])[0],
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
    def c_names(self) -> list[str]:
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
        if isinstance(V, DirectSum):
            obj.functionspace = V[0]
        elif isinstance(V, TensorProductSpace | DirectSumTPS):
            f = []
            vname = V.name
            for space in V.basespaces:
                if isinstance(space, DirectSum):
                    f.append(space[0])
                else:
                    f.append(space)
            obj.functionspace = TensorProductSpace(f, name=vname, system=V.system)
        elif isinstance(V, VectorTensorProductSpace):
            f = []
            vname = V.name
            for space in V:
                g = []
                for s in space:
                    if isinstance(s, DirectSum):
                        g.append(s[0])
                    else:
                        g.append(s)
                f.append(TensorProductSpace(g, name=vname, system=V.system))
            obj.functionspace = VectorTensorProductSpace(tuple(f), name=V.name)
        else:
            obj.functionspace = V
        obj.name = name if name is not None else "TestFunction"
        obj.own_name = "TestFunction"
        return obj

    def doit(self, **hints: Any) -> Expr | AppliedUndef:
        return _get_computational_function("test", self.functionspace)


class TrialFunction(ExpansionFunction):
    """Symbolic trial function U(x; V) for weak form assembly.

    Direct sums expand to sum of component spaces. Tensor product spaces
    with direct-sum factors expand into additive combinations of products.
    """

    argument: Literal[ArgumentTag.TRIAL]
    functionspace: TrialSpaceType

    def __new__(
        cls,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        coors = V.system
        obj: Self = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Dummy()]))
        obj.functionspace = V
        obj.name = name if name is not None else "TrialFunction"
        obj.own_name = "TrialFunction"
        obj.argument = ArgumentTag.TRIAL
        return obj

    def doit(self, **hints: dict) -> Expr | AppliedUndef:
        fspace = self.functionspace
        comp_fun = partial(_get_computational_function, "trial")
        if isinstance(fspace, DirectSum):
            return sp.Add.fromiter(comp_fun(f) for f in fspace.basespaces)
        elif isinstance(fspace, TensorProductSpace):
            spaces = fspace.basespaces
            f = []
            for space in spaces:
                if isinstance(space, DirectSum):
                    f.append(space)
                else:
                    f.append([space])
            tensorspaces = itertools.product(*f)
            return sp.Add.fromiter(
                comp_fun(TensorProductSpace(s, fspace.system, f"{fspace.name}{i}"))
                for i, s in enumerate(tensorspaces)
            )
        elif isinstance(fspace, VectorTensorProductSpace):
            vector = []
            for i, (bi, tpspaces) in enumerate(
                zip(fspace.system.base_vectors(), fspace.tensorspaces, strict=True)
            ):
                fi = []
                spaces = tpspaces.basespaces
                for space in spaces:
                    if isinstance(space, DirectSum):
                        fi.append(space)
                    else:
                        fi.append([space])
                tpspaces = itertools.product(*fi)
                for Vi in tpspaces:
                    T = TensorProductSpace(Vi, fspace.system, f"{fspace.name}{i}")
                    f = (
                        sp.Mul.fromiter(
                            get_BasisFunction(
                                v.fun_str,
                                global_index=i,
                                local_index=getattr(a, "_id", [0])[0],
                                rank=1,
                                offset=fspace.dims,
                                functionspace=v,
                                argument=ArgumentTag.TRIAL,
                                arg=a,
                            )
                            for a, v in zip(
                                fspace.system.base_scalars(), T, strict=True
                            )
                        )
                        * bi
                    )
                    vector.append(f)
            return VectorAdd.fromiter(vector)

        return comp_fun(fspace)


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
        >>> assert w.doit().__str__() == r'\hat{w}_{j}*T_j(x)'
        >>> assert isinstance(w.doit().args[0], Jaxc)

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

    b.__class__.__str__ = __str__
    b.__class__._pretty = _pretty
    b.__class__._sympystr = _sympystr
    b.__class__._latex = _latex
    del b._kwargs["array"]  # prevent printing of the raw array in the function args
    return b


class JAXFunction(ExpansionFunction):
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

    array: Array
    argument: Literal[ArgumentTag.JAXFUNC]
    functionspace: FunctionSpaceType

    def __new__(
        cls,
        array: Array | sp.Expr,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        coors: CoordSys = V.system
        obj: Self = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Dummy()]))
        if isinstance(array, sp.Expr):
            from .inner import project

            array = project(array, V)
        obj.array = array
        dof_shape = V.num_dofs if V.dims > 1 else (V.num_dofs,)
        assert array.shape == dof_shape, (
            f"Array shape {array.shape} does not match number of DOFs {V.num_dofs} for function space {V.name}"  # noqa: E501
        )

        obj.functionspace = V
        obj.argument = ArgumentTag.JAXFUNC
        obj.name = name if name is not None else "JAXFunction"
        obj.own_name = "JAXFunction"
        return obj

    def backward(self):
        assert not isinstance(self.functionspace, VectorTensorProductSpace)
        return self.functionspace.backward(self.array)

    def doit(self, **hints: Any) -> Expr | AppliedUndef:
        hints["linear"] = hints.get("linear", False)
        fs = self.functionspace
        rank = getattr(fs, "rank", 0)

        if hints.get("linear", True):
            trial = TrialFunction(fs).doit()
            offset = 1 if isinstance(fs, OrthogonalSpace | DirectSum) else fs.dims
            local_indices = slice(offset, 2 * offset)
            global_index = 0
            hat = f"\\hat{{{self.name}}}"
            if rank == 0:
                name = "".join((hat, "_{", indices[local_indices], "}"))
                return Jaxc(self.array, fs, name=name) * trial

            assert rank == 1
            assert isinstance(fs, VectorTensorProductSpace)
            s = []
            for k, v in cast(Vector, trial).components.items():
                global_index = k._id[0]
                name = "".join(
                    (hat, "_{", indices[local_indices], "}^{(", str(global_index), ")}")
                )
                s.append(
                    Jaxc(self.array[global_index], fs[global_index], name=name) * k * v
                )
            return trial.func(*s)

        # Nonlinear case, return a multivar function.
        if rank == 0:
            return get_JAXFunction(
                self.name,
                array=self.array,
                global_index=0,
                rank=0,
                functionspace=fs,
                argument=ArgumentTag.JAXFUNC,
                args=fs.system.base_scalars(),
            )

        assert rank == 1
        assert isinstance(fs, VectorTensorProductSpace)

        return VectorAdd.fromiter(
            get_JAXFunction(
                "".join((self.name, "^{(", str(i), ")}")),
                array=self.array[i],
                global_index=i,
                rank=0,
                functionspace=fs[i],
                argument=ArgumentTag.JAXFUNC,
                args=fs.system.base_scalars(),
            )
            * bi
            for i, bi in enumerate(fs.system.base_vectors())
        )

    @jax.jit(static_argnums=0)
    def __matmul__(self, a: Array) -> Array:
        return self.array @ a

    @jax.jit(static_argnums=0)
    def __rmatmul__(self, a: Array) -> Array:
        return a @ self.array

    @jax.jit(static_argnums=0)
    def __call__(self, x: Array) -> Array:
        """Evaluate the JAXFunction at given points x in the physical domain.

        Args:
            x: Coordinates (N, d). Created by calling self.functionspace.flatmesh().
        """
        if isinstance(self.functionspace, OrthogonalSpace | DirectSum):
            X = self.functionspace.map_reference_domain(x)
            return self.functionspace.evaluate(X, self.array)
        z = self.functionspace.evaluate(x, self.array, True)
        if self.functionspace.rank == 0:
            return jnp.expand_dims(z, -1)
        return z

    @jax.jit(static_argnums=0)
    def evaluate_mesh(self, x: Array | list[Array]) -> Array:
        """Evaluate the JAXFunction at given points x in the physical domain.

        Args:
            x: Cartesian product coordinates. For example, for a 2D tensor product
            space, x should be a list of two arrays [x1, x2], where x1 and x2 are 1D
            arrays of coordinates in each dimension. Such a mesh is formed by calling
            self.functionspace.mesh().
        """
        if isinstance(self.functionspace, OrthogonalSpace | DirectSum):
            assert isinstance(x, Array)
            X = self.functionspace.map_reference_domain(x)
            return self.functionspace.evaluate(X, self.array)

        return self.functionspace.evaluate_mesh(x, self.array, True)


def evaluate_jaxfunction_expr(
    a: Basic, xj: Array | tuple[Array, ...], jaxf: AppliedUndef | None = None
) -> Array:
    if jaxf is None:
        for p in sp.core.traversal.preorder_traversal(a):
            if get_arg(p) is ArgumentTag.JAXFUNC:  # JAXFunction->AppliedUndef
                jaxf = cast(AppliedUndef, p)
                break
    assert hasattr(jaxf, "functionspace") and hasattr(jaxf, "array")
    V = cast(FunctionSpaceType, jaxf.functionspace)
    if isinstance(a, sp.Pow):
        wa = a.args[0]
        variables = getattr(wa, "variables", ())
        var = tuple(variables.count(s) for s in V.system.base_scalars())
        var = var[0] if V.dims == 1 else var
        h = V.evaluate_derivative(xj, jaxf.array, k=var)
        h = h ** int(a.exp)

    elif isinstance(a, sp.Derivative):
        variables = getattr(a, "variables", ())
        var = tuple(variables.count(s) for s in V.system.base_scalars())
        var = var[0] if V.dims == 1 else var
        h = V.evaluate_derivative(xj, jaxf.array, k=var)

    else:
        if not isinstance(V, OrthogonalSpace | DirectSum):
            h = V.evaluate_mesh(xj, jaxf.array, True)
        else:
            h = V.evaluate(xj, jaxf.array)
    return h
