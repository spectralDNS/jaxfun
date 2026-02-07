"""Symbolic argument (test/trial) and basis function utilities for Galerkin.

This module builds SymPy Function objects that encode basis functions,
test/trial functions and JAX-backed coefficient arrays over different
function space types (orthogonal, tensor product, vector, direct sums).

Key constructs:
    * TestFunction / TrialFunction: Weak form symbolic arguments.
    * ScalarFunction / VectorFunction: Physical-domain symbolic fields.
    * JAXArray / JAXFunction: Bridge between symbolic and JAX arrays.
"""

import itertools
from abc import abstractmethod
from functools import partial
from typing import Any, Self, cast

import jax
import sympy as sp
from jax import Array
from sympy import Expr, Function
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

functionspacedict = {}

indices = "ijklmn"


def get_BasisFunction(
    name: str,
    *,
    global_index: int,
    local_index: int,
    rank: int,
    offset: int,
    functionspace: BaseSpace,
    argument: int,
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
        argument: 0 for test, 1 for trial (stored on Function).
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
    functionspacedict[V.name] = V
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
            argument=0 if arg == "test" else 1,
            arg=base_scalars[0],
        )

    elif isinstance(V, TensorProductSpace):
        for space in V.basespaces:
            functionspacedict[space.name] = space
        return sp.Mul.fromiter(
            get_BasisFunction(
                v.fun_str,
                global_index=0,
                local_index=getattr(a, "_id", [0])[0],
                rank=0,
                offset=offset,
                functionspace=v,
                argument=0 if arg == "test" else 1,
                arg=a,
            )
            for a, v in zip(base_scalars, V.basespaces, strict=False)
        )

    elif isinstance(V, VectorTensorProductSpace):
        b = V.system.base_vectors()
        for Vi in V:
            for v in Vi:
                functionspacedict[v.name] = v
        return VectorAdd.fromiter(
            sp.Mul.fromiter(
                get_BasisFunction(
                    v.fun_str,
                    global_index=i,
                    local_index=getattr(a, "_id", [0])[0],
                    rank=1,
                    offset=offset,
                    functionspace=v,
                    argument=0 if arg == "test" else 1,
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
        return f"{self.name}({self.c_names}; {self.functionspace.name})"

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
    argument: int
    functionspace: TestSpaceType

    def __new__(
        cls,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        coors = V.system
        obj: Self = Function.__new__(
            cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)])
        )
        obj.argument = 0
        if isinstance(V, DirectSum):
            obj.functionspace = V[0].get_homogeneous()
        elif isinstance(V, TensorProductSpace | DirectSumTPS):
            f = []
            vname = V.name
            for space in V.basespaces:
                if isinstance(space, DirectSum):
                    f.append(space[0].get_homogeneous())
                    vname += "0"
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
                        g.append(s[0].get_homogeneous())
                        vname += "0"
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

    argument: int
    functionspace: TrialSpaceType

    def __new__(
        cls,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        coors = V.system
        obj: Self = Function.__new__(
            cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)])
        )
        obj.functionspace = V
        obj.name = name if name is not None else "TrialFunction"
        obj.own_name = "TrialFunction"
        obj.argument = 1
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


class JAXArray(BaseFunction):
    """Wrapper for a raw JAX array tied to a function space.

    Primarily used as an intermediate symbolic handle that can forward
    (apply) basis transforms through functionspace.forward().
    """

    array: Array
    functionspace: FunctionSpaceType
    argument: int

    def __new__(
        cls,
        array: Array,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        obj: Self = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.argument = 3
        obj.name = name if name is not None else "JAXArray"
        return obj

    def forward(self):
        assert not isinstance(self.functionspace, VectorTensorProductSpace | DirectSum)
        return self.functionspace.forward(self.array)

    def doit(self, **hints: Any) -> Function | Array:
        if hints.get("deep", False):
            return self.array
        return self

    def __str__(self) -> str:
        return f"{self.name}({self.functionspace.name})"

    def _latex(self, printer: Any = None) -> str:
        return f"{latex_symbols[self.name]}({self.functionspace.name})"


class Jaxf(BaseFunction):
    r"""Symbolic twin of the JAXFunction in computational space.

    A JAXFunction represents a complete function in a given function space,
    backed by a JAX array of coefficients. That is, in 1D it represents the
    function u(x) defined by
    .. math::

        u(x) = \sum_i c_i \phi_i(x)

    where the coefficients :math:`c_i` are stored in the JAXFunction array, and
    :math:`\phi_i(x)` are the basis functions of the function space.
    The higher dimensional cases (tensor product spaces, vector-valued spaces)
    are handled similarly with different definitions of the basis functions.

    When assembling weak forms, the JAXFunction is expanded into a TrialFunction
    multiplied by the Jaxf symbolic coefficient array. The Jaxf object holds the
    reference to the coefficient array and function space, allowing for assembly
    of the system matrices and vectors.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jaxfun.galerkin import Chebyshev, inner
        >>> from jaxfun.galerkin.arguments import Jaxf, JAXFunction, TestFunction, \
            TrialFunction
        >>> V = Chebyshev.Chebyshev(4, name="V")
        >>> uf = JAXFunction(jnp.ones(V.dim), V)
        >>> v = TestFunction(V, name="v")
        >>> b = inner(v * uf)
        >>> assert jnp.all(b == jnp.array([3.1415927, 1.5707964, 1.5707964, 1.5707964]))
        >>> u = TrialFunction(V, name="u")
        >>> a = inner(u * v)
        >>> assert jnp.all(b == a @ uf.array)
        >>> uf.doit().__str__() == 'Jaxf(V)*T_j(x)'
        >>> assert isinstance(uf.doit().args[0], Jaxf)

    Args:
        array: JAX array of coefficients.
        V: Function space instance.
        name: Optional name for the Jaxf object.

    """

    array: Array
    functionspace: FunctionSpaceType
    name: str

    def __new__(
        cls,
        array: Array,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        obj: Self = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.name = name if name is not None else "Jaxf"
        return obj

    def backward(self):
        assert not isinstance(self.functionspace, VectorTensorProductSpace)
        return self.functionspace.backward(self.array)

    def doit(self, **hints: dict) -> Self:
        return self

    def __str__(self) -> str:
        return f"{self.name}({self.functionspace.name})"

    def _latex(self, printer: Any = None) -> str:
        return f"{latex_symbols[self.name]}({self.functionspace.name})"


class JAXFunction(ExpansionFunction):
    r"""A Galerkin function with explicit expansion coefficients.

    A JAXFunction represents a complete function in a given function space,
    backed by a JAX array of coefficients. That is, in 1D it represents the
    function u(x) defined by

    .. math::

        u(x) = \sum_i c_i \phi_i(x)

    where the coefficients :math:`c_i` are stored in the JAXFunction array, and
    :math:`\phi_i(x)` are the basis functions of the function space.

    The higher dimensional cases (tensor product spaces, vector-valued spaces)
    are handled similarly with different definitions of the basis functions.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jaxfun.galerkin import Chebyshev, inner
        >>> from jaxfun.galerkin.arguments import Jaxf, JAXFunction, TestFunction, \
            TrialFunction
        >>> V = Chebyshev.Chebyshev(4, name="V")
        >>> uf = JAXFunction(jnp.ones(V.dim), V)
        >>> v = TestFunction(V, name="v")
        >>> b = inner(v * uf)
        >>> assert jnp.all(b == jnp.array([3.1415927, 1.5707964, 1.5707964, 1.5707964]))
        >>> u = TrialFunction(V, name="u")
        >>> a = inner(u * v)
        >>> assert jnp.all(b == a @ uf.array)

    Args:
        array: JAX array of coefficients.
        V: Function space instance.
        name: Optional name for the JAXFunction object.
    """

    array: Array
    argument: int
    functionspace: FunctionSpaceType

    def __new__(
        cls,
        array: Array,
        V: FunctionSpaceType,
        name: str | None = None,
    ) -> Self:
        coors: CoordSys = V.system
        obj: Self = Function.__new__(
            cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)])
        )
        obj.array = array
        obj.functionspace = V
        obj.argument = 2
        obj.name = name if name is not None else "JAXFunction"
        obj.own_name = "JAXFunction"
        return obj

    def backward(self):
        assert not isinstance(self.functionspace, VectorTensorProductSpace)
        return self.functionspace.backward(self.array)

    def doit(self, **hints: Any) -> Expr:
        fs = self.functionspace
        trial = TrialFunction(fs).doit()
        offset = 1 if isinstance(fs, OrthogonalSpace | DirectSum) else fs.dims
        local_indices = slice(offset, 2 * offset)
        global_index = 0
        hat = f"\\hat{{{self.name}}}"
        rank = getattr(fs, "rank", 0)
        if rank == 0:
            name = "".join((hat, "_{", indices[local_indices], "}"))
            return Jaxf(self.array, fs, name=name) * trial
        assert rank == 1
        assert isinstance(fs, VectorTensorProductSpace)

        s = []
        for k, v in cast(Vector, trial).components.items():
            global_index = k._id[0]
            name = "".join(
                (hat, "_{", indices[local_indices], "}^{(", str(global_index), ")}")
            )
            s.append(
                Jaxf(self.array[global_index], fs[global_index], name=name) * k * v
            )
        return trial.func(*s)

    @jax.jit(static_argnums=0)
    def __matmul__(self, a: Array) -> Array:
        return self.array @ a

    @jax.jit(static_argnums=0)
    def __rmatmul__(self, a: Array) -> Array:
        return a @ self.array
