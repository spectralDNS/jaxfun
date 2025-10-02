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
from functools import partial
from numbers import Number
from typing import Any

import jax
import sympy as sp
from jax import Array
from sympy import Expr, Function
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import VectorAdd

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, CoordSys, latex_symbols

from .composite import DirectSum
from .orthogonal import OrthogonalSpace
from .tensorproductspace import (
    TensorProductSpace,
    VectorTensorProductSpace,
)

t, x, y, z = sp.symbols("t,x,y,z", real=True)

functionspacedict = {}

indices = "ijklmn"


def get_BasisFunction(
    name,
    *,
    global_index: int,
    local_index: int,
    rank: int,
    offset: int,
    functionspace: BaseSpace,
    argument: int,
    arg: BaseScalar,
):
    """Create a symbolic basis function with enriched printing.

    Adds (monkey‑patches) __str__, _pretty and _latex methods so tensor
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
        if cls.rank == 0:
            return "".join((cls.name, "_", index, "(", cls.args[0].name, ")"))
        elif cls.rank == 1:
            return "".join(
                (
                    cls.name,
                    "_",
                    index,
                    "^{(",
                    str(cls.global_index),
                    ")}",
                    "(",
                    cls.args[0].name,
                    ")",
                )
            )

    def _pretty(cls, printer: Any = None) -> str:
        return prettyForm(cls.__str__())

    def _sympystr(cls, printer: Any) -> str:
        return cls.__str__()

    def _latex(cls, printer: Any = None, exp: Number = None) -> str:
        index = indices[cls.local_index + cls.offset]
        if cls.rank == 0:
            s = "".join(
                (
                    latex_symbols[cls.name],
                    "_",
                    str(index),
                    "(",
                    latex_symbols[cls.args[0].name],
                    ")",
                )
            )
        elif cls.rank == 1:
            s = "".join(
                (
                    latex_symbols[cls.name],
                    "_",
                    str(index),
                    "^{(",
                    str(cls.global_index),
                    ")}",
                    "(",
                    latex_symbols[cls.args[0].name],
                    ")",
                )
            )
        return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"

    b = sp.Function(
        name,
        global_index=global_index,
        local_index=local_index,
        rank=rank,
        offset=offset,
        functionspace=functionspace,
        argument=argument,
    )(arg)

    b.__class__.__str__ = __str__
    b.__class__._pretty = _pretty
    b.__class__._sympystr = _sympystr
    b.__class__._latex = _latex
    return b


def _get_computational_function(
    arg: str, V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace
) -> Expr:
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
            for a, v in zip(base_scalars, V, strict=False)
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


# NOTE
# Need a unique Symbol in order to create a new TestFunction/TrialFunction for a new
# space. Without it all TestFunctions/TrialFunctions created with the same Cartesian
# coordinates will be the same object.


class TestFunction(Function):
    """Symbolic test function T(x; V) for weak form assembly.

    Holds a reference to its functionspace and lazily expands (via doit)
    into computational (tensor product) factor basis functions.
    """

    __test__ = False  # prevent pytest from considering this a test.

    def __new__(
        cls,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        obj.functionspace = V
        obj.argument = 0
        if isinstance(V, DirectSum):
            obj.functionspace = V[0].get_homogeneous()
        elif isinstance(V, TensorProductSpace):
            f = []
            vname = V.name
            for space in V.basespaces:
                if isinstance(space, DirectSum):
                    f.append(space[0].get_homogeneous())
                    vname += "0"
                else:
                    f.append(space)
            obj.functionspace = TensorProductSpace(f, name=vname, system=V.system)
        obj.name = name if name is not None else "TestFunction"
        return obj

    def doit(self, **hints: dict) -> Expr:
        return _get_computational_function("test", self.functionspace)

    def __str__(self) -> str:
        return "".join(
            (
                self.name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _latex(self, printer: Any = None) -> str:
        name = self.name
        if name != "TestFunction" and self.functionspace.rank == 1:
            name = r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()


class TrialFunction(Function):
    """Symbolic trial function U(x; V) for weak form assembly.

    Direct sums expand to sum of component spaces. Tensor product spaces
    with direct-sum factors expand into additive combinations of products.
    """

    def __new__(
        cls,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        obj.functionspace = V
        obj.name = name if name is not None else "TrialFunction"
        obj.argument = 1
        return obj

    def doit(self, **hints: dict) -> Expr:
        if isinstance(self.functionspace, DirectSum):
            return sp.Add.fromiter(
                _get_computational_function("trial", f)
                for f in self.functionspace.basespaces
            )
        elif isinstance(self.functionspace, TensorProductSpace):
            spaces = self.functionspace.basespaces
            f = []
            for space in spaces:
                if isinstance(space, DirectSum):
                    f.append(space)
                else:
                    f.append([space])
            tensorspaces = itertools.product(*f)
            return sp.Add.fromiter(
                _get_computational_function(
                    "trial",
                    TensorProductSpace(
                        s,
                        name=self.functionspace.name + f"{i}",
                        system=self.functionspace.system,
                    ),
                )
                for i, s in enumerate(tensorspaces)
            )

        return _get_computational_function("trial", self.functionspace)

    def __str__(self) -> str:
        return "".join(
            (
                self.name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _latex(self, printer: Any = None) -> str:
        name = self.name
        if name != "TrialFunction" and self.functionspace.rank == 1:
            name = r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()


class ScalarFunction(Function):
    """Physical-domain scalar field placeholder u(x) with mapping support.

    Calling .doit() returns the computational-domain representation
    U(X) by replacing Cartesian symbols with base scalars.
    """

    def __new__(cls, name: str, system: CoordSys) -> Function:
        obj = Function.__new__(cls, *(list(system._cartesian_xyz) + [sp.Dummy()]))
        obj.system = system
        obj.name = name.lower()
        return obj

    def doit(self, **hints: dict) -> Function:
        """Return function in computational domain"""
        return Function(self.name.upper())(*self.system.base_scalars())

    def __str__(self) -> str:
        return (
            self.name + f"({self.args[0]})"
            if len(self.args) == 2
            else self.name + str(self.args[:-1])
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        return (
            latex_symbols[self.name] + f"({self.args[0]})"
            if len(self.args) == 2
            else self.name + str(self.args[:-1])
        )


class VectorFunction(Function):
    """Physical-domain vector field placeholder v(x).

    The .doit() method builds a VectorAdd of component scalar Functions
    times basis vectors in the computational domain.
    """

    def __new__(cls, name: str, system: CoordSys) -> Function:
        obj = Function.__new__(cls, *(list(system._cartesian_xyz) + [sp.Dummy()]))
        obj.system = system
        obj.name = name.lower()
        return obj

    def __str__(self) -> str:
        return "\033[1m%s\033[0m" % (self.name,) + str(self.args[:-1])  # noqa: UP031

    def doit(self, **hints: dict) -> Function:
        """Return function in computational domain"""
        vn = self.system._variable_names
        return VectorAdd.fromiter(
            Function(self.name.upper() + f"_{latex_symbols[vn[i]]}")(
                *self.system.base_scalars()
            )
            * bi
            for i, bi in enumerate(self.system.base_vectors())
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        return r"\mathbf{{%s}}" % (latex_symbols[self.name],) + str(self.args[:-1])  # noqa: UP031


class JAXArray(Function):
    """Wrapper for a raw JAX array tied to a function space.

    Primarily used as an intermediate symbolic handle that can forward
    (apply) basis transforms through functionspace.forward().
    """

    def __new__(
        cls,
        array: Array,
        V: OrthogonalSpace | TensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        obj = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.argument = 3
        obj.name = name if name is not None else "JAXArray"
        return obj

    def forward(self):
        return self.functionspace.forward(self.array)

    def doit(self, **hints: dict) -> Function:
        if hints.get("deep", False):
            return self.array
        return self

    def __str__(self) -> str:
        return self.name + f"({self.functionspace.name})"

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        return latex_symbols[self.name] + f"({self.functionspace.name})"


class Jaxf(Function):
    """Symbolic wrapper for a JAX array interpreted in backward transform.

    Used when converting coefficient arrays back to physical space via
    functionspace.backward().
    """

    def __new__(
        cls,
        array: Array,
        V: OrthogonalSpace | TensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        obj = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.name = name if name is not None else "Jaxf"
        return obj

    def backward(self):
        return self.functionspace.backward(self.array)

    def doit(self, **hints: dict) -> Function:
        return self

    def __str__(self) -> str:
        return self.name + f"({self.functionspace.name})"

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        return latex_symbols[self.name] + f"({self.functionspace.name})"


class JAXFunction(Function):
    """Symbolic + numeric hybrid representing coefficients in a space.

    Represents a function on a given space with a JAX array of expansion coefficients.

    Behaves like a symbolic trial expansion; .doit() returns the algebraic
    product of coefficients (Jaxf) and test basis (TrialFunction).
    """

    def __new__(
        cls,
        array: Array,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        obj.array = array
        obj.functionspace = V
        obj.argument = 2
        obj.name = name if name is not None else "JAXFunction"
        return obj

    def backward(self):
        return self.functionspace.backward(self.array)

    def doit(self, **hints: dict) -> Expr:
        return (
            Jaxf(self.array, self.functionspace, self.name)
            * TrialFunction(self.functionspace).doit()
        )

    def __str__(self) -> str:
        return "".join(
            (
                self.name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _latex(self, printer: Any = None) -> str:
        name = self.name
        if name != "JAXFunction" and self.functionspace.rank == 1:
            name = r"\mathbf{ {%s} }" % (self.name,)  # noqa: UP031
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    @partial(jax.jit, static_argnums=0)
    def __matmul__(self, a: Array) -> Array:
        return self.array @ a

    @partial(jax.jit, static_argnums=0)
    def __rmatmul__(self, a: Array) -> Array:
        return a @ self.array
