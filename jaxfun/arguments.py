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

from jaxfun.Basespace import BaseSpace, OrthogonalSpace
from jaxfun.composite import DirectSum
from jaxfun.coordinates import BaseScalar, CoordSys, latex_symbols
from jaxfun.tensorproductspace import TensorProductSpace, VectorTensorProductSpace

x, y, z = sp.symbols("x,y,z", real=True)

CartCoordSys = lambda name, t: CoordSys(name, sp.Lambda(t, t))

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
    def __new__(cls, name: str, system: CoordSys) -> Function:
        obj = Function.__new__(cls, *(list(system._cartesian_xyz) + [sp.Dummy()]))
        obj.system = system
        obj.name = name.lower()
        return obj

    def __str__(self) -> str:
        return "\033[1m%s\033[0m" % (self.name,) + str(self.args[:-1])

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


# Not sure this will be useful:
class JAXArray(Function):
    def __new__(
        cls,
        array: Array,
        V: OrthogonalSpace | TensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        obj = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.name = name if name is not None else "JAXArray"
        return obj

    def forward(self):
        return self.functionspace.forward(self.array)

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


class Jaxf(Function):
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


class Constant(sp.Symbol):
    def __new__(cls, name: str, val: Number, **assumptions):
        obj = super().__new__(cls, name, **assumptions)
        obj.val = val
        return obj

    def doit(self) -> Number:
        return self.val


class Identity(sp.Expr):
    def __init__(self, sys: CoordSys):
        self.sys = sys

    def doit(self):
        return sum(
            self.sys.base_dyadics()[:: self.sys.dims + 1], sp.vector.DyadicZero()
        )
