import itertools
from functools import partial
from numbers import Number
from typing import Any

import jax
import sympy as sp
from jax import Array
from sympy import Expr, Function, Symbol
from sympy.printing import latex
from sympy.printing.pretty.stringpict import prettyForm

from jaxfun.Basespace import OrthogonalSpace
from jaxfun.composite import DirectSum
from jaxfun.coordinates import BaseScalar, CoordSys, latex_symbols
from jaxfun.tensorproductspace import TensorProductSpace, VectorTensorProductSpace

x, y, z = sp.symbols("x,y,z", real=True)

CartCoordSys = lambda name, t: CoordSys(name, sp.Lambda(t, t))

functionspacedict = {}

indices = "ijklmn"


class BasisFunction(Function):
    def __init__(self, coordinate: Symbol, dummy: Symbol) -> None:
        f, s, offset, rank, j = dummy.name.split("-")
        self.global_index = int(j)
        self.local_index = getattr(coordinate, "_id", [0])[0]
        self.fun_str = s
        self.offset = int(offset)
        self.rank = int(rank)
        self.functionspace_name = f

    def __new__(cls, coordinate: Symbol, dummy: Symbol) -> Function:
        obj = Function.__new__(cls, coordinate, dummy)
        return obj

    def __str__(self) -> str:
        index = indices[self.local_index + self.offset]
        if self.rank == 0:
            return "".join((self.fun_str, "_", index, "(", self.args[0].name, ")"))
        elif self.rank == 1:
            return "".join(
                (
                    self.fun_str,
                    "_",
                    index,
                    "^{(",
                    str(self.global_index),
                    ")}",
                    "(",
                    self.args[0].name,
                    ")",
                )
            )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None, exp: Number = None) -> str:
        index = indices[self.local_index + self.offset]
        if self.rank == 0:
            s = "".join(
                (
                    latex_symbols[self.fun_str],
                    "_",
                    str(index),
                    "(",
                    latex_symbols[self.args[0].name],
                    ")",
                )
            )
        elif self.rank == 1:
            s = "".join(
                (
                    latex_symbols[self.fun_str],
                    "_",
                    str(index),
                    "^{(",
                    str(self.global_index),
                    ")}",
                    "(",
                    latex_symbols[self.args[0].name],
                    ")",
                )
            )
        return s if exp is None else f"\\left({s}\\right)^{{{exp}}}"

    @property
    def functionspace(self):
        return functionspacedict[self.functionspace_name]


class FlaxBasisFunction(Function):
    def __init__(self, *args) -> None:
        from jaxfun.pinns.module import moduledict
        
        coordinates = args[:-1]
        dummy = args[-1]
        global_index, functionspace_name, rank_parent, name = dummy.name.split("+")
        self.name = name
        self._latex_form = latex_symbols[name]
        self.functionspace_name = functionspace_name
        self._module = moduledict[functionspace_name]
        if int(rank_parent) > 0:
            name, base_vector_name = name.split("_")
            self._latex_form = name + f"_{{{latex_symbols[base_vector_name]}}}"
        self.global_index = int(global_index)
        self.base_scalars = coordinates

    def __new__(cls, *coordinates) -> Function:
        obj = Function.__new__(cls, *coordinates)
        return obj
    
    @property
    def module(self):
        return self._module
    
    @property
    def functionspace(self):
        return functionspacedict[self.functionspace_name]

    def __str__(self) -> str:
        if len(self.args[1:]) == 1:
            return "".join((self.name, "(", str(self.args[0]), ")"))
        else:
            return "".join((self.name, str(self.args[:-1])))

    def doit(self, **hints: dict) -> Function:
        return self

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None, exp: Number = None) -> str:
        form = self._latex_form if exp is None else self._latex_form + f"^{{{exp}}}"
        if len(self.args[1:]) == 1:
            return "".join((form, "(", latex_symbols[self.args[0].name], ")"))
        else:
            return "".join((form, latex(self.args[:-1])))


class trial(BasisFunction):
    pass


class test(BasisFunction):
    pass


def _get_computational_function(
    arg: str, V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace
) -> Expr:
    func = test if arg == "test" else trial
    args = V.system.base_scalars()
    functionspacedict[V.name] = V
    offset = V.dims if arg == "trial" else 0
    if isinstance(V, OrthogonalSpace):
        assert args[0].is_Symbol
        return func(
            args[0], sp.Symbol(V.name + "-" + V.fun_str + "-" + str(offset) + "-0-0")
        )

    elif isinstance(V, TensorProductSpace):
        for space in V.basespaces:
            functionspacedict[space.name] = space
        return sp.Mul(
            *[
                func(
                    a, sp.Symbol(v.name + "-" + v.fun_str + "-" + str(offset) + "-0-0")
                )
                for a, v in zip(args, V, strict=False)
            ]
        )

    elif isinstance(V, VectorTensorProductSpace):
        b = V.system.base_vectors()
        for Vi in V:
            for v in Vi:
                functionspacedict[v.name] = v
        return sp.vector.VectorAdd(
            *[
                sp.Mul(
                    *[
                        func(
                            a,
                            sp.Symbol(
                                v.name
                                + "-"
                                + v.fun_str
                                + "-"
                                + str(offset)
                                + "-1-"
                                + str(i)
                            ),
                        )
                        for a, v in zip(args, Vi, strict=False)
                    ]
                )
                * b[i]
                for i, Vi in enumerate(V)
            ]
        )


# NOTE
# Need a unique Symbol in order to create a new TestFunction/TrialFunction for a new
# space. Without it all TestFunctions/TrialFunctions created with the same Cartesian
# coordinates will be the same object.


class TestFunction(Function):
    __test__ = False  # prevent pytest from considering this a test.

    def __init__(
        self,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> None:
        self.functionspace = V
        if isinstance(V, DirectSum):
            self.functionspace = V[0].get_homogeneous()
        elif isinstance(V, TensorProductSpace):
            f = []
            vname = V.name
            for space in V.basespaces:
                if isinstance(space, DirectSum):
                    f.append(space[0].get_homogeneous())
                    vname += "0"
                else:
                    f.append(space)
            self.functionspace = TensorProductSpace(f, name=vname, system=V.system)
        self.name = name

    def __new__(
        cls,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        return obj

    def doit(self, **hints: dict) -> Expr:
        return _get_computational_function("test", self.functionspace)

    def __str__(self) -> str:
        name = self.name if self.name is not None else "TestFunction"
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

    def _latex(self, printer: Any = None) -> str:
        name = self.name if self.name is not None else "TestFunction"
        name = name if self.functionspace.rank == 0 else r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
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
    def __init__(
        self,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> None:
        self.functionspace = V
        self.name = name

    def __new__(
        cls,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        return obj

    def doit(self, **hints: dict) -> Expr:
        if isinstance(self.functionspace, DirectSum):
            return sp.Add(
                *[
                    _get_computational_function("trial", f)
                    for f in self.functionspace.basespaces
                ]
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
            return sp.Add(
                *[
                    _get_computational_function(
                        "trial",
                        TensorProductSpace(
                            s,
                            name=self.functionspace.name + f"{i}",
                            system=self.functionspace.system,
                        ),
                    )
                    for i, s in enumerate(tensorspaces)
                ]
            )

        return _get_computational_function("trial", self.functionspace)

    def __str__(self) -> str:
        name = self.name if self.name is not None else "TrialFunction"
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

    def _latex(self, printer: Any = None) -> str:
        name = self.name if self.name is not None else "TrialFunction"
        name = name if self.functionspace.rank == 0 else r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
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
        return sp.vector.VectorAdd(
            *[
                Function(self.name.upper() + f"_{latex_symbols[vn[i]]}")(
                    *self.system.base_scalars()
                )
                * bi
                for i, bi in enumerate(self.system.base_vectors())
            ]
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
        name: str,
    ) -> Function:
        obj = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.name = name.lower()
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
        name: str,
    ) -> Function:
        obj = Function.__new__(cls, sp.Dummy())
        obj.array = array
        obj.functionspace = V
        obj.name = name.lower()
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
    def __init__(
        self,
        array: Array,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = "JAXFunction",
    ) -> None:
        self.array = array
        self.functionspace = V
        self.name = name

    def __new__(
        cls,
        array: Array,
        V: OrthogonalSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        return obj

    def backward(self):
        return self.functionspace.backward(self.array)

    def doit(self, **hints: dict) -> Expr:
        return (
            Jaxf(self.array, self.functionspace, self.name)
            * TrialFunction(self.functionspace).doit()
        )

    def __str__(self) -> str:
        name = self.name if self.name is not None else "JAXFunction"
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

    def _latex(self, printer: Any = None) -> str:
        name = self.name if self.name is not None else "JAXFunction"
        name = name if self.functionspace.rank == 0 else r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
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
