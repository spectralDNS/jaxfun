from typing import Any
import sympy as sp
import itertools
from sympy import Function, Symbol, Expr
from sympy.printing.pretty.stringpict import prettyForm
import jax.numpy as jnp
from jaxfun.Basespace import BaseSpace
from jaxfun.composite import DirectSum
from jaxfun.tensorproductspace import TensorProductSpace, VectorTensorProductSpace
from jaxfun.coordinates import CoordSys
from jaxfun.coordinates import latex_symbols

x, y, z = sp.symbols("x,y,z", real=True)

CartCoordSys = lambda name, t: CoordSys(name, sp.Lambda(t, t))

functionspacedict = {}

indices = 'ijklmn'

class BasisFunction(Function):
    def __init__(self, coordinate: Symbol, dummy: Symbol) -> None:
        f, s, offset, rank, j = dummy.name.split("-")
        self.global_index = int(j)
        self.local_index = coordinate._id[0]
        self.fun_str = s
        self.offset = int(offset)
        self.rank = int(rank)
        self.functionspace_name = f

    def __new__(cls, coordinate: Symbol, dummy: Symbol) -> Function:
        obj = Function.__new__(cls, coordinate, dummy)
        return obj

    def __str__(self) -> str:
        index = indices[self.local_index+self.offset]
        if self.rank == 0:
            return "".join((self.fun_str, "_", index, "(", self.args[0].name, ")"))
        elif self.rank == 1:
            return "".join((self.fun_str, "_", index, "^{(", str(self.global_index) ,")}", "(", self.args[0].name, ")")) 

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        index = indices[self.local_index+self.offset]
        if self.rank == 0:
            return "".join(
                (latex_symbols[self.fun_str], "_", str(index), "(", latex_symbols[self.args[0].name], ")")
            )
        elif self.rank == 1:
            return "".join(
                (latex_symbols[self.fun_str], "_", str(index),  "^{(", str(self.global_index) ,")}", "(", latex_symbols[self.args[0].name], ")")
            ) 

    @property
    def functionspace(self):
        return functionspacedict[self.functionspace_name]


class trial(BasisFunction):
    pass


class test(BasisFunction):
    pass


def _get_computational_function(
    arg: str, V: BaseSpace | TensorProductSpace | VectorTensorProductSpace
) -> Expr:
    func = test if arg == "test" else trial
    args = V.system.base_scalars()
    functionspacedict[V.name] = V
    offset = V.dims if arg == "trial" else 0
    if isinstance(V, BaseSpace):
        assert args[0].is_Symbol
        return func(args[0], sp.Symbol(V.name + "-" + V.fun_str+ "-" + str(offset) + "-0-0"))

    elif isinstance(V, TensorProductSpace):
        for space in V.spaces:
            functionspacedict[space.name] = space
        return sp.Mul(
            *[
                func(a, sp.Symbol(v.name + "-" + v.fun_str + "-" + str(offset) + "-0-0"))
                for a, v in zip(args, V)
            ]
        )

    elif isinstance(V, VectorTensorProductSpace):
        b = V.system.base_vectors()
        for i, Vi in enumerate(V):
            for v in Vi:
                functionspacedict[v.name] = v
        return sp.vector.VectorAdd(
            *[
                sp.Mul(
                    *[
                        func(a, sp.Symbol(v.name + "-" + v.fun_str + "-" + str(offset) + "-1-" + str(i)))
                        for a, v in zip(args, Vi)
                    ]
                )
                * b[i]
                for i, Vi in enumerate(V)
            ]
        )


# Note
# Need a unique Symbol in order to create a new TestFunction/TrialFunction for a new space.
# Without it all TestFunctions/TrialFunctions created with the same Cartesian coordinates
# will be the same object.

class TestFunction(Function):
    __test__ = False  # prevent pytest from considering this a test.

    def __init__(
        self,
        V: BaseSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> None:
        self.functionspace = V
        if isinstance(V, DirectSum):
            self.functionspace = V[0].get_homogeneous()
        elif isinstance(V, TensorProductSpace):
            f = []
            vname = V.name
            for space in V.spaces:
                if isinstance(space, DirectSum):
                    f.append(space[0].get_homogeneous())
                    vname += "0"
                else:
                    f.append(space)
            self.functionspace = TensorProductSpace(
                f, name=vname, system=V.system
            )
        self.name = name

    def __new__(
        cls,
        V: BaseSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Symbol(V.name)]))
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
        name = name if self.functionspace.rank == 0 else r"\mathbf{ {%s} }" % (name,)
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
        V: BaseSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> None:
        self.functionspace = V
        self.name = name

    def __new__(
        cls,
        V: BaseSpace | TensorProductSpace | VectorTensorProductSpace | DirectSum,
        name: str = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Symbol(V.name)]))
        return obj

    def doit(self, **hints: dict) -> Expr:
        if isinstance(self.functionspace, DirectSum):
            return sp.Add(
                *[
                    _get_computational_function("trial", f)
                    for f in self.functionspace.spaces
                ]
            )
        elif isinstance(self.functionspace, TensorProductSpace):
            spaces = self.functionspace.spaces
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
        name = name if self.functionspace.rank == 0 else r"\mathbf{ {%s} }" % (name,)
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
        obj = Function.__new__(cls, *(system._cartesian_xyz + [sp.Dummy()]))
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
            else str(self.args[:-1])
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def _latex(self, printer: Any = None) -> str:
        return (
            latex_symbols[self.name] + f"({self.args[0]})"
            if len(self.args) == 2
            else str(self.args[:-1])
        )


class VectorFunction(Function):
    def __new__(cls, name: str, system: CoordSys) -> Function:
        obj = Function.__new__(cls, *(system._cartesian_xyz + [sp.Dummy()]))
        obj.system = system
        obj.name = name.lower()
        return obj

    def __str__(self) -> str:
        return self.name + str(self.args[:-1])

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
        return r"\mathbf{{%s}}" % (latex_symbols[self.name],) + str(self.args[:-1])
