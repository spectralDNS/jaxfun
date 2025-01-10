from typing import Any
import sympy as sp
from sympy import Function, Symbol, Expr
from sympy.printing.pretty.stringpict import prettyForm
from jaxfun.Basespace import BaseSpace
from jaxfun.tensorproductspace import TensorProductSpace, VectorTensorProductSpace
from jaxfun.coordinates import CoordSys

x, y, z = sp.symbols("x,y,z", real=True)

CartCoordSys1D = CoordSys("N", sp.Lambda((x,), (x,)))
CartCoordSys2D = CoordSys("N", sp.Lambda((x, y), (x, y)))
CartCoordSys3D = CoordSys("N", sp.Lambda((x, y, z), (x, y, z)))
CartCoordSys = {1: CartCoordSys1D, 2: CartCoordSys2D, 3: CartCoordSys3D}


class BasisFunction(Function):
    def __init__(self, coordinate: Symbol, dummy: Symbol) -> None:
        f, s, j = dummy.name.split("_")
        self.global_index = int(j)
        self.local_index = coordinate._id[0]
        self.fun_str = s
        self.functionspace_name = f

    def __new__(cls, coordinate: Symbol, dummy: Symbol) -> Function:
        obj = Function.__new__(cls, coordinate, dummy)
        return obj

    def __str__(self) -> str:
        return "".join((self.fun_str, "(", self.args[0].name, ")"))

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()


class trial(BasisFunction):
    pass


class test(BasisFunction):
    pass


def _get_computational_function(
    arg: str, V: BaseSpace | TensorProductSpace | VectorTensorProductSpace
) -> Expr:
    func = test if arg == "test" else trial
    args = V.system.base_scalars()
    if isinstance(V, BaseSpace):
        assert args[0].is_Symbol
        return func(args[0], sp.Symbol(V.name + "_" + V.fun_str + "_0"))

    elif isinstance(V, TensorProductSpace):
        return sp.Mul(
            *[
                func(a, sp.Symbol(v.name + "_" + v.fun_str + "_0"))
                for a, v in zip(args, V)
            ]
        )

    elif isinstance(V, VectorTensorProductSpace):
        b = V.system.base_vectors()
        return sp.vector.VectorAdd(
            *[
                sp.Mul(
                    *[
                        func(a, sp.Symbol(v.name + "_" + v.fun_str + "_" + str(i)))
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
    def __init__(self, V: BaseSpace | TensorProductSpace | VectorTensorProductSpace) -> None:
        self.functionspace = V

    def __new__(cls, V: BaseSpace | TensorProductSpace | VectorTensorProductSpace) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Symbol(V.name)]))
        return obj

    def doit(self, **hints: dict) -> Expr:
        return _get_computational_function("test", self.functionspace)

    def __str__(self) -> str:
        return "".join(
            (
                "TestFunction(",
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
    def __init__(self, V: BaseSpace | TensorProductSpace | VectorTensorProductSpace) -> None:
        self.functionspace = V

    def __new__(cls, V: BaseSpace | TensorProductSpace | VectorTensorProductSpace) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Symbol(V.name)]))
        return obj

    def doit(self, **hints: dict) -> Expr:
        return _get_computational_function("trial", self.functionspace)

    def __str__(self) -> str:
        return "".join(
            (
                "TrialFunction(",
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


class SpatialFunction(Function):
    def __new__(cls, name: str, system: CoordSys) -> Function:
        obj = Function.__new__(cls, *(system._cartesian_xyz))
        obj.system = system
        obj.name = name.lower()
        return obj

    def __str__(self) -> str:
        return self.name + str(tuple(self.args))

    def doit(self, **hints: dict) -> Function:
        """Return function in computational domain"""
        return Function(self.name.upper())(*(self.system.base_scalars()))

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()
