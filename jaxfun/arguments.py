import sympy as sp
from sympy import Function

x, y, z = sp.symbols("x,y,z", real=True)


class BasisFunction(Function):
    def __init__(self, coordinate, space=None):
        self.functionspace = space

    def __new__(self, coordinate, space=None):
        obj = Function.__new__(self, coordinate)
        obj.functionspace = space
        return obj


class TrialFunction(BasisFunction):
    pass


class TestFunction(BasisFunction):
    pass
