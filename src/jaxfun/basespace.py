from functools import partial

import jax
import sympy as sp
from jax import Array

from jaxfun.coordinates import CartCoordSys, CoordSys, x

n = sp.Symbol("n", integer=True, positive=True)  # index


class BaseSpace:
    is_transient = False

    def __init__(
        self,
        system: CoordSys = None,
        name: str = "BaseSpace",
        fun_str: str = "phi",
    ) -> None:
        self.name = name
        self.fun_str = fun_str
        self.system = CartCoordSys("N", (x,)) if system is None else system

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        raise RuntimeError
