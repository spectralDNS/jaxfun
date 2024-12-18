from typing import NamedTuple
from functools import partial
from numbers import Number
import copy
import sympy as sp
import jax
from jax import Array
import jax.numpy as jnp
from jaxfun.utils.common import jacn

n = sp.Symbol("n", integer=True, positive=True)


class Domain(NamedTuple):
    lower: Number
    upper: Number


class BoundaryConditions(dict):
    """Boundary conditions as a dictionary"""

    def __init__(self, bc: dict, domain: Domain = None):
        bcs = {"left": {}, "right": {}}
        bcs.update(copy.deepcopy(bc))
        dict.__init__(self, bcs)

    def orderednames(self) -> list[str]:
        return ["L" + bci for bci in sorted(self["left"].keys())] + [
            "R" + bci for bci in sorted(self["right"].keys())
        ]

    def orderedvals(self) -> list[Number]:
        ls = []
        for lr in ("left", "right"):
            for key in sorted(self[lr].keys()):
                val = self[lr][key]
                ls.append(val[1] if isinstance(val, (tuple, list)) else val)
        return ls

    def num_bcs(self) -> int:
        return len(self.orderedvals())


class BaseSpace:
    def __init__(
        self,
        N: int,
        domain: Domain = Domain(-1, 1),
    ):
        self.N = N
        self.domain = domain
        self.bcs = None
        self.stencil = None

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        raise NotImplementedError

    def quad_points_and_weights(self, N: int = 0) -> Array:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_basis_derivative(self, x: Array, k: int = 0) -> Array:
        return jacn(self.eval_basis_functions, k)(x)

    @partial(jax.jit, static_argnums=0)
    def vandermonde(self, x: Array) -> Array:
        return self.evaluate_basis_derivative(x, 0)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        raise NotImplementedError
    