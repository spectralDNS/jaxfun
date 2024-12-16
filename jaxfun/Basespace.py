from typing import NamedTuple
from functools import partial
import jax 
from jax import Array
from jaxfun.utils.common import Domain, jacn


class BaseSpace:

    def __init__(
        self,
        N: int,
        domain: NamedTuple = Domain(-1, 1),
    ):
        self.N = N
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
