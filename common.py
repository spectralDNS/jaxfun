from typing import Callable
import jax 
from jax import Array 
from collections.abc import MutableMapping

def diff(fun : Callable[[float, Array], float], k : int = 1):
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun, in_axes=(0, None)))

def diffx(fun : Callable[[float], float], k : int = 1):
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun))

