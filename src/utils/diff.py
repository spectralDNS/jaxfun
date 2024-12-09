from collections.abc import Callable
from typing import Tuple

import jax
from jax import Array


def diff(
    fun: Callable[[float, Array], Array], k: int = 1
) -> Callable[[Array, Array], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun, in_axes=(0, None)))

def diffx(
    fun: Callable[[float, int], Array], k: int = 1
) -> Callable[[float, int], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return fun


def eval_fun(
    fun: Callable[[float, Array], Array],
    x: float,
    c: Array,
    axes: Tuple[int] = (0,),
) -> Array:
    dim: int = len(c.shape)
    for ax in axes:
        axi: int = dim - 1 - ax
        c = jax.vmap(fun, in_axes=(None, axi), out_axes=axi)(x, c)
    return c
