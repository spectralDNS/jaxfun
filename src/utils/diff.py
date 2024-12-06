from typing import Tuple
import jax 
from jax import Array 

def diff(fun : callable, k : int = 1):
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun, in_axes=(0, None)))

def eval(fun : callable, x : float, c : Array, axes : Tuple[int] = (0,)) -> Array:
    dim : int = len(c.shape)
    for ax in axes:
        axi : int = dim-1-ax
        c = jax.vmap(fun, in_axes=(None, axi), out_axes=axi)(x, c)
    return c
