from typing import Callable, Union
from functools import partial
import sympy as sp
import jax
from jax import Array
import jax.numpy as jnp
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse

n = sp.Symbol("n", positive=True, integer=True)


__all__ = ("diff", "diffx", "jacn", "evaluate", "matmat", "to_sparse", "from_dense")


def ulp(x: float):
    return jnp.nextafter(x, x+1) - x


def diff(
    fun: Callable[[float, Array], float], k: int = 1
) -> Callable[[Array, Array], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun, in_axes=(0, None)))


def diffx(
    fun: Callable[[float, int], float], k: int = 1
) -> Callable[[Array, int], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.vmap(fun, in_axes=(0, None))


def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    for _ in range(k):
        fun = jax.jacfwd(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0)


@partial(jax.jit, static_argnums=(0, 3))
def evaluate(
    fun: Callable[[float, Array], Array],
    x: tuple[float],
    c: Array,
    axes: tuple[int] = (0,),
) -> Array:
    assert len(x) == len(axes)
    dim: int = len(c.shape)
    for xi, ax in zip(x, axes):
        axi: int = dim - 1 - ax
        c = jax.vmap(fun, in_axes=(None, axi), out_axes=axi)(xi, c)
    return c


@jax.jit
def matmat(a: Union[Array, BCOO], b: Union[Array, BCOO]) -> Union[Array, BCOO]:
    return a @ b


def to_sparse(a: Array, sparse_tol: int = 100) -> sparse.BCOO:
    tol: float = ulp(jnp.max(jnp.abs(a))) * sparse_tol
    a = jnp.choose(jnp.array(jnp.abs(a) > tol, dtype=int), (jnp.zeros_like(a), a))
    return sparse.BCOO.fromdense(a)


def from_dense(a: Array, sparse_tol: int = 100) -> sparse.BCOO:
    tol: float = ulp(jnp.max(jnp.abs(a))) * sparse_tol
    z = jnp.where(jnp.abs(a) < tol, jnp.zeros(a.shape), a)
    return sparse.BCOO.from_scipy_sparse(scipy_sparse.csr_matrix(z))
