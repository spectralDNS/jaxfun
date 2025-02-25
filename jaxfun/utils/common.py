from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
from sympy import Expr, Symbol

n = Symbol("n", positive=True, integer=True)


__all__ = (
    "diff",
    "diffx",
    "jacn",
    "matmat",
    "tosparse",
    "fromdense",
    "lambdify",
)


def ulp(x: float) -> float:
    return jnp.nextafter(x, x + 1) - x


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


@jax.jit
def matmat(a: Array | BCOO, b: Array | BCOO) -> Array | BCOO:
    return a @ b


@partial(jax.jit, static_argnums=1)
def eliminate_near_zeros(a: Array, tol: int = 100) -> Array:
    atol: float = ulp(jnp.abs(a).max()) * tol
    return jnp.where(jnp.abs(a) < atol, jnp.zeros(a.shape), a)


def fromdense(a: Array, tol: int = 100) -> sparse.BCOO:
    a0: Array = eliminate_near_zeros(a, tol=tol)
    return sparse.BCOO.fromdense(a0)


def tosparse(a: Array, tol: int = 100) -> sparse.BCOO:
    a0: Array = eliminate_near_zeros(a, tol=tol)
    return sparse.BCOO.from_scipy_sparse(scipy_sparse.csr_matrix(a0))


def lambdify(
    args: tuple[Symbol],
    expr: Expr,
    modules: list[str] = None,
    printer: Any = None,
    use_imps: bool = True,
    dummify: bool = False,
    cse: bool = False,
    doctring_limit: int = 1000,
) -> Callable[[Iterable[Array]], Array]:
    system = expr.free_symbols.pop()._system
    expr = system.expr_base_scalar_to_psi(expr)
    args = system.expr_base_scalar_to_psi(args)
    return sp.lambdify(
        args,
        expr,
        modules=modules,
        printer=printer,
        use_imps=use_imps,
        dummify=dummify,
        cse=cse,
        docstring_limit=doctring_limit,
    )
