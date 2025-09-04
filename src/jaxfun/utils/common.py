from collections.abc import Callable, Iterable
from functools import partial, wraps
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
from scipy.special import sph_harm
from sympy import Expr, Number, Symbol

Ynm = lambda n, m, x, y: sph_harm(m, n, y, x)
n = Symbol("n", positive=True, integer=True)


__all__ = (
    "diff",
    "diffx",
    "Domain",
    "jacn",
    "matmat",
    "tosparse",
    "fromdense",
    "lambdify",
)


def jit_vmap(
    in_axes: int | None | tuple[Any] = (None, 0),
    out_axes: Any = 0,
    static_argnums: int | tuple[int] | None = 0,
):
    """Decorator that JIT compiles a function and applies vmap if the first argument is
    an array.

    Args:
        in_axes (optional): An integer, None, or sequence of values specifying which
            input array axes to map over. Defaults to (None, 0).
        out_axes (optional): Standard Python container (tuple/list/dict) thereof
            indicating where the mapped axis should appear in the output. Defaults to 0.
        static_argnums (optional): optional, an int or
            collection of ints that specify which positional arguments to treat as
            static (trace- and compile-time constant). Defaults to 0.
    """

    def wrap(func):
        @partial(jax.jit, static_argnums=static_argnums)
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if args[0].shape == ():
                return func(self, *args, **kwargs)
            return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)(
                self, *args, **kwargs
            )

        return wrapper

    return wrap


class Domain(NamedTuple):
    lower: Number
    upper: Number


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
        fun = jax.jacfwd(fun)  # if i % 2 else jax.jacrev(fun)
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
    # system = get_system(expr)
    # expr = system.expr_base_scalar_to_psi(expr)
    # args = system.expr_base_scalar_to_psi(args)
    modules_default = ["jax", {"Ynm": Ynm}]
    modules = modules_default if modules is None else [modules] + modules_default
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
