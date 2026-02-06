from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, overload

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
from scipy.special import sph_harm_y
from sympy import Expr, Symbol

from jaxfun.typing import FloatLike

if TYPE_CHECKING:
    from jaxfun.coordinates import BaseScalar

Ynm = lambda n, m, x, y: sph_harm_y(n, m, x, y)
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
    in_axes: int | None | tuple[int | None, ...] = 0,
    out_axes: Any = 0,
    static_argnums: int | tuple[int, ...] | None = 0,
    ndim: int = 0,
):
    """Decorator that JIT compiles a function and applies vmap if the first argument is
    an array with dimensions > ndim. If the first argument is a scalar, or an array of
    dimensions = ndim, then the function is merely jitted.

    The decorator can only be used with class methods.

    Args:
        in_axes (optional): An integer, None, or sequence of values specifying which
            input array axes to map over. Defaults to (None, 0).
        out_axes (optional): Standard Python container (tuple/list/dict) thereof
            indicating where the mapped axis should appear in the output. Defaults to 0.
        static_argnums (optional): optional, an int or
            collection of ints that specify which positional arguments to treat as
            static (trace- and compile-time constant). Defaults to 0.
        ndim (optional): Number of dimensions of the first argument that should not
            trigger vmap. Defaults to 0 (scalar).
    """
    in_axes = (None,) + in_axes if isinstance(in_axes, tuple) else (None, in_axes)

    def wrap(func):
        @jax.jit(static_argnums=static_argnums)
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if args[0].ndim == ndim:
                return func(self, *args, **kwargs)
            return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)(
                self, *args, **kwargs
            )

        return wrapper

    return wrap


class Domain(NamedTuple):
    lower: FloatLike
    upper: FloatLike


def ulp(x: float | Array) -> Array:
    return jnp.nextafter(x, x + 1) - x


def diff(
    fun: Callable[[Array, Any], Array], k: int = 1
) -> Callable[[Array, Any], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.jit(jax.vmap(fun, in_axes=(0, None)))


def diffx(
    fun: Callable[[Array, Any], Array], k: int = 1
) -> Callable[[Array, Any], Array]:
    for _ in range(k):
        fun = jax.grad(fun)
    return jax.vmap(fun, in_axes=(0, None))


def jacn(fun: Callable[[Array], Array], k: int = 1) -> Callable[[Array], Array]:
    for _ in range(k):
        fun = jax.jacfwd(fun)  # if i % 2 else jax.jacrev(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0)


@overload
def matmat(a: Array, b: Array) -> Array: ...
@overload
def matmat(a: BCOO, b: BCOO) -> BCOO: ...
@overload
def matmat(a: Array, b: BCOO) -> Array: ...  # unchecked
@overload
def matmat(a: BCOO, b: Array) -> Array: ...  # unchecked
@jax.jit
def matmat(a: Array | BCOO, b: Array | BCOO) -> Array | BCOO:
    return a @ b


@jax.jit(static_argnums=1)
def eliminate_near_zeros(a: Array, tol: int = 100) -> Array:
    atol: Array = ulp(jnp.abs(a).max()) * tol
    return jnp.where(jnp.abs(a) < atol, jnp.zeros(a.shape), a)


def fromdense(a: Array, tol: int = 100) -> BCOO:
    a0: Array = eliminate_near_zeros(a, tol=tol)
    return BCOO.fromdense(a0)


def tosparse(a: Array, tol: int = 100) -> BCOO:
    a0: Array = eliminate_near_zeros(a, tol=tol)
    return BCOO.from_scipy_sparse(scipy_sparse.csr_matrix(a0))


class ArrayFn(Protocol):
    def __call__(self, *args: Array) -> Array: ...


def lambdify(
    args: sp.Basic | tuple[Symbol | BaseScalar, ...] | sp.Tuple | None,
    expr: Expr | sp.Basic,
    modules: str | list[str | dict[str, Callable]] | None = None,
    printer: Any = None,
    use_imps: bool = True,
    dummify: bool = False,
    cse: bool = False,
    doctring_limit: int = 1000,
) -> ArrayFn:
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


def reverse_dict[K, V](d: dict[K, V]) -> dict[V, K]:
    """Reverse a dictionary.

    Args:
        d: The dictionary to reverse.

    Raises:
        ValueError: If the values in the dictionary are not unique.

    Returns:
        The dictionary with key-value pairs reversed.
    """
    rev_dict = {v: k for k, v in d.items()}
    if len(rev_dict) != len(d):
        raise ValueError("Cannot reverse dict with non-unique values.")
    return rev_dict
