from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from scipy.special import sph_harm_y
from sympy import Expr, Symbol

from jaxfun.la import DiaMatrix
from jaxfun.typing import FloatLike

if TYPE_CHECKING:
    from jaxfun.coordinates import BaseScalar

Ynm = sph_harm_y
n = Symbol("n", integer=True)


__all__ = (
    "diff",
    "diffx",
    "Domain",
    "jacn",
    "matmat",
    "tosparse",
    "lambdify",
)


# TODO: Add typehints for this
def jit_vmap[SelfT, **P](
    in_axes: int | None | tuple[int | None, ...] = 0,
    out_axes: Any = 0,
    static_argnums: int | tuple[int, ...] | None = 0,
    ndim: int = 0,
) -> Callable[
    [Callable[Concatenate[SelfT, Array, P], Array]],
    Callable[Concatenate[SelfT, Array, P], Array],
]:
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

    def wrap(
        func: Callable[Concatenate[SelfT, Array, P], Array],
    ) -> Callable[Concatenate[SelfT, Array, P], Array]:
        @wraps(func)
        def wrapper(
            self: SelfT, x: Array, /, *args: P.args, **kwargs: P.kwargs
        ) -> Array:
            if x.ndim == ndim:
                return func(self, x, *args, **kwargs)

            mapped = jax.vmap(func, in_axes=in_axes, out_axes=out_axes)
            return mapped(self, x, *args, **kwargs)

        return cast(
            Callable[Concatenate[SelfT, Array, P], Array],
            jax.jit(wrapper, static_argnums=static_argnums),
        )

    return cast(
        Callable[
            [Callable[Concatenate[SelfT, Array, P], Array]],
            Callable[Concatenate[SelfT, Array, P], Array],
        ],
        wrap,
    )


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


@jax.jit
def matmat(a: Array, b: Array) -> Array:
    return a @ b


@jax.jit(static_argnums=1)
def eliminate_near_zeros(a: Array, tol: int = 100) -> Array:
    atol: Array = ulp(jnp.abs(a).max()) * tol
    return jnp.where(jnp.abs(a) < atol, jnp.zeros(a.shape), a)


def tosparse(a: Array, tol: int = 100) -> DiaMatrix:
    a0: Array = eliminate_near_zeros(a, tol=tol)
    return DiaMatrix.from_dense(a0)


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
    modules = modules_default if modules is None else [modules] + modules_default  # ty:ignore[invalid-assignment]
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


@jax.jit(static_argnums=(1, 2, 3))
def dst(x: Array, axis: int = -1, type: int = 2, n: int | None = None) -> Array:
    N = x.shape[axis] if n is None else n
    x = (
        jnp.pad(x, [(0, N - x.shape[axis])], mode="constant")
        if x.shape[axis] < N
        else x
    )
    if type == 1:
        # odd extension to length 2(N+1) with zero endpoints
        pad_shape = list(x.shape)
        pad_shape[axis] = 1
        zeros = jnp.zeros(pad_shape, dtype=x.dtype)
        y = jnp.concatenate([zeros, x, zeros, -jnp.flip(x, axis=axis)], axis=axis)

        Y = jnp.fft.fft(y, axis=axis)
        k = jnp.arange(N)
        Yk = jnp.take(Y, indices=k + 1, axis=axis)

        return -jnp.imag(Yk)

    if type == 2:
        # odd extension to length 2N
        y = jnp.concatenate([x, -jnp.flip(x, axis=axis)], axis=axis)

        Y = jnp.fft.fft(y, axis=axis)
        k = jnp.arange(N)
        # take modes 1..N and apply phase
        Yk = jnp.take(Y, indices=k + 1, axis=axis)
        tw = jnp.exp(-1j * jnp.pi * (k + 1) / (2 * N))

        return -jnp.imag(tw * Yk)

    raise ValueError(f"Unsupported dst type: {type}")
