from __future__ import annotations

from collections.abc import Callable
from functools import cache, wraps
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
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
    "cache_static",
    "dct",
    "diff",
    "diffx",
    "Domain",
    "dst",
    "idct",
    "jacn",
    "JAX_FUNCTION_BY_NAME",
    "matmat",
    "tosparse",
    "lambdify",
)


JAX_FUNCTION_BY_NAME: dict[str, Callable[..., Array]] = {
    "Abs": jnp.abs,
    "acos": jnp.arccos,
    "acosh": jnp.arccosh,
    "asin": jnp.arcsin,
    "asinh": jnp.arcsinh,
    "atan": jnp.arctan,
    "atan2": jnp.arctan2,
    "atanh": jnp.arctanh,
    "cos": jnp.cos,
    "cosh": jnp.cosh,
    "exp": jnp.exp,
    "log": jnp.log,
    "sign": jnp.sign,
    "sin": jnp.sin,
    "sinh": jnp.sinh,
    "sqrt": jnp.sqrt,
    "tan": jnp.tan,
    "tanh": jnp.tanh,
}


def cache_static[FuncT: Callable[..., Any]](func: FuncT) -> FuncT:
    """Decorator that memoizes a method of which the result depends only on the
    instance and hashable arguments.

    Meant as a replacement for `jax.jit(static_argnums=...)` on methods taking no
    array arguments. `jax.jit` inlines such a method into every enclosing trace, so
    its entire subgraph is staged into the jaxpr (and constant folded by XLA) once
    per compilation. This evaluates the body once and hands out the stored result
    afterwards.

    The body runs under `jax.ensure_compile_time_eval`, so it is evaluated eagerly
    even when the caller is tracing: an enclosing jaxpr receives the finished
    array as a constant rather than the computation that produced it, and no
    tracer is ever stored in the cache.

    Results are held on the host as numpy arrays and converted back on the way
    out. A jax array remembers the mesh context it was created under, so one
    first computed inside a `shard_map` would carry that mesh (`axis_types=Manual`)
    into every later use outside it, and mixing it with an unsharded array then
    fails. Going via the host drops that context, and the array handed back picks
    up the caller's.

    The cache lives on the instance, keyed by the arguments and by the instance's
    `_cache_key` (empty when undefined), which spells out the state the result
    depends on.

    Args:
        func: Method taking `self` plus hashable arguments only.
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        cache: dict[Any, Any] = self.__dict__.setdefault("_static_cache", {})
        key = (
            func,  # identifies the decorated method within the instance's cache
            getattr(self, "_cache_key", ()),
            args,
            tuple(sorted(kwargs.items())),
        )
        if key not in cache:
            with jax.ensure_compile_time_eval():
                cache[key] = jax.tree.map(np.asarray, func(self, *args, **kwargs))
        return jax.tree.map(jnp.asarray, cache[key])

    return cast(FuncT, wrapper)


def jit_vmap[FuncT: Callable[..., Array]](
    in_axes: int | None | tuple[int | None, ...] = 0,
    out_axes: Any = 0,
    static_argnums: int | tuple[int, ...] | None = 0,
    ndim: int = 0,
) -> Callable[[FuncT], FuncT]:
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

    def wrap(func: FuncT) -> FuncT:
        @wraps(func)
        def wrapper(self: Any, x: Any, /, *args: Any, **kwargs: Any) -> Array:
            x = jnp.asarray(x)
            if x.ndim == ndim:
                return func(self, x, *args, **kwargs)

            mapped = jax.vmap(func, in_axes=in_axes, out_axes=out_axes)
            return mapped(self, x, *args, **kwargs)

        return cast(FuncT, jax.jit(wrapper, static_argnums=static_argnums))

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


@jax.jit
def matmat(a: Array, b: Array) -> Array:
    return a @ b


@jax.jit(static_argnums=1)
def eliminate_near_zeros(a: Array, tol: int = 100) -> Array:
    atol: Array = ulp(jnp.abs(a).max()) * tol
    return jnp.where(jnp.abs(a) < atol, jnp.zeros(a.shape), a)


def tosparse(a: Array, tol: int = 100) -> DiaMatrix:
    """Convert a dense array to a sparse DiaMatrix, eliminating near-zero entries.

    Args:
        a: The input dense array.
        tol: The tolerance for eliminating near-zero entries, in units of ULP.
            An entry is kept only if ``ulp(max|a|) * tol >= max|a|``.
    """
    return DiaMatrix.from_dense(a, tol=tol)


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


@cache
def _dst_twiddle(N: int) -> np.ndarray:
    """Return the DST-II phase factor exp(-i*pi*(k+1)/2N), as a host array."""
    return np.exp(-1j * np.pi * (np.arange(N) + 1) / (2 * N))


def _dst_modes(
    Y: Array, N: int, M: int, axis: int, tw: np.ndarray | None, complex_input: bool
) -> Array:
    """Return the DST from the FFT `Y` of the odd extension, length `M`.

    For a real input the odd extension is real, `Y` is Hermitian, and the answer
    is -Im of its modes 1..N. A complex input z = a + ib extends linearly, so
    Y = Ya + i*Yb with Ya and Yb each Hermitian, and the two come back apart
    from Y alone:

        Ya[k] = (Y[k] + conj(Y[M-k])) / 2      Yb[k] = -i (Y[k] - conj(Y[M-k])) / 2

    which is one complex FFT for both halves, where the alternative is running
    the whole transform twice. Writing P and Q for those two mode ranges, the
    -Im that finishes each half turns into -Im on one and +Re on the other,
    because -Im(-i*w) = Re(w). For real input Q == P and this collapses back to
    the real formula exactly.

    Args:
        Y: FFT of the odd extension, along `axis`.
        N: Number of output modes.
        M: Length of the odd extension.
        axis: Axis transformed along.
        tw: Per-mode phase, or None for type 1, which has none.
        complex_input: Whether the untransformed input was complex.

    Returns:
        The transform, `N` modes along `axis`.
    """
    P = jax.lax.slice_in_dim(Y, 1, N + 1, axis=axis)
    if not complex_input:
        return -jnp.imag(P if tw is None else tw * P)
    # Modes M-1 down to M-N, which is the reversed tail of Y.
    Q = jnp.conj(jnp.flip(jax.lax.slice_in_dim(Y, M - N, M, axis=axis), axis=axis))
    A, B = 0.5 * (P + Q), 0.5 * (P - Q)
    if tw is not None:
        A, B = tw * A, tw * B
    return -A.imag + 1j * B.real


@jax.jit(static_argnums=(1, 2, 3))
def dst(x: Array, axis: int = -1, type: int = 2, n: int | None = None) -> Array:
    """Return the discrete sine transform of `x`.

    Matches `scipy.fft.dst` with `norm=None`. A complex input is transformed in
    a single FFT rather than as two real transforms.

    Args:
        x: Input array, real or complex.
        axis: Axis to transform along.
        type: DST type, 1 or 2.
        n: Length of the transform. `x` is zero-padded up to it. Defaults to the
            length of `x` along `axis`.

    Returns:
        The transform, of the same shape as `x` but with `axis` of length `n`.
    """
    N = x.shape[axis] if n is None else n
    # Resized before either odd extension is built, not after: the extension has
    # to be the one belonging to `N`, or the mode ranges `_dst_modes` slices out
    # of it address the wrong entries. scipy reads `n` as the length of the
    # transform, so a shorter one crops -- `dst(x, n=k)` equals `dst(x[:k])`.
    if x.shape[axis] > N:
        x = jax.lax.slice_in_dim(x, 0, N, axis=axis)
    elif x.shape[axis] < N:
        # One spec per axis. A single `[(0, k)]` broadcasts to every axis, so a
        # 2-D input used to pad both and then fail to broadcast against the
        # twiddle.
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, N - x.shape[axis])
        x = jnp.pad(x, pad, mode="constant")
    is_complex = jnp.iscomplexobj(x)

    if type == 1:
        # odd extension to length 2(N+1) with zero endpoints
        pad_shape = list(x.shape)
        pad_shape[axis] = 1
        zeros = jnp.zeros(pad_shape, dtype=x.dtype)
        y = jnp.concatenate([zeros, x, zeros, -jnp.flip(x, axis=axis)], axis=axis)
        Y = jnp.fft.fft(y, axis=axis)
        return _dst_modes(Y, N, 2 * (N + 1), axis, None, is_complex)

    if type == 2:
        # odd extension to length 2N
        y = jnp.concatenate([x, -jnp.flip(x, axis=axis)], axis=axis)
        Y = jnp.fft.fft(y, axis=axis)
        # Built on the host and memoized, for the reasons given above the dct:
        # XLA does not fold an `arange`/`exp` built inside the jit even with N
        # static, and a cached jnp array reached from inside a trace would store
        # a tracer.
        tw = _dst_twiddle(N)
        if axis not in (-1, x.ndim - 1):
            tw = np.expand_dims(tw, [a for a in range(x.ndim) if a != axis % x.ndim])
        return _dst_modes(Y, N, 2 * N, axis, tw, is_complex)

    raise ValueError(f"Unsupported dst type: {type}")


# Makhoul's 1980 algorithm, as jax.scipy.fft implements it, but taking a complex
# input in one FFT rather than two. `jax.scipy.fft.dct`/`.idct` split a complex
# argument as `lax.complex(f(x.real), f(x.imag))` and run the whole transform on
# each half, and each half then takes a full complex-to-complex FFT even though
# its own input is real. The interleaved transform of a real sequence is
# Hermitian, so both halves ride in one complex FFT and separate again with a
# reversal and a conjugate. With W4[k] = exp(-i*pi*k/2N) and H[k] = conj(F[-k]):
#
#   dct    F    = fft(interleave(z))
#          out  = Re((F + H)*W4) + i*Im((F - H)*W4)
#
#   idct   G[k] = N*s[k]*( z[k]*conj(W4[k]) + z[-k]*W4[-k] )
#          out  = deinterleave(ifft(G))
#
# The idct takes no real part at all: its two halves land in the real and
# imaginary parts of ifft(G), so one deinterleave serves both. Measured on 257
# rows of complex data, agreeing with jax.scipy to 3e-16, in milliseconds:
#
#   N                    128      256      512
#   idct  jax.scipy     0.381    0.839    1.801
#   idct  here          0.112    0.208    0.483    3.4x  4.0x  3.7x
#   dct   jax.scipy     0.211    0.422    0.931
#   dct   here          0.132    0.251    0.546    1.6x  1.7x  1.7x
#
# The idct has more to give back because jax.scipy's carries more scaffolding to
# begin with: it applies `_dct_ortho_norm` twice, divides by W4 rather than
# multiplying by its conjugate, and ends in a strided-scatter deinterleave where
# the dct only needs a strided-slice interleave.
#
# The twiddles are built on the host and memoized, for two separate reasons. XLA
# does not constant-fold the `arange`/`exp` that makes them, even though N is
# static, and leaving them inside the jit costs 1.7x. And they must be numpy
# rather than jnp: these are reached lazily from inside a trace, and on a
# distributed run from inside a `shard_map`, where `jnp.asarray` hands back a
# tracer that the cache would then store and every later call from outside that
# trace would leak. Host arrays carry no trace or mesh affiliation and XLA embeds
# them as literals, which is also why `cache_static` goes via the host. Being
# literals they need no communication when distributed: each device gets its own
# copy compiled in, and the module picks up no collectives from them.


@cache
def _dct_twiddle(N: int) -> np.ndarray:
    """Return W4[k] = exp(-i*pi*k/2N), jax.scipy's DCT twiddle, as a host array."""
    return np.exp(-0.5j * np.pi * np.arange(N) / N)


@cache
def _idct_twiddles(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the two vectors `idct` multiplies its input by, as host arrays."""
    k = np.arange(N)
    # jax.scipy applies `_dct_ortho_norm` twice for norm=None, a division by
    # [4, 2, 2, ...]*N; the 2*N it multiplies by afterwards is folded in here.
    s = np.where(k == 0, 1.0 / (4 * N), 1.0 / (2 * N)) * (2 * N)
    w4 = _dct_twiddle(N)
    return 0.5 * s * np.conj(w4), 0.5 * s * w4


def _interleave(x: Array) -> Array:
    """Reorder the last axis as the even samples then the odd ones reversed."""
    return jnp.concatenate([x[..., 0::2], jnp.flip(x[..., 1::2], -1)], -1)


def _deinterleave(x: Array) -> Array:
    """Undo `_interleave` along the last axis."""
    # A stack-and-reshape rather than jax.scipy's `out.at[..., 0::2].set(...)`,
    # which is a strided scatter and measures 5x slower. The halves are unequal
    # for odd N, so the shorter one is padded by one and the extra column is
    # dropped again after the reshape.
    N = x.shape[-1]
    h = (N + 1) // 2
    even, odd = x[..., :h], jnp.flip(x[..., h:], -1)
    if N % 2:
        odd = jnp.pad(odd, [(0, 0)] * (x.ndim - 1) + [(0, 1)])
    return jnp.stack([even, odd], -1).reshape(*x.shape[:-1], 2 * h)[..., :N]


def _to_last(x: Array, axis: int, N: int | None) -> tuple[Array, int]:
    """Move `axis` last and resize it to `N`, cropping or zero-padding."""
    x = x if axis in (-1, x.ndim - 1) else jnp.moveaxis(x, axis, -1)
    N = x.shape[-1] if N is None else N
    # scipy takes `n` as the length of the transform, not a lower bound: a
    # shorter one crops the input rather than transforming all of it and
    # returning part, so `dct(x, n=k)` equals `dct(x[:k])`.
    if x.shape[-1] > N:
        x = x[..., :N]
    elif x.shape[-1] < N:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, N - x.shape[-1])])
    return x, N


def _scaled_dct(x: Array, w: np.ndarray) -> Array:
    """Return the type-II DCT along the last axis, scaled per mode by `w`."""
    # `w` multiplies before the Re/Im rather than after, because a real per-mode
    # factor commutes with them -- so a caller's output scaling folds in for
    # free, which is what the Chebyshev transforms below do with it.
    F = jnp.fft.fft(_interleave(x), axis=-1)
    Fw = F * w
    Hw = jnp.conj(jnp.roll(jnp.flip(F, -1), 1, -1)) * w
    if not jnp.iscomplexobj(x):
        # H == F for a real input, so the imaginary half is identically zero.
        return Fw.real + Hw.real
    return (Fw.real + Hw.real) + 1j * (Fw.imag - Hw.imag)


@jax.jit(static_argnums=(1, 2, 3))
def dct(x: Array, axis: int = -1, type: int = 2, n: int | None = None) -> Array:
    """Return the discrete cosine transform of `x`.

    Matches `scipy.fft.dct` with `norm=None`, and unlike `jax.scipy.fft.dct`
    takes a complex input in a single FFT.

    Args:
        x: Input array, real or complex.
        axis: Axis to transform along.
        type: DCT type. Only type 2 is implemented.
        n: Length of the transform. `x` is zero-padded up to it. Defaults to the
            length of `x` along `axis`.

    Returns:
        The transform, of the same shape as `x` but with `axis` of length `n`.
    """
    if type != 2:
        raise ValueError(f"Unsupported dct type: {type}")
    x, N = _to_last(x, axis, n)
    out = _scaled_dct(x, _dct_twiddle(N))
    return out if axis in (-1, x.ndim - 1) else jnp.moveaxis(out, -1, axis)


@jax.jit(static_argnums=(1, 2, 3))
def idct(x: Array, axis: int = -1, type: int = 2, n: int | None = None) -> Array:
    """Return the inverse discrete cosine transform of `x`.

    Matches `scipy.fft.idct` with `norm=None`, and unlike `jax.scipy.fft.idct`
    takes a complex input in a single FFT.

    Args:
        x: Input array, real or complex.
        axis: Axis to transform along.
        type: DCT type. Only type 2 is implemented.
        n: Length of the transform. `x` is zero-padded up to it. Defaults to the
            length of `x` along `axis`.

    Returns:
        The transform, of the same shape as `x` but with `axis` of length `n`.
    """
    if type != 2:
        raise ValueError(f"Unsupported idct type: {type}")
    x, N = _to_last(x, axis, n)
    fwd, rev = _idct_twiddles(N)
    G = x * fwd + jnp.roll(jnp.flip(x * rev, -1), 1, -1)
    out = _deinterleave(jnp.fft.ifft(G, axis=-1))
    if not jnp.iscomplexobj(x):
        # The two halves land in the real and imaginary parts of ifft(G), and
        # the imaginary one is identically zero for a real input.
        out = out.real
    return out if axis in (-1, x.ndim - 1) else jnp.moveaxis(out, -1, axis)
