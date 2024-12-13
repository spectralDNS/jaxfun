from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from utils.common import jacn
import sympy as sp

n = sp.Symbol('n', integer=True, positive=True)

# Jacobi constants
alpha, beta = -sp.S.Half, -sp.S.Half

# Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
def gn(alpha, beta, n):
    return sp.S(1)/sp.jacobi(n, alpha, beta, 1)

@jax.jit
def evaluate(x: float, c: Array) -> float:
    """
    Evaluate a Chebyshev series at points x.

    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)

    Parameters
    ----------
    x : float
    c : Array

    Returns
    -------
    values : Array

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    if len(c) == 1:
        # Multiply by 0 * x for shape
        return c[0] + 0 * x
    if len(c) == 2:
        return c[0] + c[1] * x

    def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
        c0, c1 = val

        tmp = c0
        c0 = c[-i] - c1
        c1 = tmp + c1 * 2 * x

        return c0, c1

    c0 = jnp.ones_like(x) * c[-2]
    c1 = jnp.ones_like(x) * c[-1]

    c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body_fun, (c0, c1))
    return c0 + c1 * x


def quad_points_and_weights(N: int) -> Array:
    return jnp.array(
        (
            jnp.cos(jnp.pi + (2 * jnp.arange(N) + 1) * jnp.pi / (2 * N)),
            jnp.ones(N) * jnp.pi / N,
        )
    )


@partial(jax.jit, static_argnums=(1, 2))
def evaluate_basis_derivative(x: Array, deg: int, k: int = 0) -> Array:
    return jacn(eval_basis_functions, k)(x, deg)


@partial(jax.jit, static_argnums=1)
def vandermonde(x: Array, deg: int) -> Array:
    return evaluate_basis_derivative(x, deg, 0)


@partial(jax.jit, static_argnums=1)
def eval_basis_function(x: float, i: int) -> Array:
    # return jnp.cos(i * jnp.acos(x))
    x0 = x * 0 + 1
    if i == 0:
        return x0

    def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
        x0, x1 = val
        x2 = 2 * x * x1 - x0
        return x1, x2

    return jax.lax.fori_loop(1, i, body_fun, (x0, x))[-1]


@partial(jax.jit, static_argnums=1)
def eval_basis_functions(x: float, deg: int) -> Array:
    x0 = x * 0 + 1

    def inner_loop(carry: tuple[float, float], _) -> tuple[tuple[float, float], Array]:
        x0, x1 = carry
        x2 = 2 * x * x1 - x0
        return (x1, x2), x1

    _, xs = jax.lax.scan(inner_loop, init=(x0, x), xs=None, length=deg - 1)

    return jnp.hstack((x0, xs))
