from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from utils.common import jacn
from utils.fastgl import leggauss
import sympy as sp

n = sp.Symbol("n", integer=True, positive=True)

# Jacobi constants
alpha, beta = 0, 0


# Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
def gn(alpha, beta, n):
    return 1


@jax.jit
def evaluate(x: float, c: Array) -> float:
    """
    Evaluate a Legendre series at points x.

    .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

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

    def body_fun(i: int, val: tuple[int, Array, Array]) -> tuple[int, Array, Array]:
        nd, c0, c1 = val

        tmp = c0
        nd = nd - 1
        c0 = c[-i] - (c1 * (nd - 1)) / nd
        c1 = tmp + (c1 * x * (2 * nd - 1)) / nd

        return nd, c0, c1

    nd = len(c)
    c0 = jnp.ones_like(x) * c[-2]
    c1 = jnp.ones_like(x) * c[-1]

    _, c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body_fun, (nd, c0, c1))
    return c0 + c1 * x


def quad_points_and_weights(N: int) -> Array:
    return leggauss(N)


@partial(jax.jit, static_argnums=(1, 2))
def evaluate_basis_derivative(x: Array, deg: int, k: int = 0) -> Array:
    return jacn(eval_basis_functions, k)(x, deg)


@partial(jax.jit, static_argnums=1)
def vandermonde(x: Array, deg: int) -> Array:
    return evaluate_basis_derivative(x, deg, 0)


@partial(jax.jit, static_argnums=1)
def eval_basis_function(x: float, i: int) -> float:
    return evaluate(x, (0,) * i + (1,))


@partial(jax.jit, static_argnums=1)
def eval_basis_functions(x: float, deg: int) -> Array:
    x0 = x * 0 + 1

    def inner_loop(
        carry: tuple[float, float], i: int
    ) -> tuple[tuple[float, float], Array]:
        x0, x1 = carry
        x2 = (x1 * x * (2 * i - 1) - x0 * (i - 1)) / i
        return (x1, x2), x1

    _, xs = jax.lax.scan(inner_loop, (x0, x), jnp.arange(2, deg + 1))

    return jnp.hstack((x0, xs))
