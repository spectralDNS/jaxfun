from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from utils.common import diff
from utils.fastgl import leggauss


@jax.jit
def legval(x: float, c: Array) -> float:
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
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
    return c0 + c1 * x


def quad_points_and_weights(N: int) -> Array:
    return leggauss(N)


@partial(jax.jit, static_argnums=(1, 2))  # Very slow without this
def evaluate_basis_derivative(x: Array, deg: int, k: int = 0) -> Array:
    c = jnp.eye(deg)
    f = jax.vmap(lambda i: diff(legval, k=k)(x, c[i]))(jnp.arange(deg))
    return jnp.moveaxis(f, 0, -1)


@partial(jax.jit, static_argnums=1)
def eval_basis_function(x: float, i: int) -> float:
    return legval(x, (0,) * i + (1,))


@partial(jax.jit, static_argnums=2)
def evaluate(x: Array, c: Array, axes: tuple[int] = (0,)) -> Array:
    """Evaluate along one or more axes"""
    dim: int = len(c.shape)
    for ax in axes:
        axi: int = dim - 1 - ax
        c = jax.vmap(legval, in_axes=(None, axi))(x, c)
    return c


@partial(jax.jit, static_argnums=1)
def legvander(x: Array, deg: int) -> Array:
    f = [x * 0 + 1]
    if deg > 0:
        f.append(x)
        for i in range(2, deg):
            f.append((f[i - 1] * x * (2 * i - 1) - f[i - 2] * (i - 1)) / i)
    f = jnp.array(f)
    return jnp.moveaxis(f, 0, -1)


def bilinear(N: int, i: int, j: int) -> Array:
    x, w = quad_points_and_weights(N)
    Pi = evaluate_basis_derivative(x, N, k=i)
    Pj = evaluate_basis_derivative(x, N, k=j)
    return (Pi.T * w[None, :]) @ Pj


def linear(u, N: int) -> Array:
    x, w = quad_points_and_weights(N)
    Pi = legvander(x, N)
    uj = sp.lambdify(s, u, modules=["jax"])(x)
    return (uj * w) @ Pi


if __name__ == "__main__":
    # Solve Poisson's equation
    import matplotlib.pyplot as plt
    import sympy as sp
    from jax.experimental import sparse

    s = sp.Symbol("s")

    ue = (1 - s**2) * sp.exp(sp.cos(2 * sp.pi * s))
    f = ue.diff(s, 2)
    N = 50
    S = sparse.eye(N, N + 2) - sparse.eye(N, N + 2, 2)  # Dirichlet
    # S = jnp.eye(N, N+2) - jnp.eye(N, N+2, 2)
    A = S @ bilinear(N + 2, 0, 2) @ S.T
    b = S @ linear(f, N + 2)
    u = jnp.linalg.solve(A, b)
    x = jnp.linspace(-1, 1, 100)
    plt.plot(x, sp.lambdify(s, ue)(x), "r")
    plt.plot(x, legval(x, u @ S), "b")
    plt.show()
