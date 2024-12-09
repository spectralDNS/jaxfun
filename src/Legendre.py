from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from utils.common import jacn
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
    x0 = jnp.ones_like(x)

    def inner_loop(
        carry: tuple[Array, Array], i: int
    ) -> tuple[tuple[Array, Array], Array]:
        x0, x1 = carry
        x2 = (x1 * x * (2 * i - 1) - x0 * (i - 1)) / i
        return (x1, x2), x1

    _, xs = jax.lax.scan(inner_loop, (x0, x), jnp.arange(2, deg + 1))

    xs = jnp.concatenate((x0[None, :], xs), axis=0)
    return jnp.moveaxis(xs, 0, -1)


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
