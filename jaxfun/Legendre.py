from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
import sympy as sp
from jaxfun.utils.fastgl import leggauss
from jaxfun.Jacobi import Jacobi, Domain, n


class Legendre(Jacobi):
    def __init__(self, N: int, domain: Domain = Domain(-1, 1), **kw):
        Jacobi.__init__(self, N, domain, 0, 0)

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
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

    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N+1 if N == 0 else N+1
        return leggauss(N)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        x0 = x * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = (x1 * x * (2 * i - 1) - x0 * (i - 1)) / i
            return x1, x2

        return jax.lax.fori_loop(1, i, body_fun, (x0, x))[-1]

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        x0 = x * 0 + 1

        def inner_loop(
            carry: tuple[float, float], i: int
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            x2 = (x1 * x * (2 * i - 1) - x0 * (i - 1)) / i
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, (x0, x), jnp.arange(2, self.N + 2))

        return jnp.hstack((x0, xs))

    def norm_squared(self) -> Array:
        return sp.lambdify(n, 2 / (2*n+1), modules='jax')(jnp.arange(self.N+1))
