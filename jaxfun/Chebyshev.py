from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
from jaxfun.Jacobi import Jacobi, Domain
from jaxfun.coordinates import CoordSys
import sympy as sp
from sympy import Symbol, Expr 


class Chebyshev(Jacobi):
    def __init__(
        self,
        N: int,
        domain: Domain = Domain(-1, 1),
        system: CoordSys = None,
        name: str = "Chebyshev",
        fun_str: str = "T",
        **kw,
    ) -> None:
        Jacobi.__init__(
            self,
            N,
            domain=domain,
            system=system,
            name=name,
            fun_str=fun_str,
            alpha=-sp.S.Half,
            beta=-sp.S.Half,
        )

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        """
        Evaluate a Chebyshev series at points x.

        .. math:: p(x) = c_0 * T_0(x) + T_1 * T_1(x) + ... + c_n * T_n(x)

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

    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N + 1 if N == 0 else N + 1
        return jnp.array(
            (
                jnp.cos(jnp.pi + (2 * jnp.arange(N) + 1) * jnp.pi / (2 * N)),
                jnp.ones(N) * jnp.pi / N,
            )
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        # return jnp.cos(i * jnp.acos(x))
        x0 = x * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = 2 * x * x1 - x0
            return x1, x2

        return jax.lax.fori_loop(1, i, body_fun, (x0, x))[-1]

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        x0 = x * 0 + 1

        def inner_loop(
            carry: tuple[float, float], _
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            x2 = 2 * x * x1 - x0
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, init=(x0, x), xs=None, length=self.N)

        return jnp.hstack((x0, xs))

    def norm_squared(self) -> Array:
        return jnp.hstack((jnp.array([jnp.pi]), jnp.full(self.N, jnp.pi / 2)))

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    def gn(self, n: Symbol) -> Expr:
        return sp.S(1) / sp.jacobi(n, self.alpha, self.beta, 1)
