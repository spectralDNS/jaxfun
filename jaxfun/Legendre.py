from functools import partial
import jax
import jax.numpy as jnp
from jax import Array
import sympy as sp
from jaxfun.utils.fastgl import leggauss
from jaxfun.Jacobi import Jacobi, Domain, n
from jaxfun.coordinates import CoordSys


class Legendre(Jacobi):
    def __init__(
        self,
        N: int,
        domain: Domain = None,
        system: CoordSys = None,
        name: str = "Legendre",
        fun_str: str = "P",
        **kw,
    ) -> None:
        Jacobi.__init__(
            self,
            N,
            domain=domain,
            system=system,
            name=name,
            fun_str=fun_str,
            alpha=0,
            beta=0,
        )

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
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
            return c[0] + 0 * X
        if len(c) == 2:
            return c[0] + c[1] * X

        def body_fun(i: int, val: tuple[int, Array, Array]) -> tuple[int, Array, Array]:
            nd, c0, c1 = val

            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1 * (nd - 1)) / nd
            c1 = tmp + (c1 * X * (2 * nd - 1)) / nd

            return nd, c0, c1

        nd = len(c)
        c0 = jnp.ones_like(X) * c[-2]
        c1 = jnp.ones_like(X) * c[-1]

        _, c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body_fun, (nd, c0, c1))
        return c0 + c1 * X

    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N if N == 0 else N
        return leggauss(N)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, X: float, i: int) -> float:
        x0 = X * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return x1, x2

        return jax.lax.fori_loop(2, i + 1, body_fun, (x0, X))[-1]

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, X: float) -> Array:
        x0 = X * 0 + 1

        def inner_loop(
            carry: tuple[float, float], i: int
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))

        return jnp.hstack((x0, xs))

    def norm_squared(self) -> Array:
        return sp.lambdify(n, 2 / (2 * n + 1), modules="jax")(jnp.arange(self.N))


def matrices(test: tuple[Legendre, int], trial: tuple[Legendre, int]) -> Array:
    import numpy as np
    from jax.experimental import sparse
    from scipy import sparse as scipy_sparse

    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        return sparse.BCOO.from_scipy_sparse(
            scipy_sparse.diags((v.norm_squared(),), (0,), (v.N, u.N), "csr")
        )
    if i == 0 and j == 1:

        if u.N < 2:
            return None

        return sparse.BCOO.from_scipy_sparse(
            scipy_sparse.diags(
                [2.0] * len(jnp.arange(1, u.N, 2)),
                jnp.arange(1, u.N, 2),
                (v.N, u.N),
                "csr",
            )
        )
    if i == 1 and j == 0:
        return matrices(trial, test).transpose()

    if i == 0 and j == 2:
        k = jnp.arange(max(v.N, u.N))

        def _getkey(j):
            Q = min(v.N, u.N - j)
            return (
                (k[:Q] + 0.5)
                * (k[j : (Q + j)] * (k[j : (Q + j)] + 1) - k[:Q] * (k[:Q] + 1))
                * 2.0
                / (2 * k[:Q] + 1)
            )

        d = dict.fromkeys(np.arange(2, u.N, 2), _getkey)

        if len(d) == 0:
            return None

        return sparse.BCOO.from_scipy_sparse(
            scipy_sparse.diags(
                [d[i](i) for i in np.arange(2, u.N, 2)],
                jnp.arange(2, u.N, 2),
                (v.N, u.N),
                "csr",
            )
        )
    if i == 2 and j == 0:
        return matrices(trial, test).transpose()

    return None
