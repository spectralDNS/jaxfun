import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap
from jaxfun.utils.fastgl import leggauss

from .Jacobi import Jacobi


class Legendre(Jacobi):
    """Legendre polynomial basis (Jacobi with alpha=beta=0).

    Provides series and basis evaluation, quadrature nodes/weights, and
    norm-squared values for P_n on [-1, 1].
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
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

    @jit_vmap(in_axes=(0, None))
    def _evaluate2(self, X: float | Array, c: Array) -> Array:
        """Evaluate Legendre series using backward two-term scheme.

        p(X) = sum_{k=0}^{N-1} c_k P_k(X)

        Args:
            X: Evaluation points in reference domain [-1, 1].
            c: Coefficient array (length >= 1).

        Returns:
            Series values p(X) with shape like X.
        """
        nd: int = len(c)
        if nd == 1:
            return c[0] + 0 * X  # shape preservation
        if nd == 2:
            return c[0] + c[1] * X

        def body_fun(i: int, val: tuple[int, Array, Array]) -> tuple[int, Array, Array]:
            n, c0, c1 = val
            tmp = c0
            n -= 1
            c0 = c[-i] - (c1 * (n - 1)) / n
            c1 = tmp + (c1 * X * (2 * n - 1)) / n
            return n, c0, c1

        c0: Array = jnp.ones_like(X) * c[-2]
        c1: Array = jnp.ones_like(X) * c[-1]
        _, c0, c1 = jax.lax.fori_loop(3, nd + 1, body_fun, (nd, c0, c1))
        return c0 + c1 * X

    @jit_vmap(in_axes=(0, None))
    def _evaluate3(self, X: float | Array, c: Array) -> Array:
        """Evaluate Legendre series via forward recurrence accumulation.

        Args:
            X: Evaluation points in reference domain [-1, 1].
            c: Coefficient array.

        Returns:
            Series values p(X) with shape like X.
        """
        x0 = jnp.ones_like(X)

        def inner_loop(
            carry: tuple[Array, Array], i: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return (x1, x2), x1 * c[i - 1]

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))
        return jnp.sum(xs, axis=0) + c[0]

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        """Return Gauss-Legendre quadrature nodes and weights.

        Args:
            N: Number of points (None => self.num_quad_points).

        Returns:
            Tuple (x, w) of nodes and weights.
        """
        N = self.num_quad_points if N is None else N
        x, w = leggauss(N)
        return x, w

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: Array, i: int) -> Array:
        """Evaluate single Legendre polynomial P_i at X.

        Args:
            X: Points in [-1, 1].
            i: Polynomial index (0 <= i < N).

        Returns:
            P_i(X) values.
        """
        x0 = X * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return x1, x2

        return jax.lax.fori_loop(2, i + 1, body_fun, (x0, X))[-1]

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        """Evaluate all Legendre polynomials P_0..P_{N-1} at X.

        Args:
            X: Points in [-1, 1].

        Returns:
            Array (N,) per X with stacked values.
        """
        x0 = X * 0 + 1

        def inner_loop(
            carry: tuple[Array, Array], i: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))
        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    @jax.jit(static_argnums=(0, 2))
    def derivative_coeffs(self, c: Array, k: int = 0) -> Array:
        """
        Args:
            c: Coefficients of Legendre series.
            k: Order of derivative to compute.
        Returns:
            Array (N,) of coefficients for the derivative of the series.
        """
        if k == 0:
            return c

        if k > 1:
            return self.derivative_coeffs(self.derivative_coeffs(c, k - 1), 1)

        N: int = c.shape[0] - 1
        x0: Array = jnp.array(0.0)
        x1: Array = c[-1] * (2 * N - 1)

        if N == 0:
            return x0

        if N == 1:
            return jnp.array([x1, x0])

        def inner_loop(
            carry: tuple[Array, Array], n: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = (2 * n + 1) * c[n + 1] + (2 * n + 1) / (2 * n + 5) * x0
            return (x1, x2), x2

        _, xs = jax.lax.scan(inner_loop, (x0, x1), jnp.arange(N - 2, -1, -1))
        return jnp.concatenate((xs[::-1], jnp.array([x1, x0])))

    legder = derivative_coeffs


def matrices(
    test: tuple[Legendre, int], trial: tuple[Legendre, int]
) -> sparse.BCOO | None:
    """Return sparse operator matrices for Legendre derivative coupling.

    Supported (i,j) derivative orders:
        (0,0): Mass (diagonal)
        (0,1): First derivative (odd shifts)
        (1,0): Transpose of (0,1)
        (0,2): Second derivative (even shifts)
        (2,0): Transpose of (0,2)

    Args:
        test: (space, derivative order) test side.
        trial: (space, derivative order) trial side.

    Returns:
        BCOO sparse matrix or None if unsupported.
    """
    import numpy as np
    from scipy import sparse as scipy_sparse

    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        return sparse.BCOO(
            (v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
            shape=(v.N, u.N),
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
        m = matrices(trial, test)
        if m is not None:
            return m.transpose()
        return None
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
        m = matrices(trial, test)
        if m is not None:
            return m.transpose()
    return None
