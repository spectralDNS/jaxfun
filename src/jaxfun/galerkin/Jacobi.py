from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from scipy.special import roots_jacobi
from sympy import Expr, Number, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap, n

from .orthogonal import OrthogonalSpace

alf, bet = sp.symbols("a,b", real=True)


class Jacobi(OrthogonalSpace):
    """Space of all Jacobi polynomials of order less than or equal to N"""

    def __init__(
        self,
        N: int,
        domain: Domain = None,
        system: CoordSys = None,
        name: str = "Jacobi",
        fun_str: str = "J",
        alpha: Number = 0,
        beta: Number = 0,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self.alpha = alpha
        self.beta = beta

    @jit_vmap(in_axes=(None, 0, None))
    def evaluate2(self, X: float | Array, c: Array) -> float | Array:
        """
        Evaluate a Jacobi series at points X.

        .. math:: p(X) = c_0 * P_0(X) + c_1 * P_1(X) + ... + c_n * P_n(X)

        Args:
            X: Evaluation point in reference space
            c: Expansion coefficients

        Returns:
            Jacobi series evaluated at X.
        """
        a, b = float(self.alpha), float(self.beta)

        if len(c) == 1:
            # Multiply by 0 * x for shape
            return c[0] + 0 * X
        if len(c) == 2:
            return c[0] + c[1] * ((a + 1) + (a + b + 2) * (X - 1) / 2)

        def body_fun(i: int, val: tuple[int, Array, Array]) -> tuple[int, Array, Array]:
            n, c0, c1 = val

            tmp = c0
            n = n - 1
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * X + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            c0 = c[-i] - c1 * bet / cn
            c1 = tmp + c1 * alf / cn

            return n, c0, c1

        n = len(c)
        c0 = jnp.ones_like(X) * c[-2]
        c1 = jnp.ones_like(X) * c[-1]

        _, c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body_fun, (n, c0, c1))
        return c0 + c1 * ((a + 1) + (a + b + 2) * (X - 1) / 2)

    @partial(jax.jit, static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.M if N == 0 else N
        return jnp.array(roots_jacobi(N, float(self.alpha), float(self.beta)))

    @jit_vmap(in_axes=(None, 0, None))
    def eval_basis_function(self, X: float | Array, i: int) -> float | Array:
        x0 = X * 0 + 1
        if i == 0:
            return x0

        a, b = float(self.alpha), float(self.beta)

        def body_fun(n: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * X + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            x2 = (x1 * alf - x0 * bet) / cn
            return x1, x2

        return jax.lax.fori_loop(2, i + 1, body_fun, (x0, X))[-1]

    @jit_vmap(in_axes=(None, 0))
    def eval_basis_functions(self, X: float | Array) -> Array:
        x0 = X * 0 + 1

        a, b = float(self.alpha), float(self.beta)

        def inner_loop(
            carry: tuple[float, float], n: int
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * X + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            x2 = (x1 * alf - x0 * bet) / cn
            return (x1, x2), x1

        _, xs = jax.lax.scan(
            inner_loop,
            (x0, a + 1 + (a + b + 2) * (X - 1) / 2),
            jnp.arange(2, self.N + 1),
        )

        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    def norm_squared(self) -> Array:
        return sp.lambdify(n, self.h(n, 0), modules="jax")(jnp.arange(self.N))

    @property
    def reference_domain(self) -> Domain:
        return Domain(-1, 1)

    def bnd_values(
        self, k: int = 0
    ) -> tuple[
        Callable[[int | sp.Symbol], Expr],
        Callable[[int | sp.Symbol], Expr],
    ]:
        """Return lambda function for computing boundary values"""
        alpha, beta = self.alpha, self.beta

        def gam(i: int) -> Expr:
            if k > 0:
                return sp.rf(i + alpha + beta + 1, k) * sp.Rational(1, 2 ** (k))
            return 1

        return (
            lambda i: self.gn(i)
            * (-1) ** (k + i)
            * gam(i)
            * sp.binomial(i + beta, i - k),
            lambda i: self.gn(i) * gam(i) * sp.binomial(i + alpha, i - k),
        )

    def psi(self, n: int, k: int) -> Expr:
        r"""Normalization factor for

        .. math::

            \partial^k P^{(\alpha, \beta)}_n = \psi^{(k,\alpha,\beta)}_{n} P^{(\alpha+k,\beta+k)}_{n-k}, \quad n \ge k, \quad (*)

        where :math:`\partial^k` represents the :math:`k`'th derivative

        Args:
            n, k (int) : Parameters in (*)

        """  # noqa: E501
        return sp.rf(n + self.alpha + self.beta + 1, k) / 2**k

    @staticmethod
    def gamma(alpha: Number, beta: Number, n: int) -> Expr:
        r"""Return normalization factor :math:`h_n` for inner product of Jacobi polynomials

        .. math::

            h_n = (P^{(\alpha,\beta)}_n, P^{(\alpha,\beta)}_n)_{\omega^{(\alpha,\beta)}}

        Args:
            alpha, beta (numbers) : Jacobi parameters
            n (int) : Index
        """  # noqa: E501
        f = (
            sp.rf(n + 1, alf)
            / sp.rf(n + bet + 1, alf)
            * 2 ** (alf + bet + 1)
            / (2 * n + alf + bet + 1)
        )
        return sp.simplify(f.subs([(alf, alpha), (bet, beta)]))

    def h(self, n: Number, k: int) -> Expr:
        r"""Return normalization factor :math:`h^{(k)}_n` for inner product of derivatives of Jacobi polynomials

        .. math::

            Q_n(x) = g_n(x)P^{(\alpha,\beta)}_n(x) \\
            h_n^{(k)} = (\partial^k Q_n, \partial^k Q_n)_{\omega^{(\alpha+k,\beta+k)}} \quad (*)

        where :math:`\partial^k` represents the :math:`k`'th derivative.

        Args:
            n (int) : Index
            k (int) : For derivative of k'th order, see (*)
        """  # noqa: E501
        f = self.gamma(self.alpha + k, self.beta + k, n - k) * (self.psi(n, k)) ** 2
        return sp.simplify(self.gn(n) ** 2 * f)

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    def gn(self, n: Symbol | Number) -> sp.Expr | Number:
        return 1


def matrices(test: tuple[Jacobi, int], trial: tuple[Jacobi, int]) -> Array:
    from jax.experimental import sparse

    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        # BCOO chops the array if v.N > u.N, so no need to check sizes
        return sparse.BCOO(
            (v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
            shape=(v.N, u.N),
        )
    return None
