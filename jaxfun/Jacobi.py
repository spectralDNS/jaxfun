from typing import Callable, Union
from functools import partial
import sympy as sp
from sympy import Number, Symbol, Expr
from scipy.special import roots_jacobi
from scipy import sparse as scipy_sparse
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO
from jaxfun.Basespace import BaseSpace, Domain, n
from jaxfun.coordinates import CoordSys

alf, bet = sp.symbols("a,b", real=True)

# ruff: noqa: F706


class Jacobi(BaseSpace):
    """Space of all Jacobi polynomials of order less than or equal to N"""

    def __init__(
        self,
        N: int,
        domain: Domain = Domain(-1, 1),
        coordinates: CoordSys = None,
        name: str = "Jacobi",
        fun_str: str = "J",
        alpha: Number = 0,
        beta: Number = 0,
    ) -> None:
        BaseSpace.__init__(self, N, domain, coordinates, name, fun_str)
        self.alpha = alpha
        self.beta = beta
        self.orthogonal = self
        self.stencil = {0: 1}
        self.S = BCOO.from_scipy_sparse(scipy_sparse.diags((1,), (0,), (N + 1, N + 1)))

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    def gn(self, n: Symbol | Number):
        return 1

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        """
        Evaluate a Jacobi series at points x.

        .. math:: p(x) = c_0 * P_0(x) + c_1 * P_1(x) + ... + c_n * P_n(x)

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
        a, b = self.alpha, self.beta
        if len(c) == 1:
            # Multiply by 0 * x for shape
            return c[0] + 0 * x
        if len(c) == 2:
            return c[0] + c[1] * ((a + 1) + (a + b + 2) * (x - 1) / 2)

        def body_fun(i: int, val: tuple[int, Array, Array]) -> tuple[int, Array, Array]:
            n, c0, c1 = val

            tmp = c0
            n = n - 1
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * x + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            c0 = c[-i] - c1 * bet / cn
            c1 = tmp + c1 * alf / cn

            return n, c0, c1

        n = len(c)
        c0 = jnp.ones_like(x) * c[-2]
        c1 = jnp.ones_like(x) * c[-1]

        _, c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body_fun, (n, c0, c1))
        return c0 + c1 * ((a + 1) + (a + b + 2) * (x - 1) / 2)

    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N + 1 if N == 0 else N + 1
        return jnp.array(roots_jacobi(N, self.alpha, self.beta))

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        x0 = x * 0 + 1
        if i == 0:
            return x0

        a, b = self.alpha, self.beta

        def body_fun(n: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * x + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            x2 = (x1 * x * alf - x0 * bet) / cn
            return x1, x2

        return jax.lax.fori_loop(1, i, body_fun, (x0, x))[-1]

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        x0 = x * 0 + 1

        a, b = self.alpha, self.beta

        def inner_loop(
            carry: tuple[float, float], n: int
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * x + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            x2 = (x1 * x * alf - x0 * bet) / cn
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, (x0, x), jnp.arange(2, self.N + 2))

        return jnp.hstack((x0, xs))

    def norm_squared(self) -> Array:
        return sp.lambdify(n, self.h(n, 0, self.gn), modules="jax")(
            jnp.arange(self.N + 1)
        )

    def mass_matrix(self) -> BCOO:
        return BCOO.from_scipy_sparse(
            scipy_sparse.diags(
                (self.norm_squared(),), (0,), shape=(self.N + 1, self.N + 1)
            )
        )

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_galerkin(self, b: Array) -> Array:
        return b

    @partial(jax.jit, static_argnums=0)
    def apply_stencils_petrovgalerkin(self, b: Array, P: BCOO) -> Array:
        return b @ P.T

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_left(self, b: Array) -> Array:
        return b

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_right(self, a: Array) -> Array:
        return a

    @property
    def reference_domain(self) -> Domain:
        return Domain(-1, 1)

    def bnd_values(
        self, k: int = 0
    ) -> tuple[
        Callable[[Union[int, sp.Symbol]], Expr],
        Callable[[Union[int, sp.Symbol]], Expr],
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

        Parameters
        ----------
        n, k : int
            Parameters in (*)
        """
        return sp.rf(n + self.alpha + self.beta + 1, k) / 2**k

    @staticmethod
    def gamma(alpha: Number, beta: Number, n: int) -> Expr:
        r"""Return normalization factor :math:`h_n` for inner product of Jacobi polynomials

        .. math::

            h_n = (P^{(\alpha,\beta)}_n, P^{(\alpha,\beta)}_n)_{\omega^{(\alpha,\beta)}}

        Parameters
        ----------
        alpha, beta : numbers
            Jacobi parameters
        n : int
            Index
        """
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

        Parameters
        ----------
        n : int
            Index
        k : int
            For derivative of k'th order, see (*)
        """
        f = self.gamma(self.alpha + k, self.beta + k, n - k) * (self.psi(n, k)) ** 2
        return sp.simplify(self.gn(n) ** 2 * f)
