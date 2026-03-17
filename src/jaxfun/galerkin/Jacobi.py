from collections.abc import Callable

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental import sparse
from scipy.special import roots_jacobi
from sympy import Expr, Number, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap, n

from .orthogonal import OrthogonalSpace

alf, bet = sp.symbols("a,b", real=True)
delta = sp.KroneckerDelta


class Jacobi(OrthogonalSpace):
    """Jacobi polynomial space P_n^{(α,β)} for orders 0..N-1.

    Provides:
      * Series evaluation via recursion.
      * Single / all basis function evaluation
      * Quadrature nodes/weights (Gauss-Jacobi)
      * Norm / derivative normalization factors

    Args:
        N: Number of modes (max degree N-1).
        domain: Physical interval (default [-1, 1]).
        system: Coordinate system (optional).
        name: Space name.
        fun_str: Symbol stem for basis functions.
        alpha: Jacobi α parameter (>-1 for orthogonality).
        beta: Jacobi β parameter (>-1 for orthogonality).

    Attributes:
        alpha: Jacobi α.
        beta: Jacobi β.
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Jacobi",
        fun_str: str = "J",
        alpha: Number | float = 0,
        beta: Number | float = 0,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self.alpha: Number | float = alpha
        self.beta: Number | float = beta

    @jit_vmap(in_axes=(0, None))
    def _evaluate(self, X: float | Array, c: Array) -> Array:
        """Evaluate Jacobi series sum_k c_k P_k^{(α,β)}(X)

        Uses a regular forward recursion. Handles small len(c)
        explicitly for speed.

        Args:
            X: Points in reference domain [-1, 1].
            c: Coefficient array (length >= 1).

        Returns:
            Array of series values p(X) with shape like X.
        """
        N: int = c.shape[0]

        am = sp.lambdify(n, self.a(n + 1, n), modules="jax")(jnp.arange(N))
        ap = sp.lambdify(n, self.a(n, n + 1), modules="jax")(jnp.arange(N))
        if isinstance(ap, float | int):
            ap = jnp.full(N, ap)
        if isinstance(am, float | int):
            am = jnp.full(N, am)
        aa = jnp.zeros_like(am)
        if self.alpha != self.beta:
            aa = sp.lambdify(n, self.a(n, n), modules="jax")(jnp.arange(N))

        x0 = jnp.ones_like(X)

        if N == 1:
            return c[0] * x0

        x1 = (X - aa[0]) / am[0] * x0

        if N == 2:
            return c[0] * x0 + c[1] * x1

        def inner_loop(
            carry: tuple[Array, Array], i: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = ((X - aa[i - 1]) * x1 - ap[i - 2] * x0) / am[i - 1]
            return (x1, x2), x2 * c[i]

        _, xs = jax.lax.scan(inner_loop, (x0, x1), jnp.arange(2, N))

        return jnp.sum(xs, axis=0) + c[0] + c[1] * x1

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        """Return Gauss-Jacobi quadrature nodes/weights.

        Args:
            N: Number of points (None -> self.num_quad_points).

        Returns:
            Tuple (x, w) of nodes and weights.
        """
        N = self.num_quad_points if N is None else N
        x, w = roots_jacobi(N, float(self.alpha), float(self.beta))
        return jnp.array(x), jnp.array(w)

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: Array, i: int) -> Array:
        """Evaluate single (possibly scaled) Jacobi polynomial

        .. math::
            Q_i(X) = g_i^{(α,β)} * P_i^{(α,β)}(X)

        The scale defaults to 1 for regular Jacobi polynomials
        but can be overridden by defining gn(i) for the space.

        Args:
            X: Points in [-1, 1].
            i: Basis index.

        Returns:
            Array of Q_i(X) = g_i^{(α,β)} * P_i^{(α,β)}(X).
        """
        x0 = X * 0 + 1
        if i == 0:
            return x0

        am = sp.lambdify(n, self.a(n + 1, n), modules="jax")(jnp.arange(i))
        ap = sp.lambdify(n, self.a(n, n + 1), modules="jax")(jnp.arange(i))
        if isinstance(ap, float | int):
            ap = jnp.full(i, ap)
        if isinstance(am, float | int):
            am = jnp.full(i, am)
        aa = jnp.zeros_like(am)
        if self.alpha != self.beta:
            aa = sp.lambdify(n, self.a(n, n), modules="jax")(jnp.arange(i))

        def body_fun(n: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = ((X - aa[n - 1]) * x1 - ap[n - 2] * x0) / am[n - 1]
            return x1, x2

        return jax.lax.fori_loop(2, i + 1, body_fun, (x0, X))[-1]

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: Array) -> Array:
        """Evaluate all (possibly scaled) Jacobi polynomials P_0..P_{N-1} at X.

        Args:
            X: Points in [-1, 1].

        Returns:
            Array (N,) per X containing stacked basis values.
        """
        x0 = X * 0 + 1

        am = sp.lambdify(n, self.a(n + 1, n), modules="jax")(jnp.arange(self.N))
        ap = sp.lambdify(n, self.a(n, n + 1), modules="jax")(jnp.arange(self.N))
        if isinstance(ap, float | int):
            ap = jnp.full(self.N, ap)
        if isinstance(am, float | int):
            am = jnp.full(self.N, am)
        aa = jnp.zeros_like(am)
        if self.alpha != self.beta:
            aa = sp.lambdify(n, self.a(n, n), modules="jax")(jnp.arange(self.N))

        x1 = (X - aa[0]) / am[0] * x0

        def inner_loop(
            carry: tuple[Array, Array], n: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = ((X - aa[n - 1]) * x1 - ap[n - 2] * x0) / am[n - 1]
            return (x1, x2), x1

        _, xs = jax.lax.scan(
            inner_loop,
            (x0, x1),
            jnp.arange(2, self.N + 1),
        )
        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    @jax.jit(static_argnums=(0, 2))
    def derivative_coeffs(self, c: Array, k: int = 0) -> Array:
        """
        Args:
            c: Coefficients of Jacobi series.
            k: Order of derivative to compute.

        Returns:
            Array (N,) of coefficients for the k'th derivative of the series.
        """
        if k == 0:
            return c

        if k > 1:
            return self.derivative_coeffs(self.derivative_coeffs(c, k - 1), 1)

        N: int = c.shape[0] - 1

        bm = sp.lambdify(n, self.b(n + 1, n), modules="jax")(jnp.arange(N))
        bp = sp.lambdify(n, self.b(n + 1, n + 2), modules="jax")(jnp.arange(N))
        if isinstance(bp, float | int):
            bp = jnp.full(N, bp)
        if isinstance(bm, float | int):
            bm = jnp.full(N, bm)
        bb = jnp.zeros_like(bm)
        if self.alpha != self.beta:
            bb = sp.lambdify(n, self.b(n + 1, n + 1), modules="jax")(jnp.arange(N))

        x0: Array = jnp.array(0.0)
        x1: Array = c[-1] / bm[-1]

        def inner_loop(
            carry: tuple[Array, Array], n: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = (c[n + 1] - bb[n] * x1 - bp[n] * x0) / bm[n]
            return (x1, x2), x2

        _, xs = jax.lax.scan(inner_loop, (x0, x1), jnp.arange(N - 2, -1, -1))
        return jnp.concatenate((xs[::-1], jnp.array([x1, x0])))

    def norm_squared(self) -> Array:
        """Return L2 norms squared h_n^{(0)} for n=0..N-1."""
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
        """Return callables evaluating k-th derivative boundary traces.

        Returned tuple (left_fun, right_fun) where each maps index i to
        ∂^k P_i^{(α,β)} evaluated at x=-1 (left) or x=1 (right) including
        normalizing factors g_n and derivative scaling gam(i).

        Args:
            k: Derivative order (0 => function values).

        Returns:
            (left, right) callable pair.
        """
        alpha, beta = self.alpha, self.beta

        def gam(i: int | sp.Symbol) -> Expr | int:
            if k > 0:
                return self.psi(i, k)
            return 1

        def left_fn(i: int | sp.Symbol) -> Expr:
            return self.gn(i) * (-1) ** (k + i) * gam(i) * sp.binomial(i + beta, i - k)

        def right_fn(i: int | sp.Symbol) -> Expr:
            return self.gn(i) * gam(i) * sp.binomial(i + alpha, i - k)

        return left_fn, right_fn

    def psi(self, n: Symbol | int, k: int) -> Expr:
        r"""Return derivative normalization ψ^{(k,α,β)}_n.

        Relates k-th derivative to shifted-parameter Jacobi poly:

            ∂^k P_n^{(α,β)} = ψ^{(k,α,β)}_n P_{n-k}^{(α+k,β+k)},  n ≥ k.

        Args:
            n: Polynomial degree.
            k: Derivative order (0 ≤ k ≤ n).

        Returns:
            SymPy expression ψ^{(k,α,β)}_n.
        """
        return sp.rf(n + self.alpha + self.beta + 1, k) * sp.Rational(1, 2**k)

    @staticmethod
    def h0(alpha: Expr | float, beta: Expr | float, n: int) -> Expr:
        r"""Return h_n (norm squared) for P_n^{(α,β)} under weight ω^{(α,β)}.

        h_n = (P_n^{(α,β)}, P_n^{(α,β)})_{ω^{(α,β)}}.

        Args:
            alpha: Jacobi α parameter.
            beta: Jacobi β parameter.
            n: Degree.

        Returns:
            SymPy expression h_n.
        """
        f = (
            sp.rf(n + 1, alf)
            / sp.rf(n + bet + 1, alf)
            * 2 ** (alf + bet + 1)
            / (2 * n + alf + bet + 1)
        )
        return sp.simplify(f.subs([(alf, alpha), (bet, beta)]))

    def h(self, n: Symbol | int, k: int) -> Expr:
        r"""Return h_n^{(k)} norm for k-th derivative of normalized polynomials.

        Using Q_n = g_n P_n^{(α,β)}:
            h_n^{(k)} = (∂^k Q_n, ∂^k Q_n)_{ω^{(α+k,β+k)}}.

        Args:
            n: Degree.
            k: Derivative order.

        Returns:
            SymPy expression h_n^{(k,α,β)}.
        """
        f = self.h0(self.alpha + k, self.beta + k, n - k) * (self.psi(n, k)) ** 2
        return sp.simplify(self.gn(n) ** 2 * f)

    def gn(self, n: Symbol | int) -> Expr:
        """Return scaling g_n used for normalized Jacobi polynomials.

        The scaling is used for polynomials defined as

        .. math::
            Q_n^{(α,β)}(X) = g_n^{(α,β)} P_n^{(α,β)}(X)

        where P^{(α,β)}_n is the standard Jacobi polynomial.

        Args:
            n: Polynomial index symbol.
        """
        return sp.S.One

    def _a(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        """Matrix A for non-normalized Jacobi polynomials"""
        a, b = self.alpha, self.beta
        return (
            2
            * (j + a)
            * (j + b)
            / ((2 * j + a + b + 1) * (2 * j + a + b))
            * delta(i + 1, j)
            - (a**2 - b**2) / ((2 * j + a + b + 2) * (2 * j + a + b)) * delta(i, j)
            + 2
            * (j + 1)
            * (j + a + b + 1)
            / ((2 * j + a + b + 2) * (2 * j + a + b + 1))
            * delta(i - 1, j)
        )

    def _b(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        """Matrix B for non-normalized Jacobi polynomials"""
        a, b = self.alpha, self.beta
        delta = lambda m, n: int(m == n)

        f = (2 * (i + a + b) / ((2 * i + a + b) * (2 * i + a + b - 1))) * delta(
            i, j + 1
        ) - (
            (2 * (i + a + 1) * (i + b + 1))
            / ((2 * i + a + b + 3) * (2 * i + a + b + 2) * (i + a + b + 1))
        ) * delta(i, j - 1)
        if a != b:
            f += (
                (2 * (a**2 - b**2)) / ((a + b) * (2 * i + a + b + 2) * (2 * i + a + b))
            ) * delta(i, j)
        return f

    def b(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        r"""Recursion matrix :math:`B` for normalized Jacobi polynomials

        The recursion is

        .. math::

            \boldsymbol{Q} = {B}^T \partial \boldsymbol{Q}

        where :math:`\partial` represents the derivative and

        .. math::

            Q_n(x) = g_n^{(\alpha,\beta)} P^{(\alpha,\beta)}_n(x) \\
            \boldsymbol{Q} = (Q_0, Q_1, \ldots, Q_{N})^T

        Parameters
        ----------
        i, j : int
            Indices for row and column

        """
        f = self._b(i, j)
        factor = self.gn(j) / self.gn(i)
        if isinstance(i, sp.Basic) or isinstance(j, sp.Basic):
            return sp.simplify(factor * f)
        return factor * f

    def a(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        r"""Recursion matrix :math:`A` for normalized Jacobi polynomials

        The recursion is

        .. math::

            x \boldsymbol{Q} = {A}^T \boldsymbol{Q}

        where

        .. math::

            Q_n(x) = g_n^{(\alpha,\beta)} P^{(\alpha,\beta)}_n(x) \\
            \boldsymbol{Q} = (Q_0, Q_1, \ldots, Q_{N})^T

        Parameters
        ----------
        i, j : int
            Indices for row and column
        """

        f = self._a(i, j)
        factor = self.gn(j) / self.gn(i)
        if isinstance(i, sp.Basic) or isinstance(j, sp.Basic):
            return sp.simplify(factor * f)
        return factor * f


def matrices(test: tuple[Jacobi, int], trial: tuple[Jacobi, int]) -> sparse.BCOO | None:
    """Return sparse mass matrix for (i,j)=(0,0) else None.

    Args:
        test: (space, derivative order) for test function.
        trial: (space, derivative order) for trial function.

    Returns:
        BCOO diagonal mass matrix or None if derivative combo unsupported.
    """
    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        return sparse.BCOO(
            (v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
            shape=(v.N, u.N),
        )
    return None
