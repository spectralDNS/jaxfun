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


class Jacobi(OrthogonalSpace):
    """Jacobi polynomial space P_n^{(α,β)} for orders 0..N-1.

    Provides:
      * Series evaluation (Clenshaw-like backward recurrence)
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

    @jit_vmap(in_axes=(0, None))
    def evaluate2(self, X: float, c: Array) -> Array:
        """Evaluate Jacobi series sum_k c_k P_k^{(α,β)}(X) (backward scheme).

        Uses a stable backward recurrence accumulating two running
        quantities (c0, c1) similar to Clenshaw. Handles small len(c)
        explicitly for speed.

        Args:
            X: Points in reference domain [-1, 1].
            c: Coefficient array (length >= 1).

        Returns:
            Array of series values p(X) with shape like X.
        """
        a, b = float(self.alpha), float(self.beta)

        if len(c) == 1:
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

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> tuple[Array, Array]:
        """Return Gauss–Jacobi quadrature nodes/weights.

        Args:
            N: Number of points (0 -> self.num_quad_points).

        Returns:
            jnp.array((2, N)) first row nodes, second row weights.
        """
        N = self.num_quad_points if N == 0 else N
        x, w = roots_jacobi(N, float(self.alpha), float(self.beta))
        return jnp.array(x), jnp.array(w)

    @jit_vmap(in_axes=(0, None))
    def eval_basis_function(self, X: float, i: int) -> float:
        """Evaluate single Jacobi polynomial P_i^{(α,β)}(X).

        Iterative two-term recurrence:
            P_0 = 1, P_1 = ((α+1)+(α+β+2)(X-1)/2)
            P_{n+1} derived via standard Jacobi recurrence.

        Args:
            X: Points in [-1, 1].
            i: Basis index.

        Returns:
            Array of P_i^{(α,β)}(X).
        """
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

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float) -> Array:
        """Evaluate all Jacobi polynomials P_0..P_{N-1} at X.

        Args:
            X: Points in [-1, 1].

        Returns:
            Array (N,) per X containing stacked basis values.
        """
        x0 = X * 0 + 1
        a, b = float(self.alpha), float(self.beta)

        def inner_loop(
            carry: tuple[float, float], n: int
        ) -> tuple[tuple[float, float], float]:
            x0_, x1_ = carry
            alf = (2 * n + a + b - 1) * (
                (2 * n + a + b) * (2 * n + a + b - 2) * X + a**2 - b**2
            )
            bet = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)
            cn = 2 * n * (n + a + b) * (2 * n + a + b - 2)
            x2 = (x1_ * alf - x0_ * bet) / cn
            return (x1_, x2), x1_

        _, xs = jax.lax.scan(
            inner_loop,
            (x0, a + 1 + (a + b + 2) * (X - 1) / 2),
            jnp.arange(2, self.N + 1),
        )
        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    def norm_squared(self) -> Array:
        """Return L2 norms squared h_n^{(0)} for n=0..N-1."""
        return sp.lambdify(n, self.h(n, 0), modules="jax")(jnp.arange(self.N))

    @property
    def reference_domain(self) -> Domain:
        """Return reference domain [-1, 1]."""
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

        def gam(i: int) -> Expr | int:
            if k > 0:
                return sp.rf(i + alpha + beta + 1, k) * sp.Rational(1, 2**k)
            return 1

        return (
            lambda i: self.gn(i)
            * (-1) ** (k + i)
            * gam(i)
            * sp.binomial(i + beta, i - k),
            lambda i: self.gn(i) * gam(i) * sp.binomial(i + alpha, i - k),
        )

    def psi(self, n: int, k: int) -> Expr:
        r"""Return derivative normalization ψ^{(k,α,β)}_n.

        Relates k-th derivative to shifted-parameter Jacobi poly:

            ∂^k P_n^{(α,β)} = ψ^{(k,α,β)}_n P_{n-k}^{(α+k,β+k)},  n ≥ k.

        Args:
            n: Polynomial degree.
            k: Derivative order (0 ≤ k ≤ n).

        Returns:
            SymPy expression ψ^{(k,α,β)}_n.
        """
        return sp.rf(n + self.alpha + self.beta + 1, k) / 2**k

    @staticmethod
    def gamma(alpha: Number, beta: Number, n: int) -> Expr:
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

    def h(self, n: Number, k: int) -> Expr:
        r"""Return h_n^{(k)} norm for k-th derivative of scaled polynomials.

        Using Q_n = g_n P_n^{(α,β)}:
            h_n^{(k)} = (∂^k Q_n, ∂^k Q_n)_{ω^{(α+k,β+k)}}.

        Args:
            n: Degree.
            k: Derivative order.

        Returns:
            SymPy expression h_n^{(k)}.
        """
        f = self.gamma(self.alpha + k, self.beta + k, n - k) * (self.psi(n, k)) ** 2
        return sp.simplify(self.gn(n) ** 2 * f)

    def gn(self, n: Symbol | int) -> sp.Expr:
        """Scaling g_n used in alternative normalizations (default 1)."""
        return sp.S.One


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
