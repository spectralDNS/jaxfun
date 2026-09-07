import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array
from sympy import Expr, Number, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.la import DiaMatrix, diags
from jaxfun.utils.common import Domain, cache_static

from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace


class Ultraspherical(Jacobi):
    r"""Ultraspherical (Gegenbauer) polynomial basis space.

    Implements an ultraspherical basis via the Jacobi formulation with
    alpha = beta = lambda - 1/2. Provides several evaluation kernels:
      * eval_basis_function: Single C^{(\lambda)}_i evaluation.
      * eval_basis_functions: Vectorized generation of all modes < N.

    The series expansion (degree N-1):
        p(X) = sum_{k=0}^{N-1} c_k C^{(\lambda)}_k(X)

    The ultraspherical polynomials are defined as

    .. math::
        C^{(\lambda)}_n(X) = g_n^{(\lambda-1/2)} P_n^{(\lambda-1/2, \lambda-1/2)}(X)

    where g_n^{(\lambda-1/2)} is a scaling factor that ensures C^{(\lambda)}_n(\pm 1)
    = (\pm 1)^n. Hence, g_n^{(\lambda-1/2)} = 1 / P_n^{(\lambda-1/2, \lambda-1/2)}(1).

    Args:
        N: Number of basis functions (polynomial order = N-1).
        domain: Physical interval (maps to reference [-1, 1]).
        system: Coordinate system (optional).
        name: Basis family name.
        fun_str: Symbol stem for basis functions (default "C").
        **kw: Extra keyword args passed to parent Jacobi constructor.
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Ultraspherical",
        fun_str: str = "C",
        lambda_: Number | float = 1,
        **kw,
    ) -> None:
        Jacobi.__init__(
            self,
            N,
            domain=domain,
            system=system,
            name=name,
            fun_str=fun_str,
            alpha=lambda_ - sp.S.Half,
            beta=lambda_ - sp.S.Half,
        )

    @property
    def lambda_(self):
        return self.alpha + sp.S.Half

    def h(self, n: Symbol | int, k: int) -> Expr:
        if self.lambda_ == 0:  # Chebyshev
            if k > 0:
                return sp.simplify(
                    sp.pi * n * sp.gamma(n + k) / (2 * sp.factorial(n - k))
                )
            return sp.Piecewise((sp.pi, sp.Eq(n, 0)), (sp.pi / 2, True))
        return super().h(n, k)

    def gn(self, n: Symbol | int) -> Expr:
        """Return scaling g_n used in Jacobi-based normalization.

        Args:
            n: Polynomial index symbol.

        Returns:
            SymPy expression 1 / P_n^{(alpha,beta)}(1).
        """
        return sp.S.One / sp.jacobi(n, self.alpha, self.beta, 1)

    def a(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        if self.lambda_ == 0:  # Chebyshev requires tweaking
            if (i - j) == 1:
                return sp.Piecewise((1, sp.Eq(j, 0)), (sp.S.Half, True))
            if (j - i) == 1:
                return sp.S.Half
            return 0
        return super().a(i, j)

    def b(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        if self.lambda_ == 0:  # Chebyshev requires tweaking
            if (i - j) == 1:
                return sp.Piecewise((1, sp.Eq(j, 0)), (1 / (2 * i), True))
            if (j - i) == 1:
                return -1 / (2 * i)
            return 0
        return super().b(i, j)

    @cache_static
    def _derivative_weights(self, N: int) -> tuple[Array, Array]:
        """Return the two multipliers of the derivative cumulative sum.

        Args:
            N: Highest mode index, i.e. `len(c) - 1`.

        Returns:
            `(1 / (b_- g), g)`, both of length N.
        """
        lam = float(self.lambda_)
        j = np.arange(N, dtype=np.float64)
        if lam == 0:  # Chebyshev, where g is the limit (1/2, 1, 1, ...)
            bm = 1.0 / (2 * (j + 1))
            bm[0] = 1.0  # the same removable 0/0 that `b` patches with Piecewise
            g = np.ones(N)
            g[0] = 0.5
        else:
            bm = (j + 2 * lam) / (2 * (j + 1) * (j + lam))
            i = j[1:]
            ratio = (i + lam) * (i + 2 * lam - 1) / ((i + lam - 1) * i)
            g = np.concatenate((np.ones(1), np.cumprod(ratio)))
            g /= g[N // 2]
        return jnp.asarray(1.0 / (bm * g)), jnp.asarray(g)

    @jax.jit(static_argnums=(0, 2))
    def derivative_coeffs(self, c: Array, k: int = 0) -> Array:
        """
        Args:
            c: Coefficients of ultraspherical series.
            k: Order of derivative to compute.

        Returns:
            Array (N,) of coefficients for the k'th derivative of the series.
        """
        if k == 0:
            return c

        if k > 1:
            return self.derivative_coeffs(self.derivative_coeffs(c, k - 1), 1)

        N: int = c.shape[0] - 1
        x0: Array = jnp.zeros((), dtype=c.dtype)
        if N == 0:
            return jnp.array([x0])

        # The generic Jacobi recurrence `x[n] = (c[n+1] - b_+[n] x[n+2]) / b_-[n]`
        # has, for alpha = beta = lambda - 1/2,
        #
        #     b_-[n] = (n + 2L) / (2 (n + 1)(n + L))
        #     b_+[n] = -(n + 2) / (2 (n + 2L + 1)(n + L + 2))
        #
        # writing L for lambda, and their ratio telescopes:
        # `-b_+[n] / b_-[n] = g[n] / g[n+2]` for `g[n] = (n + L) Gamma(n + 2L) /
        # Gamma(n + 1)`. So `w[n] = x[n] / g[n]` obeys `w[n] = c[n+1] / (b_-[n]
        # g[n]) + w[n+2]` -- lag two, and a coefficient of exactly one -- making
        # each parity class of `n` a running total, and the whole thing a
        # reversed cumulative sum. Written as a scan it is `N` sequential steps,
        # which on an accelerator is `N` kernel launches for a handful of
        # arithmetic each; as a cumsum it is logarithmic depth.
        #
        # `g` has to be built from the exact product of its own ratios: taking it
        # from `exp(gammaln(n + 2L) - gammaln(n + 1))` instead costs two to three
        # digits, since the two gammaln values are large and nearly equal.
        #
        # L = 0 is Chebyshev, where `b_-` has a removable 0/0 that `b` patches up
        # with a Piecewise. `_derivative_weights` hardcodes the limit instead, so
        # that case takes this path too rather than the inherited scan -- though
        # the Chebyshev class itself remains the one to reach for.

        inv_bg, g = self._derivative_weights(N)
        m = (N + 1) // 2
        a = jnp.pad(c[1:] * inv_bg, (0, 2 * m - N)).reshape(m, 2)
        w = jnp.flip(jnp.cumsum(jnp.flip(a, axis=0), axis=0), axis=0)
        return jnp.concatenate((g * w.reshape(2 * m)[:N], jnp.stack([x0])))

    def _matrices(
        self, i: int, trial: tuple[OrthogonalSpace, int], q: int = 0
    ) -> DiaMatrix | None:
        """Return sparse mass matrix for (i,j)=(0,0) else None.

        Args:
            i: Derivative order for test function.
            trial: (space, derivative order) for trial function.
            q: polynomial degree of coefficient.

        Returns:
            DiaMatrix diagonal mass matrix or None if derivative combo unsupported.
        """
        u, j = trial
        assert isinstance(u, Ultraspherical), (
            "Trial space must be Ultraspherical for Ultraspherical matrices"
        )
        A = None
        if q != 0:
            if self.N != trial[0].N:
                # x**q couples modes across the two ranges, and a recursion
                # matrix truncated to either range drops that coupling, so
                # fall back to quadrature for rectangular matrices.
                return None
            # Size by N rather than num_quad_points: a space may hold more
            # quadrature points than modes (a boundary space does), and A
            # has to match the shape of the matrices it multiplies below.
            A = self.A(self.N).power(q)
        if i == 0 and j == 0:
            M = diags([self.norm_squared()], offsets=(0,), shape=(self.N, u.N))
            return M if A is None else A.T @ M

        return None
