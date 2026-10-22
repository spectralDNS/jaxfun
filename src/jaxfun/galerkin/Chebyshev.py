from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array
from sympy import Expr, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.la import DiaMatrix, Matrix, diags
from jaxfun.utils.common import (
    Domain,
    _dct_twiddle,
    _deinterleave,
    _idct_twiddles,
    _scaled_dct,
    cache_static,
    dct,
    jit_method,
    jit_vmap,
)

from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace

n = sp.Symbol("n", integer=True)


@jax.jit(static_argnums=(0, 1, 2))
def _dense_derivative_matrix_data(
    test_modes: int, trial_modes: int, derivative: int
) -> Array:
    """Return the dense Chebyshev derivative coupling (T_m, T_n^(derivative))_w."""
    rows = jnp.arange(test_modes)[:, None]
    cols = jnp.arange(trial_modes)[None, :]
    offsets = cols - rows
    mask = (offsets >= derivative) & ((offsets - derivative) % 2 == 0)
    if derivative == 1:
        return jnp.where(mask, jnp.pi * cols, 0.0)
    if derivative == 2:
        return jnp.where(mask, cols * (cols**2 - rows**2) * jnp.pi / 2, 0.0)
    # Higher orders from T_n^(j) = sum_m D^j[m, n] T_m, with T_n' = sum_m D[m, n]
    # T_m. D has no negative entries, so neither has D^j: the power is free of
    # the cancellation that makes quadrature lose about N^(2j-1) in accuracy.
    k = jnp.arange(trial_modes)
    d = k[None, :] - k[:, None]
    D = jnp.where((d > 0) & (d % 2 == 1), 2.0 * k[None, :], 0.0)
    D = D.at[0].multiply(0.5)
    Dj = jnp.linalg.matrix_power(D, derivative)
    if test_modes > trial_modes:
        Dj = jnp.pad(Dj, ((0, test_modes - trial_modes), (0, 0)))
    h = jnp.full((test_modes, 1), jnp.pi / 2).at[0].set(jnp.pi)
    return h * Dj[:test_modes]


def _dense_derivative_matrix(
    test_modes: int, trial_modes: int, derivative: int
) -> Matrix:
    return Matrix(_dense_derivative_matrix_data(test_modes, trial_modes, derivative))


@cache
def _chebyshev_backward_scales(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the idct input scalings with the backward transform's folded in."""
    fwd, rev = _idct_twiddles(N)
    sign = (-1.0) ** np.arange(N)
    return N * sign * fwd, N * sign * rev


@cache
def _chebyshev_forward_scale(N: int) -> np.ndarray:
    """Return W4 with the Chebyshev forward transform's scaling folded in."""
    c = (-1.0) ** np.arange(N) / N
    c[0] *= 0.5
    return _dct_twiddle(N) * c


@cache
def _chebyshev_scalar_product_scale(N: int, domain_factor: float) -> np.ndarray:
    """Return W4 with the Chebyshev scalar product's scaling folded in."""
    return _dct_twiddle(N) * (np.pi * (-1.0) ** np.arange(N) / (2 * N * domain_factor))


class Chebyshev(Jacobi):
    """Chebyshev (first kind) polynomial basis space.

    Implements a Chebyshev basis via the Jacobi formulation with
    alpha = beta = -1/2. Provides several evaluation kernels:
      * evaluate2: Clenshaw-like backward recurrence for series sum.
      * evaluate3: Forward recurrence accumulating T_n on the fly.
      * eval_basis_function: Single T_i evaluation (iterative).
      * eval_basis_functions: Vectorized generation of all modes < N.

    The series expansion (degree N-1):
        p(X) = sum_{k=0}^{N-1} c_k T_k(X)

    Args:
        N: Number of basis functions (polynomial order = N-1).
        domain: Physical interval (maps to reference [-1, 1]).
        system: Coordinate system (optional).
        name: Basis family name.
        fun_str: Symbol stem for basis functions (default "T").
        **kw: Extra keyword args passed to parent Jacobi constructor.
    """

    has_fast_transform = True

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
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

    @jit_vmap(in_axes=(0, None))
    def _evaluate2(self, X: float | Array, c: Array) -> Array:
        """Evaluate Chebyshev series using backward (Clenshaw-like) scheme.

        Uses a modified two-term recurrence sweeping coefficients from
        highest to lowest degree.

        Args:
            X: Evaluation points in reference domain [-1, 1].
            c: Coefficient array of length >= 1.

        Returns:
            Array of same shape as X with p(X) values.
        """
        if len(c) == 1:
            # Multiply by 0 * x for shape
            return c[0] + 0 * X
        if len(c) == 2:
            return c[0] + c[1] * X

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            c0, c1 = val

            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1 * 2 * X

            return c0, c1

        c0 = jnp.ones_like(X) * c[-2]
        c1 = jnp.ones_like(X) * c[-1]

        c0, c1 = jax.lax.fori_loop(3, len(c) + 1, body_fun, (c0, c1))
        return c0 + c1 * X

    @jit_vmap(in_axes=(0, None))
    def _evaluate3(self, X: float | Array, c: Array) -> Array:
        """Evaluate Chebyshev series via forward recurrence.

        Builds successive T_n(X) terms with a scan, accumulating
        contributions c_n T_n(X) except the final (handled separately).

        Args:
            X: Evaluation points in [-1, 1].
            c: Coefficient array length N (self.N expected).

        Returns:
            Series evaluation p(X) at each X.
        """
        X = jnp.asarray(X)
        x0 = jnp.ones_like(X)

        def inner_loop(
            carry: tuple[Array, Array], i: int | Array
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = 2 * X * x1 - x0
            return (x1, x2), x1 * c[i - 1]

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))

        return jnp.sum(xs, axis=0) + c[0]

    @cache_static
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        """Return Gauss-Chebyshev (first kind) nodes and weights.

        Nodes:
            x_k = cos(pi*(2k+1)/(2N)), k=0..N-1

        Weights:
            w_k = pi / N

        Args:
            N: Number of quadrature points (defaults self.num_quad_points
               if None).

        Returns:
            Tuple (x, w) of nodes and weights.
        """
        N = self.num_quad_points if N is None else N
        return (
            jnp.cos(jnp.pi + (2 * jnp.arange(N) + 1) * jnp.pi / (2 * N)),
            jnp.ones(N) * jnp.pi / N,
        )

    @cache_static
    def integration_points_and_weights(
        self, N: int | None = None
    ) -> tuple[Array, Array]:
        """Return Fejer's first rule: the Gauss-Chebyshev nodes, unweighted.

        With theta_k = (2k+1)pi/(2N), the weights are

            q_k = 2/N (1 - 2 sum_{j=1}^{N/2} cos(2j theta_k) / (4j^2 - 1)),

        a DCT-III. The rule is symmetric, so it does not matter that the points run from
        -1 to 1 here. See

          L. Fejer, Mechanische Quadraturen mit positiven Cotesschen Zahlen.
          Math. Z. 37:287-309, 1933. doi:10.1007/BF01474575

          J. Waldvogel, Fast construction of the Fejer and Clenshaw-Curtis
          quadrature rules. BIT 46:195-202, 2006. doi:10.1007/s10543-006-0045-4
        """
        N = self.num_quad_points if N is None else N
        # The j = N/2 term of an even N drops out, cos(N theta_k) being zero.
        j = np.arange(1, (N - 1) // 2 + 1)
        c = np.zeros(N)
        c[0] = 1
        c[2 * j] = -1 / (4 * j**2 - 1)
        return self.quad_points_and_weights(N)[0], 2 / N * dct(jnp.asarray(c), type=3)

    @jit_vmap(in_axes=(0, None), static_argnums=2)
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        """Evaluate single Chebyshev polynomial T_i at points X.

        Iterative two-term recurrence:
            T_0 = 1, T_1 = X,
            T_{n+1} = 2 X T_n - T_{n-1}

        Args:
            X: Points in [-1, 1].
            i: Basis index (0 <= i < N).

        Returns:
            Array of T_i(X).
        """
        X = jnp.asarray(X)
        x0 = X * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = 2 * X * x1 - x0
            return x1, x2

        return jax.lax.fori_loop(1, i, body_fun, (x0, X))[-1]

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        """Evaluate all basis functions T_0..T_{N-1} at X.

        Uses a scan to build the recurrence efficiently.

        Args:
            X: Points in [-1, 1].

        Returns:
            Array (N,) for each X containing T_k(X) stacked along axis 0.
        """
        X = jnp.asarray(X)
        x0 = X * 0 + 1

        def inner_loop(
            carry: tuple[Array, Array], _
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = 2 * X * x1 - x0
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, init=(x0, X), xs=None, length=self.N - 1)

        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    @jit_method(static_argnums=2)
    def backward(self, c: Array, N: int | None = None) -> Array:
        """Return Chebyshev series evaluated at quadrature points.

        Args:
            c: Coefficient array of length self.N.

        Returns:
            Reversed coefficient array.
        """
        n: int = self.num_quad_points if N is None else N
        fwd, rev = _chebyshev_backward_scales(n)
        if n > len(c):
            c = jnp.pad(c, (0, n - len(c)))
        G = c * fwd + jnp.roll(jnp.flip(c * rev, -1), 1, -1)
        out = _deinterleave(jnp.fft.ifft(G, axis=-1))
        if not jnp.iscomplexobj(c):
            out = out.real
        out = out + 0.5 * c[:1]
        return out

    @jit_method
    def forward(self, u: Array) -> Array:
        """Return Chebyshev coefficients for function values at quadrature points.

        Args:
            u: Function values at quadrature points.

        Returns:
            Coefficient array of length self.N.
        """
        n: int = u.shape[0]
        assert n >= self.N, "Only truncation supported for forward transform"
        uh = _scaled_dct(u, _chebyshev_forward_scale(n))
        return uh[: self.N] if n > self.N else uh

    def _scalar_product_ref(self, u: Array) -> Array:
        """Return the scalar product without the curvilinear measure.

        Args:
            u: Function values at quadrature points.

        Returns:
            Coefficient array of length self.N.
        """
        n: int = len(u)
        assert n >= self.N, "Only truncation supported for forward transform"
        scale = _chebyshev_scalar_product_scale(n, float(self.domain_factor))
        uh = _scaled_dct(u, scale)
        return uh[: self.N] if n > self.N else uh

    @jit_method(static_argnums=2)
    def derivative_coeffs(self, c: Array, k: int = 0) -> Array:
        """
        Args:
            c: Coefficients of Chebyshev series.
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
        x1: Array = c[-1] * N * 2
        if N == 1:
            # Index 0 is the one entry carrying the 1/2 that the tail below
            # applies, and for N == 1 that is where x1 lands.
            return jnp.array([x1 / 2, x0])

        # Old version using scan, which is slow on gpus because it launches a kernel
        # for each iteration.
        # def inner_loop(
        #     carry: tuple[Array, Array], n: int | Array
        # ) -> tuple[tuple[Array, Array], Array]:
        #     x0, x1 = carry
        #     x2 = 2 * (n + 1) * c[n + 1] + x0
        #     return (x1, x2), x2
        # xs = jax.lax.scan(inner_loop, (x0, x1), jnp.arange(N - 2, -1, -1))[1]
        # return jnp.concatenate(
        #     (jnp.array([xs[-1] / 2]), xs[-2::-1], jnp.array([x1, x0]))
        # )

        # The recurrence is `x[t] = a[t] + x[t-2]` -- lag two, and a coefficient
        # of exactly one -- so each parity class of `t` is a running total and
        # the whole thing is a cumulative sum. Written as a scan it is `N`
        # sequential steps, which on an accelerator is `N` kernel launches for a
        # handful of arithmetic each; as a cumsum it is logarithmic depth and
        # the same answer to round-off.
        # Note that a cumsum approach is also obtainable from a straightforward
        # Galerkin derivation, where the derivative matrix is appropriately diagonally
        # scaled.

        n = N - 1
        a = jnp.flip(2 * jnp.arange(1, N) * c[1:N])
        m = (n + 1) // 2
        a = jnp.pad(a, (0, 2 * m - n)).reshape(m, 2)
        xs = (jnp.cumsum(a, axis=0) + jnp.stack([x0, x1])).reshape(2 * m)[:n]
        return jnp.concatenate((xs[-1:] / 2, jnp.flip(xs[:-1]), jnp.stack([x1, x0])))

    chebder = derivative_coeffs

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    def gn(self, n: Symbol | int) -> Expr:
        """Return scaling g_n used in Jacobi-based normalization.

        Args:
            n: Polynomial index symbol.

        Returns:
            SymPy expression 1 / P_n^{(alpha,beta)}(1).
        """
        return sp.S.One / sp.jacobi(n, self.alpha, self.beta, 1)

    def h(self, n: Symbol | int, k: int) -> Expr:
        # Chebyshev has a weird limit behaviour for h that can only be reached
        # by substituting n=0 before fixing alpha, beta to -1/2. The resulting
        # piecewise implemented here is consistent with this limit of the general
        # formula for h.
        if k > 0:
            return sp.simplify(sp.pi * n * sp.gamma(n + k) / (2 * sp.factorial(n - k)))
        return sp.Piecewise((sp.pi, sp.Eq(n, 0)), (sp.pi / 2, True))

    def sympy_basis_function(self, i: int, X: Symbol) -> Expr:
        """Return symbolic Chebyshev polynomial T_i(X).

        Args:
            i: Basis function index (0 <= i < N).
            X: SymPy symbol (Reference coordinate in [-1, 1]).

        Returns:
            SymPy expression for T_i(X).
        """
        return sp.cos(i * sp.acos(X))

    def a(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        r"""Return Jacobi recurrence coefficient a_{i,j} for Chebyshev.

        Args:
            i: Row index symbol.
            j: Column index symbol.

        Returns:
            SymPy expression for a_{i,j}.
        """
        if (i - j) == 1:
            return sp.Piecewise((1, sp.Eq(j, 0)), (sp.S.Half, True))
        if (j - i) == 1:
            return sp.S.Half
        return 0

    def b(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        r"""Return Jacobi recurrence coefficient b_{i,j} for Chebyshev.

        For Chebyshev (alpha=beta=-1/2), b_{n,n} = 0 for all n.

        Args:
            i: Row index symbol.
            j: Column index symbol.

        Returns:
            SymPy expression for b_{i,j}.
        """
        if (i - j) == 1:
            return sp.Piecewise((1, sp.Eq(j, 0)), (1 / (2 * i), True))
        if (j - i) == 1:
            return sp.Piecewise((0, sp.Eq(i, 0)), (-1 / (2 * i), True))
        return 0

    def a_(self, i: Symbol | int, j: Symbol | int, k: int = 0) -> Expr | float:
        from .Ultraspherical import Ultraspherical

        U = Ultraspherical(0, domain=self.domain, system=self.system, lambda_=k)

        return (
            U._a(i, j)
            * self.psi(j + k, k)
            / self.psi(i + k, k)
            * self.gn(j + k)
            / self.gn(i + k)
        )

    def _matrices(
        self, i: int, trial: tuple[OrthogonalSpace, int], q: int = 0
    ) -> Matrix | DiaMatrix | None:
        """Return (possibly sparse) operator matrices between test/trial Chebyshev modes

        Constructs square sparse differentiation / mass-like
        matrices for combinations of test index i and trial index j flags:

            (i, j):
              (0,0): Diagonal mass-matrix.
              (0,j): j'th derivative, for any j > 0.
              (i,0): Transpose of (0,i).

        Args:
            i: Derivative order for test function.
            trial: Tuple (u, j) with trial space u and derivative order j.
            q: polynomial degree of coefficient.

        Returns:
            Matrix | DiaMatrix | None if combination unsupported.
        """
        u, j = trial
        if not isinstance(u, Chebyshev):
            return None

        if i != 0 and j != 0:
            return None

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
            # The power needs q modes to spare, since x**q links two modes
            # below N also through modes above it.
            A = self.A(self.N + q).power(q).crop(self.N, self.N)

        if i == 0 and j == 0:
            M = diags([self.norm_squared()], offsets=(0,), shape=(self.N, u.N))
            return M if A is None else A.T @ M

        if i > 0 and j == 0:
            m = u._matrices(j, (self, i), q=q)
            if m is not None:
                m = m.T
            return m

        if j >= u.N:
            return None

        M = _dense_derivative_matrix(self.N, u.N, j)
        return M if A is None else A.T @ M
