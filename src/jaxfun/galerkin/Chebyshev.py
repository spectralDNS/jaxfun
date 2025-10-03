from functools import partial

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from sympy import Expr, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap

from .Jacobi import Jacobi


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

    def __init__(
        self,
        N: int,
        domain: Domain = None,
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

    @jit_vmap(in_axes=(0, None))
    def evaluate2(self, X: float | Array, c: Array) -> Array:
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
    def evaluate3(self, X: float | Array, c: Array) -> Array:
        """Evaluate Chebyshev series via forward recurrence.

        Builds successive T_n(X) terms with a scan, accumulating
        contributions c_n T_n(X) except the final (handled separately).

        Args:
            X: Evaluation points in [-1, 1].
            c: Coefficient array length N (self.N expected).

        Returns:
            Series evaluation p(X) at each X.
        """
        x0 = jnp.ones_like(X)

        def inner_loop(
            carry: tuple[float, float], i: int
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            x2 = 2 * X * x1 - x0
            return (x1, x2), x1 * c[i - 1]

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))

        return jnp.sum(xs, axis=0) + c[0]

    @partial(jax.jit, static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> Array:
        """Return Gauss–Chebyshev (first kind) nodes and weights.

        Nodes:
            x_k = cos(pi*(2k+1)/(2N)), k=0..N-1
        Weights:
            w_k = pi / N

        Args:
            N: Number of quadrature points (defaults self.num_quad_points
               if 0).

        Returns:
            Array((2, N)) with first row nodes, second row weights.
        """
        N = self.num_quad_points if N == 0 else N
        return jnp.array(
            (
                jnp.cos(jnp.pi + (2 * jnp.arange(N) + 1) * jnp.pi / (2 * N)),
                jnp.ones(N) * jnp.pi / N,
            )
        )

    @jit_vmap(in_axes=(0, None))
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
        x0 = X * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = 2 * X * x1 - x0
            return x1, x2

        return jax.lax.fori_loop(1, i, body_fun, (x0, X))[-1]

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float) -> Array:
        """Evaluate all basis functions T_0..T_{N-1} at X.

        Uses a scan to build the recurrence efficiently.

        Args:
            X: Points in [-1, 1].

        Returns:
            Array (N,) for each X containing T_k(X) stacked along axis 0.
        """
        x0 = X * 0 + 1

        def inner_loop(
            carry: tuple[float, float], _
        ) -> tuple[tuple[float, float], Array]:
            x0, x1 = carry
            x2 = 2 * X * x1 - x0
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, init=(x0, X), xs=None, length=self.N - 1)

        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    def norm_squared(self) -> Array:
        """Return L2 norms squared over [-1, 1] with Chebyshev weight.

        For T_0: integral = pi
        For T_n (n>0): integral = pi/2

        Returns:
            Array of length N with norm^2 values.
        """
        return jnp.hstack((jnp.array([jnp.pi]), jnp.full(self.N - 1, jnp.pi / 2)))

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    def gn(self, n: Symbol) -> Expr:
        """Return scaling g_n used in Jacobi-based normalization.

        Args:
            n: Polynomial index symbol.

        Returns:
            SymPy expression 1 / P_n^{(alpha,beta)}(1).
        """
        return sp.S(1) / sp.jacobi(n, self.alpha, self.beta, 1)


def matrices(test: tuple[Chebyshev, int], trial: tuple[Chebyshev, int]) -> Array:
    """Sparse operator matrices between test/trial Chebyshev modes.

    Constructs (possibly rectangular) sparse differentiation / mass-like
    matrices for combinations of test index i and trial index j flags:

        (i, j):
          (0,0): Diagonal mass-like matrix (norm squares).
          (0,1): First derivative coupling (odd offsets).
          (1,0): Transpose of (0,1).
          (0,2): Second derivative coupling (even offsets).
          (2,0): Transpose of (0,2).

    Args:
        test: Tuple (v, i) with Chebyshev space v and number of derivatives i.
        trial: Tuple (u, j) with Chebyshev space u and number of derivatives j.

    Returns:
        jax.experimental.sparse.BCOO or None if combination unsupported.
    """
    import numpy as np
    from jax.experimental import sparse
    from scipy import sparse as scipy_sparse

    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        # BCOO chops the array if v.N > u.N, so no need to check sizes
        return sparse.BCOO(
            (v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
            shape=(v.N, u.N),
        )
    if i == 0 and j == 1:
        k = jnp.arange(max(v.N, u.N))

        def _getkey(j):
            Q = min(v.N, u.N - j)
            return np.pi * k[j : (Q + j)]

        d = dict.fromkeys(np.arange(1, u.N + 1, 2), _getkey)

        if len(d) == 0:
            return None

        return sparse.BCOO.from_scipy_sparse(
            scipy_sparse.diags(
                [d[i](i) for i in np.arange(1, u.N, 2)],
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
            return k[j : (Q + j)] * (k[j : (Q + j)] ** 2 - k[:Q] ** 2) * jnp.pi / 2

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
