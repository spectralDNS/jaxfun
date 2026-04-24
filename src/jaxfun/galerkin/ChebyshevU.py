import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from sympy import Expr, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.la import DiaMatrix, diags
from jaxfun.typing import MeshKind
from jaxfun.utils.common import Domain, dst, jit_vmap

from .Jacobi import Jacobi


class ChebyshevU(Jacobi):
    """Chebyshev (second kind) polynomial basis space.

    Implements a Chebyshev basis via the Jacobi formulation with
    alpha = beta = 1/2. Provides several evaluation kernels:
      * eval_basis_function: Single U_i(x) evaluation (iterative).
      * eval_basis_functions: Vectorized generation of all modes < N.

    The series expansion (degree N-1):
        p(X) = sum_{k=0}^{N-1} c_k U_k(X)

    Args:
        N: Number of basis functions (polynomial order = N-1).
        domain: Physical interval (maps to reference [-1, 1]).
        system: Coordinate system (optional).
        name: Basis family name.
        fun_str: Symbol stem for basis functions (default "U").
        **kw: Extra keyword args passed to parent Jacobi constructor.
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "ChebyshevU",
        fun_str: str = "U",
        **kw,
    ) -> None:
        Jacobi.__init__(
            self,
            N,
            domain=domain,
            system=system,
            name=name,
            fun_str=fun_str,
            alpha=sp.S.Half,
            beta=sp.S.Half,
        )

    @jit_vmap(in_axes=(0, None))
    def _evaluate2(self, X: float, c: Array) -> Array:
        """Evaluate Chebyshev U series via forward recurrence.

        Builds successive U_n(X) terms with a scan, accumulating
        contributions c_n U_n(X) except the final (handled separately).

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

        _, xs = jax.lax.scan(inner_loop, (x0, 2 * X), jnp.arange(2, self.N + 1))

        return jnp.sum(xs, axis=0) + c[0]

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        """Return Gauss-Chebyshev (second kind) nodes and weights.

        Nodes:
            x_k = cos((k + 1) * pi / (N + 1) + pi), k=0..N-1

        Weights:
            w_k = pi / (N + 1) * (1 - x_k^2)

        Args:
            N: Number of quadrature points (defaults self.num_quad_points
               if 0).

        Returns:
            Tuple (x, w) of nodes and weights.
        """
        N = self.num_quad_points if N is None else N
        theta = (jnp.arange(N) + 1) * jnp.pi / (N + 1)
        points = jnp.cos(theta + jnp.pi)
        weights = jnp.full(N, jnp.pi / (N + 1)) * (1 - points**2)
        return points, weights

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: Array, i: int) -> Array:
        """Evaluate single Chebyshev polynomial U_i at points X.

        Iterative two-term recurrence:
            U_0 = 1, U_1 = 2X,
            U_{n+1} = 2 X U_n - U_{n-1}

        Args:
            X: Points in [-1, 1].
            i: Basis index (0 <= i < N).

        Returns:
            Array of U_i(X).
        """
        x0 = X * 0 + 1
        if i == 0:
            return x0

        def body_fun(i: int, val: tuple[Array, Array]) -> tuple[Array, Array]:
            x0, x1 = val
            x2 = 2 * X * x1 - x0
            return x1, x2

        return jax.lax.fori_loop(1, i, body_fun, (x0, 2 * X))[-1]

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float) -> Array:
        """Evaluate all basis functions U_0..U_{N-1} at X.

        Uses a scan to build the recurrence efficiently.

        Args:
            X: Points in [-1, 1].

        Returns:
            Array (N,) for each X containing U_k(X) stacked along axis 0.
        """
        x0 = X * 0 + 1

        def inner_loop(
            carry: tuple[float, float], _
        ) -> tuple[tuple[float, float], float]:
            x0, x1 = carry
            x2 = 2 * X * x1 - x0
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, init=(x0, 2 * X), xs=None, length=self.N - 1)

        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    def norm_squared(self) -> Array:
        return jnp.full(self.N, jnp.pi / 2)

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(
        self, c: Array, kind: MeshKind = MeshKind.QUADRATURE, N: int | None = None
    ) -> Array:
        """Return Chebyshev series U_k evaluated at quadrature points.

        Args:
            c: Coefficient array of length self.N.

        Returns:
            Reversed coefficient array.
        """
        n: int = self.num_quad_points if N is None else N

        if MeshKind(kind) is not MeshKind.QUADRATURE:
            return super().backward(c, kind=kind, N=n)  # Does not require padding of c

        if n > len(c):
            c = jnp.pad(c, (0, n - len(c)))
        d = dst(c, n=n, type=1)
        return (d / (2 * jnp.sin((jnp.arange(n) + 1) * jnp.pi / (n + 1))))[::-1]

    @jax.jit(static_argnums=0)
    def forward(self, u: Array) -> Array:
        """Return Chebyshev coefficients for function values at quadrature points.

        Args:
            u: Function values at quadrature points.

        Returns:
            Coefficient array of length self.N.
        """
        uh = self.scalar_product(u)
        return uh * (2 * self.domain_factor / jnp.pi)

    @jax.jit(static_argnums=0)
    def scalar_product(self, u: Array) -> Array:
        """Return scalar product for function u.

        Args:
            u: Function values at quadrature points.

        Returns:
            Coefficient array of length self.N.
        """
        n: int = len(u)
        assert len(u) >= self.N, "Only truncation supported for forward transform"
        uh = u * jnp.sin(jnp.pi / (n + 1) * jnp.arange(1, n + 1))
        uh = dst(uh, n=n, type=1)
        uh = uh * (-1) ** jnp.arange(n) * jnp.pi / (2 * (n + 1) * self.domain_factor)
        if len(u) > self.N:
            uh = uh[: self.N]
        return uh

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    def gn(self, n: Symbol | int) -> Expr:
        """Return scaling g_n used in Jacobi-based normalization.

        Args:
            n: Polynomial index symbol.

        Returns:
            SymPy expression (n + 1) / P_n^{(alpha,beta)}(1).
        """
        return (n + 1) / sp.jacobi(n, self.alpha, self.beta, 1)


def matrices(
    test: tuple[ChebyshevU, int], trial: tuple[ChebyshevU, int]
) -> DiaMatrix | None:
    """Sparse operator matrices between test/trial ChebyshevU modes.

    Constructs (possibly rectangular) sparse differentiation / mass-like
    matrices for combinations of test index i and trial index j flags:

        (i, j):
          (0,0): Diagonal mass-matrix.

    Args:
        test: Tuple (v, i) with ChebyshevU space v and number of derivatives i.
        trial: Tuple (u, j) with ChebyshevU space u and number of derivatives j.

    Returns:
        DiaMatrix or None if combination unsupported.
    """
    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        return diags([v.norm_squared()], offsets=(0,), shape=(v.N, u.N))

    return None
