from __future__ import annotations

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.composite import PGComposite
from jaxfun.la import DiaMatrix, Matrix, diags
from jaxfun.utils.common import Domain, jit_vmap, n
from jaxfun.utils.fastgl import leggauss

from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace


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
        if N == 0:
            return jnp.array([x0])
        x1: Array = c[-1] * (2 * N - 1)
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
        self, i: int, trial: tuple[OrthogonalSpace, int], q: int = 0
    ) -> Matrix | DiaMatrix | None:
        """Return (sparse) operator matrices for Legendre derivative coupling.

        Supported (i,j) derivative orders:
            (0,0): Mass (diagonal)
            (0,1): First derivative (odd shifts)
            (1,0): Transpose of (0,1)
            (0,2): Second derivative (even shifts)
            (2,0): Transpose of (0,2)

        Args:
            i: Derivative order for test function.
            trial: (space, derivative order) for trial function.
            q: polynomial degree of coefficient.

        Returns:
            Matrix or DiaMatrix diagonal mass matrix or None if unsupported.
        """

        u, j = trial
        assert isinstance(u, Legendre), (
            "Trial space must be Legendre for Legendre matrices"
        )
        A = None
        if q != 0:
            A = self.A().power(q)

        if i == 0 and j == 0:
            M = diags([self.norm_squared()], offsets=(0,), shape=(self.N, u.N))
            return M if A is None else A.T @ M

        if i == 0 and j == 1:
            if u.N < 2:
                return None
            M = diags(
                [jnp.full(u.N - k, 2.0) for k in jnp.arange(1, u.N, 2).tolist()],
                offsets=tuple(jnp.arange(1, u.N, 2).tolist()),
                shape=(self.N, u.N),
            ).to_matrix()  # Matrix is upper triangular, better and faster to use dense.
            return M if A is None else A.T @ M

        if i == 1 and j == 0:
            m = u.matrices(j, (self, i), q=q)
            if m is not None:
                return m.T
            return None
        if i == 0 and j == 2:
            k = jnp.arange(max(self.N, u.N))
            offsets = jnp.arange(2, u.N, 2).tolist()
            if len(offsets) == 0:
                return None

            def _getkey(j):
                Q = min(self.N, u.N - j)
                return (
                    (k[:Q] + 0.5)
                    * (k[j : (Q + j)] * (k[j : (Q + j)] + 1) - k[:Q] * (k[:Q] + 1))
                    * 2.0
                    / (2 * k[:Q] + 1)
                )

            M = diags(
                [_getkey(j) for j in offsets],
                offsets=tuple(offsets),
                shape=(self.N, u.N),
            ).to_matrix()  # Matrix is upper triangular, better and faster to use dense.
            return M if A is None else A.T @ M

        if i == 2 and j == 0:
            m = u.matrices(j, (self, i), q=q)
            if m is not None:
                return m.T

        return None


### Predefined Composite spaces #####


class Phi_1(PGComposite):
    r"""
    Composite space for Mortensen's Petrov-Galerkin method order 1.

    The test functions are defined by

    .. math::

        \phi_k = \frac{1}{2}(L_k - L_{k+2}) = \frac{(2k+3)(1-x^2)}{2(k+1)(k+2)} L'_{k+1}

    where :math:`L'_{k+1}` is the derivative of Legendre polynomial k+1.

    When used as a test function space, the resulting inner product is a skewed
    diagonal matrix with entries

    .. math::
        \langle \phi_k, L'_j \rangle = \delta_{k,k+1}

    The advantage of using this space is that the resulting operator matrices are
    sparse, whereas a regular Legendre test space (Galerkin method) would yield
    dense operator matrices for the same problem.

    Examples:
        >>> from jaxfun.galerkin.Legendre import Phi_1
        >>> from jaxfun.galerkin import inner, Legendre, TrialFunction, TestFunction
        >>> from jaxfun.galerkin import FunctionSpace
        >>> N = 6
        >>> P = Phi_1(N)
        >>> V = Legendre.Legendre(N)
        >>> u = TrialFunction(V)
        >>> v = TestFunction(P)
        >>> x = V.system.x
        >>> A = inner(u.diff(x, 1) * v, sparse=True)
        >>> A.todense()
        Array([[0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 1.]], dtype=float32)
        >>> D = FunctionSpace(N, Legendre.Legendre, {"left": {"D": 0}})
        >>> x = D.system.x
        >>> u = TrialFunction(D)
        >>> A = inner(u.diff(x, 1) * v, sparse=True)
        >>> A.todense()
        Array([[1., 1., 0., 0., 0.],
               [0., 1., 1., 0., 0.],
               [0., 0., 1., 1., 0.],
               [0., 0., 0., 1., 1.],
               [0., 0., 0., 0., 1.]], dtype=float32)
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Phi_1",
        fun_str: str = "phi_1",
    ) -> None:
        PGComposite.__init__(
            self,
            N + 1,
            Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=domain,
            stencil={0: sp.S.Half, 2: -sp.S.Half},
            name=name,
            fun_str=fun_str,
            order=1,
        )


class Phi_2(PGComposite):
    r"""Composite space for Mortensen's Petrov-Galerkin method order 2.

    The test functions are defined by

    .. math::

        \phi_k &= \frac{(1-x^2)^2 L''_{k+2}}{h^{(2)}_{k+2}} \\
        h^{(2)}_{k+2} &= \int_{-1}^1 L''_{k+2} L''_{k+2} (1-x^2)^2 dx, \\
               &= \frac{2 (k+1)(k+2)(k+3)(k+4)}{2k+5},

    where :math:`L''_{k+2}` is the second derivative of Legendre polynomials k+2.

    When used as a test function space, the resulting inner product is a skewed
    diagonal matrix with entries

    .. math::
        \langle \phi_k, L''_j \rangle = \delta_{k,k+2}

    The advantage of using this space is that the resulting operator matrices are
    sparse, whereas a regular Legendre test space (Galerkin method) would yield
    dense operator matrices for the same problem.

    Examples:
        >>> from jaxfun.galerkin.Legendre import Phi_2
        >>> from jaxfun.galerkin import inner, Legendre, TrialFunction, TestFunction
        >>> from jaxfun.galerkin import FunctionSpace
        >>> N = 7
        >>> P = Phi_2(N)
        >>> V = Legendre.Legendre(N)
        >>> u = TrialFunction(V)
        >>> v = TestFunction(P)
        >>> x = V.system.x
        >>> A = inner(u.diff(x, 2) * v, sparse=True)
        >>> A.todense()
        Array([[0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 0., 1.]], dtype=float32)
        >>> D = FunctionSpace(
        ...     N, Legendre.Legendre, {"left": {"D": 0}, "right": {"D": 0}}
        ... )
        >>> x = D.system.x
        >>> u = TrialFunction(D)
        >>> A = inner(u.diff(x, 2) * v, sparse=True)
        >>> A.todense()
        Array([[-1.,  0.,  1.,  0.,  0.],
               [ 0., -1.,  0.,  1.,  0.],
               [ 0.,  0., -1.,  0.,  1.],
               [ 0.,  0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  0., -1.]], dtype=float32)
    """  # noqa: E501

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Phi_2",
        fun_str: str = "phi_2",
    ) -> None:
        PGComposite.__init__(
            self,
            N + 2,
            Legendre,
            bcs={"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}},
            domain=domain,
            stencil={
                0: 1 / (2 * (2 * n + 3)),
                2: -(2 * n + 5) / (2 * n + 7) / (2 * n + 3),
                4: 1 / (2 * (2 * n + 7)),
            },
            name=name,
            fun_str=fun_str,
            order=2,
        )


class Phi_4(PGComposite):
    r"""Composite space for Mortensen's Petrov-Galerkin method order 4.

    The test functions are defined by

    .. math::

        \phi_k &= \frac{(1-x^2)^4}{h^{(4)}_{k+4}} L^{(4)}_{k+4}, \\
        h^{(4)}_{k+4} &= \frac{2\Gamma(k+9)}{\Gamma(k+1)(2k+9)} = \int_{-1}^1 L^{(4)}_{k+4} L^{(4)}_{k+4} (1-x^2)^4 dx,

    where :math:`L^{(4)}_{k+4}` is the fourth derivative of Legendre polynomial
    k+4.

    When used as a test function space, the resulting inner product is a skewed
    diagonal matrix with entries

    .. math::
        \langle \phi_k, L^{(4)}_j \rangle = \delta_{k,k+4}

    The advantage of using this space is that the resulting operator matrices are
    sparse, whereas a regular Legendre test space (Galerkin method) would yield
    dense operator matrices for the same problem.

    Examples:
        >>> from jaxfun.galerkin.Legendre import Phi_4
        >>> from jaxfun.galerkin import inner, Legendre, TrialFunction, TestFunction
        >>> from jaxfun.galerkin import FunctionSpace
        >>> N = 9
        >>> P = Phi_4(N)
        >>> V = Legendre.Legendre(N)
        >>> u = TrialFunction(V)
        >>> v = TestFunction(P)
        >>> x = V.system.x
        >>> A = inner(u.diff(x, 4) * v, sparse=True)
        >>> A.todense()
        Array([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32)
        >>> D = FunctionSpace(
        ...     N,
        ...     Legendre.Legendre,
        ...     {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}},
        ... )
        >>> x = D.system.x
        >>> u = TrialFunction(D)
        >>> A = inner(u.diff(x, 4) * v, sparse=True)
        >>> A.todense()
        Array([[ 0.42857143,  0.        , -1.6363636 ,  0.        ,  1.        ],
               [ 0.        ,  0.5555556 ,  0.        , -1.6923077 ,  0.        ],
               [ 0.        ,  0.        ,  0.6363636 ,  0.        , -1.7333333 ],
               [ 0.        ,  0.        ,  0.        ,  0.6923077 ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.73333335]],      dtype=float32)
    """  # noqa: E501

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Phi_4",
        fun_str: str = "phi_4",
    ) -> None:
        PGComposite.__init__(
            self,
            N + 4,
            Legendre,
            bcs={
                "left": {"D": 0, "N": 0, "N2": 0, "N3": 0},
                "right": {"D": 0, "N": 0, "N2": 0, "N3": 0},
            },
            domain=domain,
            stencil={
                0: 1 / (2 * (8 * n**3 + 60 * n**2 + 142 * n + 105)),
                2: -2 / (8 * n**3 + 84 * n**2 + 262 * n + 231),
                4: 3
                * (2 * n + 9)
                / ((2 * n + 5) * (2 * n + 7) * (2 * n + 11) * (2 * n + 13)),
                6: -2 / (8 * n**3 + 132 * n**2 + 694 * n + 1155),
                8: 1 / (2 * (8 * n**3 + 156 * n**2 + 1006 * n + 2145)),
            },
            name=name,
            fun_str=fun_str,
            order=4,
        )
