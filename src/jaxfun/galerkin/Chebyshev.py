from __future__ import annotations

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from sympy import Expr, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.composite import PGComposite
from jaxfun.la import DiaMatrix, Matrix, diags
from jaxfun.typing import MeshKind
from jaxfun.utils.common import Domain, jit_vmap

from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace

n = sp.Symbol("n", integer=True)


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
    def _evaluate3(self, X: float, c: Array) -> Array:
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

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        """Return Gauss-Chebyshev (first kind) nodes and weights.

        Nodes:
            x_k = cos(pi*(2k+1)/(2N)), k=0..N-1

        Weights:
            w_k = pi / N

        Args:
            N: Number of quadrature points (defaults self.num_quad_points
               if 0).

        Returns:
            Tuple (x, w) of nodes and weights.
        """
        N = self.num_quad_points if N is None else N
        return (
            jnp.cos(jnp.pi + (2 * jnp.arange(N) + 1) * jnp.pi / (2 * N)),
            jnp.ones(N) * jnp.pi / N,
        )

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: Array, i: int) -> Array:
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
        ) -> tuple[tuple[float, float], float]:
            x0, x1 = carry
            x2 = 2 * X * x1 - x0
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, init=(x0, X), xs=None, length=self.N - 1)

        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(
        self, c: Array, kind: MeshKind = MeshKind.QUADRATURE, N: int | None = None
    ) -> Array:
        """Return Chebyshev series evaluated at quadrature points.

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
        sign = (-1) ** jnp.arange(n)
        uh = c * sign
        return 0.5 * uh[0] + n * jax.scipy.fft.idct(uh, n=n)

    @jax.jit(static_argnums=0)
    def forward(self, u: Array) -> Array:
        """Return Chebyshev coefficients for function values at quadrature points.

        Args:
            u: Function values at quadrature points.

        Returns:
            Coefficient array of length self.N.
        """
        n: int = u.shape[0]
        assert n >= self.N, "Only truncation supported for forward transform"
        sign = (-1) ** jnp.arange(n)
        uh = jax.scipy.fft.dct(u, n=n)
        uh = uh.at[0].set(uh[0] / 2) * sign / n
        if n > self.N:
            uh = uh[: self.N]
        return uh

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
        sign = (-1) ** jnp.arange(n)
        uh = jax.scipy.fft.dct(u, n=n)
        uh = uh * jnp.pi * sign / n / 2 / self.domain_factor
        if len(u) > self.N:
            uh = uh[: self.N]
        return uh

    @jax.jit(static_argnums=(0, 2))
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
        x0: Array = jnp.array(0.0, dtype=float)
        if N == 0:
            return jnp.array([x0])
        x1: Array = c[-1] * N * 2
        if N == 1:
            return jnp.array([x1, x0])

        def inner_loop(
            carry: tuple[Array, Array], n: int
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = 2 * (n + 1) * c[n + 1] + x0
            return (x1, x2), x2

        xs = jax.lax.scan(inner_loop, (x0, x1), jnp.arange(N - 2, -1, -1))[1]
        return jnp.concatenate(
            (jnp.array([xs[-1] / 2]), xs[-2::-1], jnp.array([x1, x0]))
        )

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

        U = Ultraspherical(self.N, domain=self.domain, system=self.system, lambda_=k)

        return (
            U._a(i, j)
            * self.psi(j + k, k)
            / self.psi(i + k, k)
            * self.gn(j + k)
            / self.gn(i + k)
        )

    def matrices(
        self, i: int, trial: tuple[OrthogonalSpace, int], q: int = 0
    ) -> Matrix | DiaMatrix | None:
        """Return (possibly sparse) operator matrices between test/trial Chebyshev modes

        Constructs square sparse differentiation / mass-like
        matrices for combinations of test index i and trial index j flags:

            (i, j):
              (0,0): Diagonal mass-matrix.
              (0,1): First derivative.
              (1,0): Transpose of (0,1).
              (0,2): Second derivative.
              (2,0): Transpose of (0,2).

        Args:
            i: Derivative order for test function. Should be 0.
            trial: Tuple (u, j) with trial space u and derivative order j.
            q: polynomial degree of coefficient.

        Returns:
            Matrix | DiaMatrix | None if combination unsupported.
        """
        u, j = trial
        assert isinstance(u, Chebyshev), (
            "Trial space must be Chebyshev for Chebyshev matrices"
        )
        if (i, j) not in ((0, 0), (0, 1), (1, 0), (0, 2), (2, 0)):
            return None

        A = None
        if q != 0:
            A = self.A().power(q)

        if i == 0 and j == 0:
            M = diags([self.norm_squared()], offsets=(0,), shape=(self.N, u.N))
            return M if A is None else A.T @ M

        if i in (1, 2) and j == 0:
            m = u.matrices(j, (self, i), q=q)
            if m is not None:
                m = m.T
            return m

        offsets = jnp.arange(j, u.N, 2)
        if len(offsets) == 0:
            return None
        k = jnp.arange(max(self.N, u.N))

        def _getkey1(offset):
            Q = min(self.N, u.N - offset)
            return jnp.pi * k[offset : (Q + offset)]

        def _getkey2(offset):
            Q = min(self.N, u.N - offset)
            return (
                k[offset : (Q + offset)]
                * (k[offset : (Q + offset)] ** 2 - k[:Q] ** 2)
                * jnp.pi
                / 2
            )

        _getkey = _getkey1 if j == 1 else _getkey2

        M = diags(
            [_getkey(m) for m in offsets], tuple(offsets.tolist()), (self.N, u.N)
        ).to_matrix()  # Matrix is upper triangular, better and faster to use dense.
        return M if A is None else A.T @ M


##### Predefined Composite spaces #####


class Phi_1(PGComposite):
    r"""
    Composite space for Mortensen's Petrov-Galerkin method order 1.

    The test functions are defined by

    .. math::

        \phi_k = \frac{2(1-x^2)}{\pi k(k+1)} T'_{k+1}, \, k=0, 1, \ldots, N-3

    where :math:`T'_{k+1}` is the derivative of Chebyshev polynomial k+1 of the
    first kind.

    When used as a test function space, the resulting inner product is a skewed
    diagonal matrix with entries

    .. math::
        \langle \phi_k, T'_j \rangle = \delta_{k,k+1}

    The advantage of using this space is that the resulting operator matrices are
    sparse, whereas a regular Chebyshev test space (Galerkin method) would yield
    dense operator matrices for the same problem.

    Examples:
        >>> from jaxfun.galerkin.Chebyshev import Phi_1
        >>> from jaxfun.galerkin import inner, Chebyshev, TrialFunction, TestFunction
        >>> from jaxfun.galerkin import FunctionSpace
        >>> N = 6
        >>> P = Phi_1(N)
        >>> V = Chebyshev.Chebyshev(N)
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
        >>> D = FunctionSpace(N, Chebyshev.Chebyshev, {"left": {"D": 0}})
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
            Chebyshev,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=domain,
            stencil={0: 1 / sp.pi / (n + 1), 2: -1 / sp.pi / (n + 1)},
            name=name,
            fun_str=fun_str,
            order=1,
        )


class Phi_2(PGComposite):
    r"""Composite space for Mortensen's Petrov-Galerkin method order 2.

    The test functions are defined by

    .. math::

        \phi_k = \frac{2(1-x^2)^2 T''_{k+2}}{\pi (k+1)(k+2)^2(k+3)}

    where :math:`T''_{k+2}` is the second derivative of Chebyshev polynomial
    k+2 of the first kind.

    When used as a test function space, the resulting inner product is a skewed
    diagonal matrix with entries

    .. math::
        \langle \phi_k, T''_j \rangle = \delta_{k,k+2}

    The advantage of using this space is that the resulting operator matrices are
    sparse, whereas a regular Chebyshev test space (Galerkin method) would yield
    dense operator matrices for the same problem.

    Examples:
        >>> from jaxfun.galerkin.Chebyshev import Phi_2
        >>> from jaxfun.galerkin import inner, Chebyshev, TrialFunction, TestFunction
        >>> from jaxfun.galerkin import FunctionSpace
        >>> N = 7
        >>> P = Phi_2(N)
        >>> V = Chebyshev.Chebyshev(N)
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
        ...     N, Chebyshev.Chebyshev, {"left": {"D": 0}, "right": {"D": 0}}
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
            Chebyshev,
            bcs={"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}},
            domain=domain,
            stencil={
                0: 1 / (2 * sp.pi * (n + 1) * (n + 2)),
                2: -1 / (sp.pi * (n**2 + 4 * n + 3)),
                4: 1 / (2 * sp.pi * (n + 2) * (n + 3)),
            },
            name=name,
            fun_str=fun_str,
            order=2,
        )


class Phi_4(PGComposite):
    r"""Composite space for Mortensen's Petrov-Galerkin method order 4.

    The test functions are defined by

    .. math::

        \phi_k = \frac{(1-x^2)^4}{h^{(4)}_{k+4}} T^{(4)}_{k+4}

    where :math:`T^{(4)}_{k+4}` is the fourth derivative of Chebyshev polynomial
    k+4 of the first kind.

    When used as a test function space, the resulting inner product is a skewed
    diagonal matrix with entries

    .. math::
        \langle \phi_k, T^{(4)}_j \rangle = \delta_{k,k+4}

    The advantage of using this space is that the resulting operator matrices are
    sparse, whereas a regular Chebyshev test space (Galerkin method) would yield
    dense operator matrices for the same problem.

    Examples:
        >>> from jaxfun.galerkin.Chebyshev import Phi_4
        >>> from jaxfun.galerkin import inner, Chebyshev, TrialFunction, TestFunction
        >>> from jaxfun.galerkin import FunctionSpace
        >>> N = 9
        >>> P = Phi_4(N)
        >>> V = Chebyshev.Chebyshev(N)
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
        ...     Chebyshev.Chebyshev,
        ...     {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}},
        ... )
        >>> x = D.system.x
        >>> u = TrialFunction(D)
        >>> A = inner(u.diff(x, 4) * v, sparse=True)
        >>> A.todense()
        Array([[ 0.33333334,  0.        , -1.6       ,  0.        ,  1.        ],
               [ 0.        ,  0.5       ,  0.        , -1.6666666 ,  0.        ],
               [ 0.        ,  0.        ,  0.6       ,  0.        , -1.7142857 ],
               [ 0.        ,  0.        ,  0.        ,  0.6666667 ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.71428573]],      dtype=float32)
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
            Chebyshev,
            bcs={
                "left": {"D": 0, "N": 0, "N2": 0, "N3": 0},
                "right": {"D": 0, "N": 0, "N2": 0, "N3": 0},
            },
            domain=domain,
            stencil={
                0: 1 / (8 * sp.pi * (n + 1) * (n + 2) * (n + 3) * (n + 4)),
                2: -1 / (2 * sp.pi * (n + 1) * (n + 3) * (n + 4) * (n + 5)),
                4: 3 / (4 * sp.pi * (n + 2) * (n + 3) * (n + 5) * (n + 6)),
                6: -1 / (2 * sp.pi * (n + 3) * (n + 4) * (n + 5) * (n + 7)),
                8: 1 / (8 * sp.pi * (n + 4) * (n + 5) * (n + 6) * (n + 7)),
            },
            name=name,
            fun_str=fun_str,
            order=4,
        )
