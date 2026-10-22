from __future__ import annotations

from functools import cached_property

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.composite import BCGeneric, Composite, Stencil
from jaxfun.la import DiaMatrix, Matrix, diags
from jaxfun.typing import TestSpaceKind
from jaxfun.utils.common import Domain, cache_static, jit_method, jit_vmap, n
from jaxfun.utils.fastgl import leggauss

from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace


@jax.jit(static_argnums=(0, 1))
def _dense_derivative_matrix_data(
    test_modes: int, trial_modes: int, derivative: int
) -> Array:
    """Return dense Legendre first/second derivative coupling matrices."""
    rows = jnp.arange(test_modes)[:, None]
    cols = jnp.arange(trial_modes)[None, :]
    offsets = cols - rows
    mask = (offsets >= derivative) & ((offsets - derivative) % 2 == 0)
    values = jax.lax.select(
        derivative == 1,
        jnp.full((test_modes, trial_modes), 2),
        cols * (cols + 1) - rows * (rows + 1),
    )
    return jnp.where(mask, values, 0.0)


def _dense_derivative_matrix(
    test_modes: int, trial_modes: int, derivative: int
) -> Matrix:
    return Matrix(_dense_derivative_matrix_data(test_modes, trial_modes, derivative))


def _differentiate_stencil(stencil: dict[int, sp.Expr]) -> dict[int, sp.Expr] | None:
    r"""Return the stencil of the derivative of a composite Legendre basis.

    A composite basis function :math:`\psi_k = \sum_r c_r L_{k+r}` has, by the
    Legendre identity :math:`L'_m = \sum_{p=m-1,m-3,\ldots} (2p+1) L_p`,

    .. math::
        \psi'_k = \sum_{s \ge 0} (2(k+s)+1)\, T_s\, L_{k+s}, \qquad
        T_s = \sum_{r > s,\; r - s \text{ odd}} c_r .

    That inner sum runs all the way down to :math:`L_0`, so the derivative is
    only sparse if the tails below the stencil window vanish, i.e.
    :math:`T_{-1} = T_{-2} = 0`. Homogeneous boundary conditions are exactly
    what makes them vanish, and every derivative then narrows the stencil by
    one diagonal.

    Args:
        stencil: Ordered dict of diagonal shift -> expression in n.

    Returns:
        Stencil dict of the derivative, or None if it is not sparse.
    """
    offsets = sorted(stencil)

    def tail(s: int) -> sp.Expr:
        terms = [sp.sympify(stencil[r]) for r in offsets if r > s and (r - s) % 2]
        return sp.simplify(sp.Add(*terms))

    if tail(-1) != 0 or tail(-2) != 0:
        return None
    derivative = {
        s: sp.simplify((2 * (n + s) + 1) * tail(s)) for s in range(offsets[-1])
    }
    return {s: c for s, c in derivative.items() if c != 0} or None


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
        X = jnp.asarray(X)
        x0 = jnp.ones_like(X)

        def inner_loop(
            carry: tuple[Array, Array], i: int | Array
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return (x1, x2), x1 * c[i - 1]

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))
        return jnp.sum(xs, axis=0) + c[0]

    @cache_static
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

    def integration_points_and_weights(
        self, N: int | None = None
    ) -> tuple[Array, Array]:
        """Return the Gauss-Legendre nodes and weights; the weight function is 1."""
        return self.quad_points_and_weights(N)

    @jit_vmap(in_axes=(0, None), static_argnums=2)
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        """Evaluate single Legendre polynomial P_i at X.

        Args:
            X: Points in [-1, 1].
            i: Polynomial index (0 <= i < N).

        Returns:
            P_i(X) values.
        """
        X = jnp.asarray(X)
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
        X = jnp.asarray(X)
        x0 = X * 0 + 1

        def inner_loop(
            carry: tuple[Array, Array], i: int | Array
        ) -> tuple[tuple[Array, Array], Array]:
            x0, x1 = carry
            x2 = (x1 * X * (2 * i - 1) - x0 * (i - 1)) / i
            return (x1, x2), x1

        _, xs = jax.lax.scan(inner_loop, (x0, X), jnp.arange(2, self.N + 1))
        return jnp.concatenate((jnp.expand_dims(x0, axis=0), xs))

    @jit_method(static_argnums=2)
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
        x0: Array = jnp.zeros((), dtype=c.dtype)
        if N == 0:
            return jnp.array([x0])
        if N == 1:
            return jnp.array([c[-1] * (2 * N - 1), x0])

        # Old version using scan, which is slow on gpus because it launches a kernel
        # for each iteration.
        # def inner_loop(
        #     carry: tuple[Array, Array], n: int | Array
        # ) -> tuple[tuple[Array, Array], Array]:
        #     x0, x1 = carry
        #     x2 = (2 * n + 1) * c[n + 1] + (2 * n + 1) / (2 * n + 5) * x0
        #     return (x1, x2), x2
        # _, xs = jax.lax.scan(inner_loop, (x0, x1), jnp.arange(N - 2, -1, -1))
        # return jnp.concatenate((xs[::-1], jnp.array([x1, x0])))

        # The recurrence is `x[n] = (2n+1) c[n+1] + (2n+1)/(2n+5) x[n+2]`, and
        # since 2n+5 = 2(n+2)+1 the substitution `w[n] = x[n] / (2n+1)`
        # telescopes it into `w[n] = c[n+1] + w[n+2]` -- lag two, and a
        # coefficient of exactly one -- so each parity class of `n` is a running
        # total and the whole thing is a reversed cumulative sum. Written as a
        # scan it is `N` sequential steps, which on an accelerator is `N` kernel
        # launches for a handful of arithmetic each; as a cumsum it is
        # logarithmic depth, and losing the divisions costs no accuracy.

        m = (N + 2) // 2
        a = jnp.pad(c, (0, 2 * m - (N + 1))).reshape(m, 2)
        w = jnp.flip(jnp.cumsum(jnp.flip(a, axis=0), axis=0), axis=0)
        return jnp.concatenate(
            ((2 * jnp.arange(N) + 1) * w.reshape(2 * m)[1 : N + 1], jnp.stack([x0]))
        )

    legder = derivative_coeffs

    def _matrices(
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

        if i == 0 and j in (1, 2):
            if j >= u.N:
                return None
            M = _dense_derivative_matrix(self.N, u.N, j)
            return M if A is None else A.T @ M

        if j == 0 and i in (1, 2):
            m = u._matrices(j, (self, i), q=q)
            if m is not None:
                return m.T
            return None

        return None


### Predefined Composite spaces #####


class LGComposite(Composite):
    """Legendre Composite basis with Galerkin test functions.

    The 'tau' test functions are the Galerkin ones recombined, since those keep
    the operator matrices sparser than the Petrov-Galerkin ones do for
    Legendre; see `TauRecombined`.

    Args:
        N: Target (unconstrained) number of modes of underlying orthogonal.
        orthogonal: Underlying orthogonal basis class.
        bcs: BoundaryConditions specification.
        domain: Physical domain (defaults to [-1, 1]).
        name: Space name.
        fun_str: Symbol stem for basis functions.
        system: Optional coordinate system.
        stencil: Optional custom stencil dict {shift: sympy_expr}.
        scaling: SymPy expression scaling the stencil diagonals.
        order: Highest derivative order for trial function.

    Attributes:
        orthogonal: Instance of underlying orthogonal basis.
        stencil: Ordered dict of diagonal shift -> expression / scaling.
        S: Sparse (DiaMatrix) stencil matrix for test functions.
        ST: Pre-computed transpose of S for efficiency.
        scaling: Scaling expression applied to user stencil.
        derivative_stencils: Stencils of the derivatives of the basis.
    """

    _tau_base = TestSpaceKind.GALERKIN

    @cached_property
    def derivative_stencils(self) -> tuple[DiaMatrix, ...]:
        r"""Return stencils of :math:`\psi^{(p)}` for p = 0, 1, ...

        Entry p maps composite coefficients to coefficients of the p'th
        derivative in the orthogonal basis, :math:`\psi_k^{(p)} = \sum_l
        G^{(p)}_{kl} L_l`, with entry 0 being the basis stencil S itself.

        Derivatives of the orthogonal basis are dense, but for a basis with
        homogeneous boundary conditions the composite ones telescope down to a
        few diagonals, one fewer for every derivative (see
        :func:`_differentiate_stencil`). The tuple ends where they stop being
        sparse, which is after as many derivatives as there are boundary
        conditions at each end.
        """
        stencils = [self.S]
        if not self.stencil.is_pure:
            return tuple(stencils)  # the rule's derivative misses overridden rows
        stencil = _differentiate_stencil(self.stencil.rule)
        while stencil is not None:
            # Scaling basis function k scales its derivatives alike.
            G = Stencil(stencil).scaled(self.scaling).matrix(self.S.shape)
            stencils.append(G)
            stencil = _differentiate_stencil(stencil)
        return tuple(stencils)

    def _matrices(
        self, i: int, trial: tuple[OrthogonalSpace, int], q: int = 0
    ) -> DiaMatrix | Matrix | None:
        r"""Return (sparse) operator matrices for Galerkin method.

        .. math::
            \langle \psi_m^{(i)}, x^q \phi_n^{(trial[1])} \rangle

        where \psi_m^{(i)} are i'th derivative of test functions and
        \phi_n^{(trial[1])} are trial functions with derivative order
        trial[1].

        Args:
            i: Derivative order for test function.
            trial: Tuple (u, j) with trial space u and derivative order j.
            q: polynomial order for scaling.

        """

        u, j = trial

        assert isinstance(u, Legendre | Composite), (
            "Trial space must be Legendre or Composite."
        )

        if isinstance(u, BCGeneric):
            return None

        if u.bcs != self.bcs:  # only fast paths for Galerkin.
            return None

        assert isinstance(u, Composite)

        if self.num_dofs != u.num_dofs:  # only fast paths for square matrices.
            return None

        if not isinstance(u, LGComposite):
            return None

        # Both bases are sums over the orthogonal basis, psi_m^(i) = sum_k
        # G^(i)_{mk} L_k, so the matrix is G^(i) @ D @ G^(j).T with D the
        # orthogonal <x**q L_m, L_n>. All factors are sparse, so the products
        # never touch a dense intermediate. Note the use of the private
        # _matrices, since the public matrices() applies the domain factor and
        # maps x**q to the reference domain, both of which are already taken
        # care of by the matrices() wrapper calling this method.
        Gt, Gu = self.derivative_stencils, u.derivative_stencils

        if i < len(Gt) and j < len(Gu):
            D = self.orthogonal._matrices(0, (u.orthogonal, 0), q=q)
            if D is None:
                return None
            return Gt[i] @ D @ Gu[j].T

        # Derivatives beyond the sparse stencils are integrated by parts onto
        # the side that still has one. psi^(p) vanishes at both boundaries for
        # p < len(G) - 1, which is what kills the boundary term:
        #   <x**q psi^(i), phi^(j)> = -q <x**(q-1) psi^(i), phi^(j-1)>
        #                              - <x**q psi^(i+1), phi^(j-1)>
        #   <x**q psi^(i), phi^(j)> = -q <x**(q-1) psi^(i-1), phi^(j)>
        #                              - <x**q psi^(i-1), phi^(j+1)>
        if j >= len(Gu) and i + 1 < len(Gt):
            A = self._matrices(i + 1, (u, j - 1), q=q)
            B = None if q == 0 else self._matrices(i, (u, j - 1), q=q - 1)
        elif i >= len(Gt) and j + 1 < len(Gu):
            A = self._matrices(i - 1, (u, j + 1), q=q)
            B = None if q == 0 else self._matrices(i - 1, (u, j), q=q - 1)
        else:
            return None  # boundary terms do not vanish: use quadrature

        if A is None:
            return None
        if q == 0:
            return -A
        if B is None:
            return None
        return -A - q * B

        return None
