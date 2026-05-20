from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array
from scipy import sparse as scipy_sparse

from jaxfun.la.diamatrix import DiaMatrix, diakron
from jaxfun.la.matrix import LUFactors, Matrix
from jaxfun.la.matrixprotocol import (
    DiaMatrixSolveMethod,
    MatrixProtocol,
    SolverNotApplicable,
    _CacheBox,
)

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction


class TPMatrix(nnx.Pytree):  # noqa: B903
    """Rank-d separable tensor product operator A = kron(A0, A1, ...).

    Provides efficient matvec via successive multiplications instead of
    forming the full Kronecker product explicitly.

    Attributes:
        mats: List of per-axis sparse/dense matrices.
        scale: Scalar scaling (multiplicative).
        global_indices: Tuple of global index into vectorized expansions.
    """

    def __init__(
        self,
        mats: Sequence[MatrixProtocol],
        scale: complex,
        global_indices: tuple[int, int] = (0, 0),
    ) -> None:
        self.mats = nnx.List(mats)
        self.scale = scale
        self.global_indices = global_indices

    @property
    def dims(self) -> int:
        return len(self.mats)

    def __len__(self) -> int:
        return len(self.mats)

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        sparse_box: _CacheBox[DiaMatrix] | None = getattr(self, "_sparse_cache", None)
        if sparse_box is not None:
            return sparse_box.value
        kron = tpmats_to_kron(self)
        if not isinstance(kron, DiaMatrix):
            kron = DiaMatrix.from_dense(kron.todense(), tol=tol)
        object.__setattr__(self, "_sparse_cache", _CacheBox(kron))
        return kron

    def todense(self) -> Array:
        """Return the dense Kronecker product as a raw array.

        The underlying :class:`~jaxfun.la.Matrix` or
        :class:`~jaxfun.la.DiaMatrix` is cached for repeated calls.

        Returns:
            2-D :class:`~jaxfun.Array` of shape ``(N, N)`` where ``N`` is the
            total number of degrees of freedom.
        """
        dense_box: _CacheBox[Matrix] | None = getattr(self, "_dense_cache", None)
        if dense_box is not None:
            return dense_box.value.todense()
        sparse_box: _CacheBox[DiaMatrix] | None = getattr(self, "_sparse_cache", None)
        if sparse_box is not None:
            return sparse_box.value.todense()
        kron = tpmats_to_kron(self)
        if isinstance(kron, Matrix):
            object.__setattr__(self, "_dense_cache", _CacheBox(kron))
            return kron.todense()
        object.__setattr__(self, "_sparse_cache", _CacheBox(kron))
        return kron.todense()

    def to_matrix(self) -> Matrix:
        return Matrix(self.todense())

    def _matmul_array(self, w: Array) -> Array:
        result = w
        for i, mat in enumerate(self.mats):
            result = mat.matvec(result, axis=i)
        return result * jnp.asarray(self.scale)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply matrix to rank-2 coefficient array u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._matmul_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def _rmatmul_array(self, w: Array) -> Array:
        result = w
        for i, mat in enumerate(self.mats):
            result = mat.T.matvec(result, axis=i)
        return result * jnp.asarray(self.scale)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._rmatmul_array(w)

    def solve(self, rhs: Array) -> Array:
        """Solve ``(scale * A0 ⊗ A1 ⊗ …) x = rhs`` using Kronecker-factored LU.

        Exploits the mixed-product property

        .. math::

            (A_0 \\otimes A_1 \\otimes \\cdots)^{-1}
            = A_0^{-1} \\otimes A_1^{-1} \\otimes \\cdots

        to avoid forming the full Kronecker product.  Each factor's LU is
        computed once and cached on the factor matrix itself, so repeated
        ``solve`` calls pay only the substitution cost.

        Args:
            rhs: Right-hand side array.  May be flat ``(n,)`` or have the
                multidimensional shape ``(n0, n1, …)``.

        Returns:
            Solution array with the same shape as ``rhs``.

        """
        return self.lu_factor().solve(rhs)

    def lu_factor(self) -> TPLUFactors:
        """Pre-compute LU factors for every Kronecker factor.

        Returns a :class:`TPLUFactors` whose :meth:`~TPLUFactors.solve` method
        solves the Kronecker system without rebuilding the factorisation.
        """
        lu_factors = [mat.lu_factor() for mat in self.mats]
        shape = tuple(int(mat.shape[0]) for mat in self.mats)
        return TPLUFactors(lu_factors=lu_factors, scale=self.scale, shape=shape)


class TPLUFactors:
    """LU factorisation of a :class:`TPMatrix` (Kronecker product).

    Holds the per-factor LU objects and applies them sequentially on their
    respective axes to solve the full tensor-product system.

    Attributes:
        lu_factors: Per-axis LU factorisation objects (DiaMatrix or Matrix).
        scale: Scalar from the parent :class:`TPMatrix`.
        shape: Tuple of per-factor sizes ``(n0, n1, …)``.
    """

    def __init__(
        self, lu_factors: list, scale: complex, shape: tuple[int, ...]
    ) -> None:
        self.lu_factors = lu_factors
        self.scale = scale
        self.shape = shape

    @jax.jit(static_argnums=(0,))
    def solve(self, rhs: Array) -> Array:
        """Solve ``(scale * A0 ⊗ A1 ⊗ …) x = rhs``.

        Args:
            rhs: Right-hand side.  Flat ``(n,)`` or shaped ``(n0, n1, …)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        y = rhs.reshape(self.shape)
        for i, lu in enumerate(self.lu_factors):
            y = lu.solve(y, axis=i)
        return (y / jnp.asarray(self.scale)).reshape(rhs.shape)


class TPSolveMethod(StrEnum):
    """High-level solver selection for :meth:`TPMatrices.solve`.

    Attributes:
        AUTO: Try the factored path (:meth:`TPMatrices.lu_factor`) first;
            fall back to explicit Kronecker assembly if it raises
            :exc:`ValueError`.
        LU: Force the factored path (diagonalization or wavenumber solver).
            Propagates :exc:`ValueError` if the factor-matrix structure is
            not suitable.
        KRON: Force explicit Kronecker product assembly.  The assembled
            :class:`~jaxfun.la.DiaMatrix` or :class:`~jaxfun.la.Matrix` is
            cached; the DIA-matrix solver is selected via *kron_method* in
            :meth:`TPMatrices.solve`.
    """

    AUTO = "auto"
    LU = "lu"
    KRON = "kron"


class TPMatrices(nnx.Pytree):
    """Container for list of TPMatrix bilinear operator tensors."""

    def __init__(self, tpmats: list[TPMatrix]) -> None:
        self.tpmats = nnx.List(tpmats)

    @jax.jit
    def _apply_array(self, u: Array) -> Array:
        return jnp.sum(jnp.array([mat._matmul_array(u) for mat in self.tpmats]), axis=0)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply summed tensor product operator to u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._apply_array(w)

    def __len__(self) -> int:
        """Return number of TPMatrix terms."""
        return len(self.tpmats)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return jnp.sum(
            jnp.array([mat._rmatmul_array(w) for mat in self.tpmats]), axis=0
        )

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        sparse_box: _CacheBox[DiaMatrix] | None = getattr(self, "_sparse_cache", None)
        if sparse_box is not None:
            return sparse_box.value
        kron = tpmats_to_kron(list(self.tpmats))
        if not isinstance(kron, DiaMatrix):
            kron = DiaMatrix.from_dense(kron.todense(), tol=tol)
        object.__setattr__(self, "_sparse_cache", _CacheBox(kron))
        return kron

    def todense(self) -> Array:
        """Return the dense Kronecker product as a raw array.

        The underlying :class:`~jaxfun.la.Matrix` or
        :class:`~jaxfun.la.DiaMatrix` is cached for repeated calls.

        Returns:
            2-D :class:`~jaxfun.Array` of shape ``(N, N)`` where ``N`` is the
            total number of degrees of freedom.
        """
        dense_box: _CacheBox[Matrix] | None = getattr(self, "_dense_cache", None)
        if dense_box is not None:
            return dense_box.value.todense()
        sparse_box: _CacheBox[DiaMatrix] | None = getattr(self, "_sparse_cache", None)
        if sparse_box is not None:
            return sparse_box.value.todense()
        kron = tpmats_to_kron(list(self.tpmats))
        if isinstance(kron, Matrix):
            object.__setattr__(self, "_dense_cache", _CacheBox(kron))
            return kron.todense()
        object.__setattr__(self, "_sparse_cache", _CacheBox(kron))
        return kron.todense()

    def to_matrix(self) -> Matrix:
        return Matrix(self.todense())

    def lu_factor(
        self,
    ) -> TPMatricesDenseLUFactors | TPMatricesLUFactors | TPMatricesWavenumberSolver:
        """Pre-compute factors for repeated fast solves.

        Dispatch order:

        1. If all per-axis factor matrices are dense :class:`~jaxfun.la.Matrix`
           instances, use :func:`tpmats_dense_lu_factor` (simple full Kronecker
           LU — works for any linear system).
        2. Otherwise try :func:`tpmats_wavenumber_factor` (efficient for
           Fourier x polynomial problems).
        3. Fall back to :func:`tpmats_lu_factor` (diagonalization — requires
           simultaneously diagonalizable factor matrices per axis).

        Returns:
            :class:`TPMatricesDenseLUFactors`, :class:`TPMatricesWavenumberSolver`,
            or :class:`TPMatricesLUFactors` for repeated fast solves.
        """
        cached: (
            TPMatricesDenseLUFactors
            | TPMatricesLUFactors
            | TPMatricesWavenumberSolver
            | None
        ) = getattr(self, "_lu_cache", None)
        if cached is not None:
            return cached.value
        result: (
            TPMatricesDenseLUFactors | TPMatricesLUFactors | TPMatricesWavenumberSolver
        )
        tpmats_list = list(self.tpmats)
        if all(isinstance(mat, Matrix) for tp in tpmats_list for mat in tp.mats):
            result = tpmats_dense_lu_factor(tpmats_list)
        else:
            try:
                result = tpmats_wavenumber_factor(tpmats_list)
            except SolverNotApplicable:
                result = tpmats_lu_factor(tpmats_list)
        object.__setattr__(self, "_lu_cache", _CacheBox(result))
        return result

    def solve(
        self,
        rhs: Array,
        *,
        method: TPSolveMethod | str = TPSolveMethod.AUTO,
        kron_method: DiaMatrixSolveMethod | str = DiaMatrixSolveMethod.AUTO,
    ) -> Array:
        """Solve the summed tensor-product system.

        Args:
            rhs: Right-hand side array.
            method: High-level solver selection. One of:

                * ``"auto"`` (default) — tries the factored path
                  (:meth:`lu_factor`) first; falls back to explicit Kronecker
                  product assembly if the factor-matrix structure is not
                  suitable (e.g. not simultaneously diagonalizable).
                * ``"lu"`` — force the factored path (diagonalization or
                  wavenumber solver). Raises :exc:`ValueError` if the structure
                  is not suitable.
                * ``"kron"`` — force explicit Kronecker product assembly.
                  The assembled matrix is cached for repeated solves.

            kron_method: Solver method forwarded to
                :meth:`~jaxfun.la.DiaMatrix.lu_solve` when the Kronecker-product
                path is used and the assembled matrix is a
                :class:`~jaxfun.la.DiaMatrix` (i.e. all factor matrices are
                sparse).  One of ``"auto"``, ``"banded"``, ``"rcm"``,
                ``"dense"``.  Ignored when the assembled matrix is a dense
                :class:`~jaxfun.la.Matrix`.

        Returns:
            Solution array with the same shape as *rhs*.

        Raises:
            ValueError: If ``method="lu"`` but the factor-matrix structure is
                not suitable for the factored solver.
        """
        method = TPSolveMethod(method)

        def _kron_solve(r: Array) -> Array:
            flat = r.flatten()
            # DiaMatrix path: shared cache with tosparse()
            sparse_box: _CacheBox[DiaMatrix] | None = getattr(
                self, "_sparse_cache", None
            )
            if sparse_box is not None:
                return sparse_box.value.lu_solve(flat, method=kron_method).reshape(
                    r.shape
                )
            # Dense Matrix path
            dense_box: _CacheBox[Matrix] | None = getattr(self, "_dense_cache", None)
            if dense_box is not None:
                return dense_box.value.solve(flat).reshape(r.shape)
            # No cache yet: compute and store
            kron = tpmats_to_kron(list(self.tpmats))
            if isinstance(kron, DiaMatrix):
                object.__setattr__(self, "_sparse_cache", _CacheBox(kron))
                return kron.lu_solve(flat, method=kron_method).reshape(r.shape)
            object.__setattr__(self, "_dense_cache", _CacheBox(kron))
            return kron.solve(flat).reshape(r.shape)

        if method == TPSolveMethod.LU:
            return self.lu_factor().solve(rhs)
        if method == TPSolveMethod.KRON:
            return _kron_solve(rhs)
        # AUTO: try factored path, fall back to kron
        try:
            return self.lu_factor().solve(rhs)
        except SolverNotApplicable:
            return _kron_solve(rhs)


class TPMatricesLUFactors:
    """Diagonalization-based solver for a sum of tensor-product operators.

    Solves

    .. math::

        \\sum_k s_k \\, (A_k^{(0)} \\otimes A_k^{(1)} \\otimes \\cdots)\\, x = f

    by simultaneously diagonalizing the factor matrices on each axis.

    Given a shared eigenbasis :math:`V` satisfying
    :math:`V^T A V = \\Lambda` (diagonal) and :math:`V^T B V = I`, the system
    reduces to element-wise division in the transformed space — :math:`O(n^d)`
    work after the :math:`O(n^3)` per-axis factorisation.

    For 2D Poisson (``K⊗M + M⊗K``): the denominator is
    :math:`D_{ij} = \\lambda_i + \\lambda_j` and the back-transform is
    :math:`U = V \\tilde{U} V^T`.
    """

    def __init__(
        self,
        eigvecs: list[Array],
        per_term_eigenvalues: list[list[Array]],
        scales: list[complex],
        shape: tuple[int, ...],
    ) -> None:
        self.eigvecs = eigvecs  # list of (n_i, n_i) eigenvector matrices
        self.per_term_eigenvalues = per_term_eigenvalues  # [term][axis] -> (n_axis,)
        self.scales = scales
        self.shape = shape

    @jax.jit(static_argnums=(0,))
    def solve(self, rhs: Array) -> Array:
        """Solve the summed tensor-product system for RHS ``rhs``.

        Args:
            rhs: Right-hand side, flat ``(n,)`` or shaped ``(n0, n1, ...)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        shape = self.shape
        ndim = len(shape)
        F = rhs.reshape(shape)

        # Forward transform: apply V_i^T along each axis i.
        # jnp.tensordot(V.T, X, axes=[[1],[i]]) contracts V^T with axis i of X,
        # placing the result at position 0; moveaxis restores it to position i.
        Ftilde = F
        for i, V in enumerate(self.eigvecs):
            Ftilde = jnp.tensordot(V.T, Ftilde, axes=[[1], [i]])
            Ftilde = jnp.moveaxis(Ftilde, 0, i)

        # Denominator: D[i0,i1,...] = sum_k s_k * Λ_k[0][i0] * Λ_k[1][i1] * ...
        dtype = jnp.result_type(rhs.dtype, jnp.float32)
        D = jnp.zeros(shape, dtype=dtype)
        for evals_k, s_k in zip(self.per_term_eigenvalues, self.scales):
            term = jnp.ones(shape, dtype=dtype)
            for i, ev in enumerate(evals_k):
                idx: list = [None] * ndim
                idx[i] = slice(None)
                term = term * ev[tuple(idx)]
            D = D + jnp.asarray(s_k, dtype=dtype) * term

        # Solve in the transformed space (element-wise division).
        Utilde = Ftilde / D

        # Back-transform: apply V_i along each axis i.
        U = Utilde
        for i, V in enumerate(self.eigvecs):
            U = jnp.tensordot(V, U, axes=[[1], [i]])
            U = jnp.moveaxis(U, 0, i)

        return U.reshape(rhs.shape)


def _make_wavenumber_vmap_solve(
    L_offsets: tuple[int, ...],
    U_offsets: tuple[int, ...],
    n_P: int,
    dtype: Any,
) -> Callable[..., Array]:
    """Build a ``jax.vmap``-compiled batch solver for the wavenumber loop.

    Returns a function ``f(L_data_batch, U_data_batch, rhs_2d) -> sol_2d``
    that solves each 1-D banded system ``B_k x = b_k`` using forward and
    backward substitution compiled via :func:`jax.lax.scan`.  DIA offsets are
    captured as static Python values in the closure so :func:`jax.vmap` only
    traces over the array data — avoiding any pytree-metadata issues that
    would arise from constructing :class:`~jaxfun.la.DiaMatrix` instances
    with traced arrays.

    Args:
        L_offsets: Sub-diagonal offsets of the L factor (shared across all k).
        U_offsets: Super-diagonal offsets of the U factor (shared across all k).
        n_P: Length of each 1-D polynomial system.
        dtype: JAX dtype used for zero-padding of missing diagonals.

    Returns:
        A vmapped callable ``(L_data_batch, U_data_batch, rhs_2d) -> sol_2d``
        where each batch dimension corresponds to one Fourier wavenumber.
    """
    p = max((-o for o in L_offsets if o < 0), default=0)
    q = max((o for o in U_offsets if o > 0), default=0)

    # Index of each sub/super-diagonal in the data array, or None if absent.
    l_indices: list[int | None] = [
        L_offsets.index(-s) if -s in L_offsets else None for s in range(1, p + 1)
    ]
    U_main_idx: int = U_offsets.index(0)
    u_indices: list[int | None] = [
        U_offsets.index(s) if s in U_offsets else None for s in range(1, q + 1)
    ]

    # Reversal index for backward substitution — static since n_P is fixed.
    rev = jnp.arange(n_P - 1, -1, -1)

    def _fwd_elim(L_data: Array, b: Array) -> Array:
        """Solve L y = b (unit lower-triangular) via forward scan."""
        if p == 0:
            return b
        l_rows: list[Array] = []
        for s, idx in enumerate(l_indices, start=1):
            if idx is not None:
                d = L_data[idx]
                l_rows.append(
                    jnp.concatenate([jnp.zeros(s, dtype=d.dtype), d[: n_P - s]])
                )
            else:
                l_rows.append(jnp.zeros(n_P, dtype=dtype))
        l_mat = jnp.stack(l_rows)  # (p, n_P); l_mat[j, i] = L[i, i-(j+1)]

        def step(window: Array, xs: tuple) -> tuple[Array, Array]:
            bi, l_i = xs  # scalar, (p,)
            yi = bi - jnp.dot(l_i, window)
            return jnp.concatenate([yi[None], window[:-1]]), yi

        carry0 = jnp.zeros(p, dtype=b.dtype)
        _, ys = jax.lax.scan(step, carry0, (b, l_mat.T))
        return ys

    def _bwd_sub(U_data: Array, y: Array) -> Array:
        """Solve U x = y (upper-triangular) via backward scan."""
        diag_d = U_data[U_main_idx]
        if q == 0:
            return y / diag_d
        u_rows: list[Array] = []
        for s, idx in enumerate(u_indices, start=1):
            if idx is not None:
                d = U_data[idx]
                u_rows.append(jnp.concatenate([d[s:n_P], jnp.zeros(s, dtype=d.dtype)]))
            else:
                u_rows.append(jnp.zeros(n_P, dtype=dtype))
        u_mat = jnp.stack(u_rows)  # (q, n_P)
        y_rev, diag_rev, u_mat_rev = y[rev], diag_d[rev], u_mat[:, rev].T  # (n_P, q)

        def step(window: Array, xs: tuple) -> tuple[Array, Array]:
            yi, u_i, dii = xs  # scalar, (q,), scalar
            xi = (yi - jnp.dot(u_i, window)) / dii
            return jnp.concatenate([xi[None], window[:-1]]), xi

        carry0 = jnp.zeros(q, dtype=y.dtype)
        _, xs_out = jax.lax.scan(step, carry0, (y_rev, u_mat_rev, diag_rev))
        return xs_out[rev]

    def _solve_one(L_data: Array, U_data: Array, b: Array) -> Array:
        return _bwd_sub(U_data, _fwd_elim(L_data, b))

    return jax.jit(jax.vmap(_solve_one))


class TPMatricesWavenumberSolver:
    """Per-wavenumber solver for Fourier x polynomial tensor-product systems.

    Solves

    .. math::

        \\sum_i s_i \\bigl(A_i^{(0)} \\otimes \\cdots\\bigr)\\, x = f

    where all axes except one are *Fourier* (every per-axis matrix is diagonal)
    and exactly one axis is *polynomial* (banded but not purely diagonal).

    For each combination of Fourier wavenumber indices the 1-D polynomial
    problem

    .. math::

        B_k\\, \\hat{u}_k = \\hat{f}_k, \\quad
        B_k = \\sum_i s_i \\Bigl(\\prod_{a \\in \\text{Fourier}}
        F_i^{(a)}[k_a]\\Bigr)\\, P_i

    is assembled using banded :class:`~jaxfun.la.DiaMatrix` arithmetic and
    pre-factorised with :meth:`~jaxfun.la.DiaMatrix.lu_factor` (result
    cached on each matrix).

    Args:
        poly_axis: Index of the polynomial axis in the full tensor.
        B_matrices: Per-wavenumber :class:`~jaxfun.la.DiaMatrix` objects,
            length ``n_F`` (product of all Fourier-axis sizes), each
            carrying a warm :meth:`~jaxfun.la.DiaMatrix.lu_factor` cache.
        shape: Full solution shape ``(n_0, n_1, ...)``.
    """

    def __init__(
        self,
        poly_axis: int,
        shape: tuple[int, ...],
        B_matrices: list | None = None,
        B_data_batch: Array | None = None,
        poly_offsets: tuple[int, ...] | None = None,
    ) -> None:
        from jaxfun.la.diamatrix import (
            LUFactors as _DiaLUFactors,
            _lu_banded_no_pivot_kernel,
        )

        self.poly_axis = poly_axis
        self.shape = shape

        if B_data_batch is not None and poly_offsets is not None:
            # ---- Fast batched path: one vmapped XLA call for all wavenumbers ----
            # Avoids the O(n_F) Python loop of per-wavenumber B.lu_factor() calls.
            n_F, _n_diags, n_P_local = B_data_batch.shape
            _dtype = B_data_batch.dtype
            p = max((-o for o in poly_offsets if o < 0), default=0)
            q = max((o for o in poly_offsets if o > 0), default=0)
            center = p
            bw = p + q + 1
            all_L_offsets: tuple[int, ...] = tuple(o for o in poly_offsets if o < 0)
            all_U_offsets: tuple[int, ...] = tuple(o for o in poly_offsets if o >= 0)

            def _batch_lu(data: Array) -> tuple[Array, Array]:
                """Batched LU for a slice of B_data_batch.

                Converts DIA format → band, runs all LU factorisations in one
                vmapped XLA call, then extracts L/U diagonal data.

                Args:
                    data: shape (n_batch, n_diags, n_P)

                Returns:
                    (L_data, U_data) each shape (n_batch, n_offsets, n_P)
                """
                n_batch = data.shape[0]
                band_rows = jnp.array([center + off for off in poly_offsets])
                band = (
                    jnp.zeros((n_batch, bw, n_P_local), dtype=_dtype)
                    .at[:, band_rows, :]
                    .set(data)
                )
                band_lu = jax.jit(
                    jax.vmap(lambda b: _lu_banded_no_pivot_kernel(b, p, q, center))
                )(band)
                L = jnp.stack(
                    [band_lu[:, center + off, :] for off in all_L_offsets], axis=1
                )
                U = jnp.stack(
                    [band_lu[:, center + off, :] for off in all_U_offsets], axis=1
                )
                return L, U

            if len(jax.devices()) > 1:
                if poly_axis == 0:
                    raise ValueError(
                        "Multi-process solve requires axis 0 to be a Fourier axis "
                        f"(poly_axis=0 not supported). Got shape={shape}, "
                        f"poly_axis={poly_axis}."
                    )
                n_total = len(jax.devices())
                n_local = jax.local_device_count()
                n_F_per_device = n_F // n_total
                proc_dev_offset = jax.process_index() * n_local
                k_start = proc_dev_offset * n_F_per_device
                k_end = k_start + n_local * n_F_per_device
                _local_L, _local_U = _batch_lu(B_data_batch[k_start:k_end])
                self._L_per_device = [
                    jax.device_put(
                        _local_L[d * n_F_per_device : (d + 1) * n_F_per_device],
                        jax.local_devices()[d],
                    )
                    for d in range(n_local)
                ]
                self._U_per_device = [
                    jax.device_put(
                        _local_U[d * n_F_per_device : (d + 1) * n_F_per_device],
                        jax.local_devices()[d],
                    )
                    for d in range(n_local)
                ]
                self._L_data_local = _local_L
                self._U_data_local = _local_U
            else:
                _local_L, _local_U = _batch_lu(B_data_batch)
                self._L_data_local: Array = _local_L
                self._U_data_local: Array = _local_U

        else:
            # ---- Legacy path: per-wavenumber Python loop ----------------------
            # Kept for backward compatibility when B_data_batch is not provided.
            assert B_matrices is not None, (
                "Either B_data_batch+poly_offsets or B_matrices must be provided."
            )
            n_P_local = B_matrices[0].shape[0]
            _all_offsets = sorted({off for B in B_matrices for off in B.offsets})
            all_L_offsets = tuple(o for o in _all_offsets if o < 0)
            all_U_offsets = tuple(o for o in _all_offsets if o >= 0)

            def _align_data(
                dia_mat: DiaMatrix, target_offsets: tuple[int, ...]
            ) -> Array:
                rows: list[Array] = []
                for off in target_offsets:
                    if off in dia_mat.offsets:
                        rows.append(dia_mat.data[list(dia_mat.offsets).index(off)])
                    else:
                        rows.append(jnp.zeros(n_P_local, dtype=dia_mat.data.dtype))
                return jnp.stack(rows)

            n_F = len(B_matrices)

            if len(jax.devices()) > 1:
                if poly_axis == 0:
                    raise ValueError(
                        "Multi-process solve requires axis 0 to be a Fourier axis "
                        f"(poly_axis=0 not supported). Got shape={shape}, "
                        f"poly_axis={poly_axis}."
                    )
                n_total = len(jax.devices())
                n_local = jax.local_device_count()
                n_F_per_device = n_F // n_total
                proc_dev_offset = jax.process_index() * n_local
                k_start = proc_dev_offset * n_F_per_device
                k_end = k_start + n_local * n_F_per_device
                _local_lu = [B.lu_factor() for B in B_matrices[k_start:k_end]]
                _local_L = jnp.stack(
                    [_align_data(lu.L, all_L_offsets) for lu in _local_lu]
                )
                _local_U = jnp.stack(
                    [_align_data(lu.U, all_U_offsets) for lu in _local_lu]
                )
                self._L_per_device = [
                    jax.device_put(
                        _local_L[d * n_F_per_device : (d + 1) * n_F_per_device],
                        jax.local_devices()[d],
                    )
                    for d in range(n_local)
                ]
                self._U_per_device = [
                    jax.device_put(
                        _local_U[d * n_F_per_device : (d + 1) * n_F_per_device],
                        jax.local_devices()[d],
                    )
                    for d in range(n_local)
                ]
                self._L_data_local = _local_L
                self._U_data_local = _local_U
            else:
                _local_lu = [B.lu_factor() for B in B_matrices]
                _local_L = jnp.stack(
                    [_align_data(lu.L, all_L_offsets) for lu in _local_lu]
                )
                _local_U = jnp.stack(
                    [_align_data(lu.U, all_U_offsets) for lu in _local_lu]
                )
                self._L_data_local: Array = _local_L
                self._U_data_local: Array = _local_U

        self.L_offsets: tuple[int, ...] = all_L_offsets
        self.U_offsets: tuple[int, ...] = all_U_offsets

        self._vmap_solve = _make_wavenumber_vmap_solve(
            all_L_offsets, all_U_offsets, n_P_local, self._L_data_local.dtype
        )

        # Build _lu_stacked directly from the already-stacked _local_L/_local_U.
        # This avoids the O(n_F) Python loop + jax.tree.map(stack) of the old
        # approach, which was a significant contributor to startup latency.
        n = n_P_local
        self._lu_stacked = _DiaLUFactors(
            L=DiaMatrix(data=_local_L, offsets=all_L_offsets, shape=(n, n)),
            U=DiaMatrix(data=_local_U, offsets=all_U_offsets, shape=(n, n)),
            shape=(n, n),
            perm=None,
        )
        self._vmap_solve2 = jax.jit(jax.vmap(lambda lu, b: lu.solve(b)))

        # Build per-instance JIT'd solve functions.  L/U data are fixed after
        # init and captured as closure constants; only rhs is a dynamic argument.
        _vmap_fn = self._vmap_solve
        _poly_axis = poly_axis
        _ndim = len(shape)
        _fourier_axes = [a for a in range(_ndim) if a != _poly_axis]
        _fourier_shape = tuple(shape[a] for a in _fourier_axes)
        _n_F = int(np.prod(_fourier_shape)) if _fourier_shape else 1
        _n_P = shape[_poly_axis]
        _axes_order = _fourier_axes + [_poly_axis]
        _inv_perm = [0] * _ndim
        for _new_pos, _old_pos in enumerate(_axes_order):
            _inv_perm[_old_pos] = _new_pos

        if len(jax.devices()) > 1:
            # One JIT per local device, each closing over that device's L/U
            # slice.  All JITs are dispatched independently so XLA can
            # schedule them concurrently across local devices.
            _n_total = len(jax.devices())
            _n_local = jax.local_device_count()
            _n_F_per_device = _n_F // _n_total
            _fourier_shape_per_device = (
                _fourier_shape[0] // _n_total,
            ) + _fourier_shape[1:]

            # Factory avoids the Python late-binding closure pitfall.
            def _make_device_jit(L_d: Array, U_d: Array):
                @jax.jit
                def _jit(rhs_d: Array) -> Array:
                    rhs_2d = jnp.transpose(rhs_d, _axes_order).reshape(
                        _n_F_per_device, _n_P
                    )
                    sol_2d = _vmap_fn(L_d, U_d, rhs_2d)
                    sol_perm = sol_2d.reshape(_fourier_shape_per_device + (_n_P,))
                    return jnp.transpose(sol_perm, _inv_perm)

                return _jit

            self._local_solve_jits = [
                _make_device_jit(self._L_per_device[d], self._U_per_device[d])
                for d in range(_n_local)
            ]

        else:

            @jax.jit
            def _solve_jit(rhs: Array) -> Array:
                rhs_2d = jnp.transpose(rhs, _axes_order).reshape(_n_F, _n_P)
                sol_2d = _vmap_fn(_local_L, _local_U, rhs_2d)
                sol_perm = sol_2d.reshape(_fourier_shape + (_n_P,))
                return jnp.transpose(sol_perm, _inv_perm)

            self._solve_jit = _solve_jit

    def solve(self, rhs: Array) -> Array:
        """Solve the wavenumber-loop system for RHS ``rhs``.

        All per-wavenumber 1-D banded polynomial solves are executed in a
        single :func:`jax.vmap` call over the stacked ``L`` / ``U`` factor
        data arrays.  The scan kernels are compiled once on the first call
        and reused for subsequent solves.

        In multi-process mode ``rhs`` must carry sharding ``P("k", None, None)``
        so that each process holds a contiguous block of Fourier wavenumber
        rows.  The reshape, local vmap, and global assembly are all
        communication-free.

        Args:
            rhs: Right-hand side shaped ``self.shape``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        if len(jax.devices()) > 1:
            # Dispatch one JIT per local device (JAX async — XLA schedules
            # them concurrently).  Each JIT is communication-free and closes
            # over its own L/U slice already placed on that device.
            n_local = jax.local_device_count()
            results = [
                self._local_solve_jits[d](rhs.addressable_data(d))
                for d in range(n_local)
            ]
            if n_local == 1:
                return results[0]
            # Gather to device 0 (intra-host copy on CPU) and concatenate
            # to form the full process-local result for the caller.
            return jnp.concatenate(
                [jax.device_put(r, jax.local_devices()[0]) for r in results],
                axis=0,
            )
        return self._solve_jit(rhs)

    @jax.jit(static_argnums=(0,))
    def solve2(self, rhs: Array) -> Array:
        """Alternative solve that vmaps :meth:`~jaxfun.la.diamatrix.LUFactors.solve`
        directly over a batched :class:`~jaxfun.la.diamatrix.LUFactors` pytree.

        All ``LUFactors`` for the wavenumber batch share the same aligned
        ``L``/``U`` offsets and ``shape``, so their leaves can be stacked once
        (in ``__init__``) and ``jax.vmap`` maps ``lu.solve(b)`` over the
        leading batch axis — no custom scan kernels required.

        Args:
            rhs: Right-hand side shaped ``self.shape``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        shape = self.shape
        ndim = len(shape)
        poly_axis = self.poly_axis
        n_P = shape[poly_axis]

        fourier_axes = [a for a in range(ndim) if a != poly_axis]
        fourier_shape = tuple(shape[a] for a in fourier_axes)
        n_F = int(np.prod(fourier_shape)) if fourier_shape else 1

        axes_order = fourier_axes + [poly_axis]
        rhs_2d = jnp.transpose(rhs, axes_order).reshape(n_F, n_P)

        sol_2d = self._vmap_solve2(self._lu_stacked, rhs_2d)

        sol_perm = sol_2d.reshape(fourier_shape + (n_P,))
        inv_perm = [0] * ndim
        for new_pos, old_pos in enumerate(axes_order):
            inv_perm[old_pos] = new_pos
        return jnp.transpose(sol_perm, inv_perm)


class TPMatricesDenseLUFactors:
    """Dense Kronecker-product LU solver for a sum of :class:`TPMatrix`.

    Assembles the full (dense) Kronecker product ``sum_k s_k A_k^(0) ⊗ …``
    into a single :class:`~jaxfun.la.Matrix`, LU-factorizes it once, and
    solves by a single triangular-substitution call.

    This is the appropriate solver when all per-axis factor matrices are
    dense :class:`~jaxfun.la.Matrix` instances.  It imposes no structural
    requirement on the system (unlike the diagonalization-based
    :class:`TPMatricesLUFactors` which requires simultaneously diagonalizable
    factor matrices).

    Attributes:
        lu: Pre-computed :class:`~jaxfun.la.matrix.LUFactors` of the full
            assembled Kronecker product.
        shape: Per-axis sizes ``(n0, n1, …)``.
    """

    def __init__(self, lu: LUFactors, shape: tuple[int, ...]) -> None:
        self.lu = lu
        self.shape = shape

    @jax.jit(static_argnums=(0,))
    def solve(self, rhs: Array) -> Array:
        """Solve the summed tensor-product system for RHS ``rhs``.

        Args:
            rhs: Right-hand side, flat ``(n,)`` or shaped ``(n0, n1, …)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        return self.lu.solve(rhs.ravel()).reshape(rhs.shape)


def tpmats_dense_lu_factor(
    A: TPMatrix | list[TPMatrix],
) -> TPMatricesDenseLUFactors:
    """Assemble and LU-factorize the dense Kronecker product of a :class:`TPMatrices`.

    Sums all Kronecker-product terms into a single dense
    :class:`~jaxfun.la.Matrix` and computes its LU factorisation.  This is
    the simplest solver and is appropriate when all per-axis factor matrices
    are dense :class:`~jaxfun.la.Matrix` instances.

    Args:
        A: Single :class:`TPMatrix` or list thereof (as returned by
           :func:`~jaxfun.galerkin.inner.inner`).

    Returns:
        :class:`TPMatricesDenseLUFactors` whose :meth:`~TPMatricesDenseLUFactors.solve`
        method solves the system without re-factorising.

    Raises:
        TypeError: if any factor matrix is not a :class:`~jaxfun.la.Matrix`.
    """
    if isinstance(A, TPMatrix):
        A = [A]
    tpmats = list(A)
    for tp in tpmats:
        for mat in tp.mats:
            if not isinstance(mat, Matrix):
                raise TypeError(
                    f"tpmats_dense_lu_factor requires all factor matrices to be "
                    f"Matrix (dense); got {type(mat).__name__}."
                )
    mat = tpmats_to_kron(tpmats)
    assert isinstance(mat, Matrix)
    shape = tuple(int(tpmats[0].mats[i].shape[0]) for i in range(tpmats[0].dims))
    return TPMatricesDenseLUFactors(lu=mat.lu_factor(), shape=shape)


def tpmats_lu_factor(A: TPMatrix | list[TPMatrix]) -> TPMatricesLUFactors:
    """Compute diagonalization-based LU factors for a sum of :class:`TPMatrix`.

    Simultaneously diagonalizes the factor matrices on each axis so that the
    full Kronecker-sum system reduces to element-wise division in the
    transformed space.

    **Algorithm** (2D, generalises to any number of dims):

    Given a list of TPMatrices representing :math:`\\sum_k s_k A_k \\otimes B_k`,
    find :math:`V` such that :math:`V^T A V = \\Lambda_A` and
    :math:`V^T B V = I` (generalized eigenproblem :math:`A v = \\lambda B v`).
    Then:

    .. math::

        \\tilde{F} = V^T F V, \\quad
        D_{ij} = \\textstyle\\sum_k s_k \\lambda_k^{(0)}{}_i \\lambda_k^{(1)}{}_j,
        \\quad U = V (\\tilde{F} / D) V^T.

    **Requirement**: all factor matrices on each axis must be simultaneously
    diagonalizable — true whenever each axis has at most 2 distinct matrices
    that form a symmetric-definite pair (e.g. stiffness K and mass M from the
    same 1D function space).  Axes that share the same unordered matrix pair
    automatically reuse the same eigenvectors.

    Args:
        A: Single :class:`TPMatrix` or list of :class:`TPMatrix` objects (as
            returned by :func:`~jaxfun.galerkin.inner.inner`).

    Returns:
        :class:`TPMatricesLUFactors` whose :meth:`~TPMatricesLUFactors.solve`
        method solves the system without re-factorising.

    Raises:
        ValueError: if any axis has more than 2 distinct factor matrices.
    """
    if isinstance(A, TPMatrix):
        A = [A]
    tpmats = list(A)
    ndim = tpmats[0].dims

    # --- value-based deduplication of factor matrices ----------------------
    # Matrices that are numerically equal but have different Python ids (e.g.
    # M from K⊗M and M from the M⊗M term in a Helmholtz problem) are treated
    # as the same matrix.  All ids are mapped to a single representative id.
    _mat_by_id: dict[int, object] = {}
    _dense_by_id: dict[int, Array] = {}
    for tp in tpmats:
        for mat in tp.mats:
            mid = id(mat)
            if mid not in _mat_by_id:
                _mat_by_id[mid] = mat
                _dense_by_id[mid] = mat.todense()

    _seen_repr: list[int] = []  # canonical ids in first-seen order
    _id_to_repr: dict[int, int] = {}
    for mid in _mat_by_id:
        for rid in _seen_repr:
            if _dense_by_id[mid].shape == _dense_by_id[rid].shape and jnp.allclose(
                _dense_by_id[mid], _dense_by_id[rid], rtol=1e-5, atol=1e-8
            ):
                _id_to_repr[mid] = rid
                break
        else:
            _id_to_repr[mid] = mid
            _seen_repr.append(mid)

    def _repr(mat) -> int:
        return _id_to_repr[id(mat)]

    # --- per-axis pair → (eigvecs, {repr_id: eigenvalues}) ----------------
    # Axes that share the same unordered pair of matrices reuse eigenvectors.
    pair_cache: dict[frozenset, tuple[Array, dict[int, Array]]] = {}

    for i in range(ndim):
        mats_i = list(
            {_repr(tp.mats[i]): _mat_by_id[_repr(tp.mats[i])] for tp in tpmats}.values()
        )
        pair_key = frozenset(_repr(m) for m in mats_i)
        if pair_key in pair_cache:
            continue
        if len(mats_i) == 1:
            A_dense = cast(MatrixProtocol, mats_i[0]).todense()
            evals, evecs = jnp.linalg.eigh(A_dense)
            pair_cache[pair_key] = (evecs, {_repr(mats_i[0]): evals})
        elif len(mats_i) == 2:
            import numpy as _np
            import scipy.linalg as _scipy_linalg

            A0_np = _np.array(cast(MatrixProtocol, mats_i[0]).todense())
            A1_np = _np.array(cast(MatrixProtocol, mats_i[1]).todense())
            # Generalized eigenproblem: try A0 v = λ A1 v (A1 must be PD).
            # If that fails (A1 not PD), swap to A1 v = λ A0 v.
            try:
                evals_np, evecs_np = _scipy_linalg.eigh(A0_np, A1_np)
                evals = jnp.array(evals_np)
                evecs = jnp.array(evecs_np)
                pair_cache[pair_key] = (
                    evecs,
                    {
                        _repr(mats_i[0]): evals,
                        _repr(mats_i[1]): jnp.ones_like(evals),
                    },
                )
            except _scipy_linalg.LinAlgError:
                evals_np, evecs_np = _scipy_linalg.eigh(A1_np, A0_np)
                evals = jnp.array(evals_np)
                evecs = jnp.array(evecs_np)
                pair_cache[pair_key] = (
                    evecs,
                    {
                        _repr(mats_i[1]): evals,
                        _repr(mats_i[0]): jnp.ones_like(evals),
                    },
                )
        else:
            raise SolverNotApplicable(
                f"Axis {i} has {len(mats_i)} distinct factor matrices; "
                "simultaneous diagonalization requires ≤ 2 distinct matrices per axis."
            )

    # Build per-axis eigenvector list and global repr_id→eigenvalues map.
    eigvecs: list[Array] = []
    axis_eigenvalues: dict[int, Array] = {}
    for i in range(ndim):
        mats_i = list(
            {_repr(tp.mats[i]): _mat_by_id[_repr(tp.mats[i])] for tp in tpmats}.values()
        )
        pair_key = frozenset(_repr(m) for m in mats_i)
        evecs, evals_map = pair_cache[pair_key]
        eigvecs.append(evecs)
        axis_eigenvalues.update(evals_map)

    per_term_eigenvalues = [
        [axis_eigenvalues[_repr(tp.mats[i])] for i in range(ndim)] for tp in tpmats
    ]
    scales: list[complex] = [tp.scale for tp in tpmats]
    shape: tuple[int, ...] = tuple(int(tpmats[0].mats[i].shape[0]) for i in range(ndim))
    return TPMatricesLUFactors(
        eigvecs=eigvecs,
        per_term_eigenvalues=per_term_eigenvalues,
        scales=scales,
        shape=shape,
    )


def tpmats_wavenumber_factor(
    A: list[TPMatrix] | TPMatrices,
) -> TPMatricesWavenumberSolver:
    """Pre-factorize a Fourier x polynomial :class:`TPMatrices` system.

    Detects which axes are Fourier (every term has a purely diagonal
    :class:`~jaxfun.la.DiaMatrix` — ``offsets == (0,)`` — on that axis) and
    which is the polynomial axis (banded but not purely diagonal).

    For each Fourier wavenumber index ``k`` assembles the 1-D banded
    polynomial system

    .. math::

        B_k = \\sum_i s_i \\Bigl(\\prod_{a \\in \\text{Fourier}}
        F_i^{(a)}[k_a]\\Bigr)\\, P_i

    as a :class:`~jaxfun.la.DiaMatrix` (preserving the banded sparsity
    pattern of the polynomial matrices) and warms its
    :meth:`~jaxfun.la.DiaMatrix.lu_factor` cache.

    Args:
        A: :class:`list` of :class:`TPMatrix` (as returned by
            :func:`~jaxfun.galerkin.inner.inner`) or a
            :class:`TPMatrices` instance.

    Returns:
        :class:`TPMatricesWavenumberSolver` for repeated fast solves.

    Raises:
        TypeError: If ``A`` is not a ``list[TPMatrix]`` or
            :class:`TPMatrices`.
        ValueError: If the structure does not have exactly one non-diagonal
            (polynomial) axis, e.g. for fully symmetric problems where
            :func:`tpmats_lu_factor` should be used instead.
    """
    if isinstance(A, TPMatrices):
        tpmats: list[TPMatrix] = list(A.tpmats)
    elif isinstance(A, list):
        tpmats = A
    else:
        raise TypeError(
            f"tpmats_wavenumber_factor expects a list[TPMatrix] or TPMatrices, "
            f"got {type(A).__name__!r}."
        )
    ndim: int = tpmats[0].dims

    def _is_diagonal_axis(axis: int) -> bool:
        return all(set(cast(DiaMatrix, tp.mats[axis]).offsets) == {0} for tp in tpmats)

    fourier_axes = [a for a in range(ndim) if _is_diagonal_axis(a)]
    poly_axes = [a for a in range(ndim) if not _is_diagonal_axis(a)]

    if len(poly_axes) != 1:
        raise SolverNotApplicable(
            f"tpmats_wavenumber_factor requires exactly 1 polynomial "
            f"(non-diagonal) axis; found {len(poly_axes)}: {poly_axes}. "
            f"Use tpmats_lu_factor for fully-symmetric problems."
        )

    poly_axis = poly_axes[0]
    shape = tuple(int(tpmats[0].mats[a].shape[0]) for a in range(ndim))
    n_P = shape[poly_axis]

    # Determine working dtype from the polynomial-axis matrices so the solver
    # honours float64 when JAX is configured for 64-bit precision.
    _dtype = jnp.result_type(*[tp.mats[poly_axis].data.dtype for tp in tpmats])

    # Build weight matrix W[i, k] = scale_i * prod_a(diag(F_i^(a))[k_a]).
    # The flat Fourier index k varies in C-order (last Fourier axis fastest),
    # matching the transposed layout used in TPMatricesWavenumberSolver.solve.
    W_list: list[Array] = []
    for tp in tpmats:
        w: Array = jnp.asarray(tp.scale, dtype=_dtype).reshape(1)
        for a in fourier_axes:
            # Diagonal DiaMatrix: data has shape (1, n_a); data[0] is the diagonal.
            diag_a = jnp.asarray(tp.mats[a].data[0], dtype=_dtype)  # (n_a,)
            w = jnp.outer(w, diag_a).flatten()  # 1 → n_{a0} → n_{a0}*n_{a1} → …
        W_list.append(w)  # (n_F,)

    W = jnp.stack(W_list)  # (n_terms, n_F)

    # Union of offsets across all polynomial matrices, in sorted order.
    poly_offsets: tuple[int, ...] = tuple(
        sorted(
            {
                int(off)
                for tp in tpmats
                for off in cast(DiaMatrix, tp.mats[poly_axis]).offsets
            }
        )
    )

    # Stack polynomial DIA data aligned to poly_offsets.
    # P_data_stack[i, d, :] = data of term i for offset poly_offsets[d].
    P_data_rows: list[Array] = []
    for tp in tpmats:
        mat = cast(DiaMatrix, tp.mats[poly_axis])
        rows: list[Array] = []
        for off in poly_offsets:
            if off in mat.offsets:
                idx = list(mat.offsets).index(off)
                rows.append(jnp.asarray(mat.data[idx], dtype=_dtype))
            else:
                rows.append(jnp.zeros(n_P, dtype=_dtype))
        P_data_rows.append(jnp.stack(rows))  # (n_diags, n_P)

    P_data_stack = jnp.stack(P_data_rows)  # (n_terms, n_diags, n_P)

    # Assemble per-wavenumber DIA data:
    # B_data_batch[k, d, :] = sum_i W[i,k] * P_data_stack[i, d, :].
    B_data_batch = jnp.einsum("tf,tdp->fdp", W, P_data_stack)  # (n_F, n_diags, n_P)

    return TPMatricesWavenumberSolver(
        poly_axis=poly_axis,
        shape=shape,
        B_data_batch=B_data_batch,
        poly_offsets=poly_offsets,
    )


def tpmats_to_kron(A: TPMatrix | list[TPMatrix], tol: int = 100) -> Matrix | DiaMatrix:
    """Return summed Kronecker expansion of a (list of) TPMatrix.

    Args:
        A: :class:`TPMatrix` or list of :class:`TPMatrix` objects with identical
            result shape.
        tol: Near-zero elimination tolerance applied to dense factor matrices
            before Kronecker expansion.

    Returns:
        :class:`~jaxfun.la.DiaMatrix` or :class:`~jaxfun.la.Matrix` representing
            the summed Kronecker expansion of the input TPMatrix objects.
    """

    if isinstance(A, TPMatrix):
        A = [A]

    if not A:
        raise ValueError("tpmats_to_kron requires a non-empty argument.")

    if isinstance(A[0].mats[0], Matrix):
        result: Array | None = None
        for tpm in A:
            a0 = tpm.mats[0].todense()
            a0 = a0 * jnp.asarray(tpm.scale)
            for m in tpm.mats[1:]:
                a0 = jnp.kron(a0, m.todense())
            result = a0 if result is None else result + a0
        assert result is not None
        return Matrix(result)

    def _get_dia(mat: MatrixProtocol) -> DiaMatrix:
        if isinstance(mat, Matrix):
            return DiaMatrix.from_dense(mat.todense(), tol=tol)
        assert isinstance(mat, DiaMatrix)
        return mat

    result: DiaMatrix | None = None
    for tpm in A:
        dmat: DiaMatrix = _get_dia(tpm.mats[0]) * jnp.asarray(tpm.scale)
        for m in tpm.mats[1:]:
            dmat = diakron(dmat, _get_dia(m))
        dmat = dmat
        result = dmat if result is None else result + dmat
    assert result is not None
    return result


@overload
def vec(A: Array, tol: int = 100) -> Array: ...
@overload
def vec(A: TPMatrix, tol: int = 100) -> Matrix | DiaMatrix: ...
@overload
def vec(A: list[TPMatrix], tol: int = 100) -> Matrix | DiaMatrix: ...
def vec(
    A: Array | TPMatrix | list[TPMatrix], tol: int = 100
) -> Array | Matrix | DiaMatrix:
    """Vectorize array or TPMatrix objects.

    Args:
        A: Dense :class:`jax.Array`, :class:`TPMatrix`, or list of :class:`TPMatrix`
            objects.
        tol: Near-zero elimination tolerance (only used for TPMatrix objects).

    Returns:
        Flattened :class:`jax.Array` or the summed Kronecker expansion as a
        :class:`~jaxfun.la.DiaMatrix`.
    """
    if not isinstance(A, Array):
        return tpmats_to_kron(A, tol=tol)

    return A.flatten()


def tpmats_to_scipy_sparse(
    A: list[TPMatrix], tol: int = 1
) -> list[tuple[scipy_sparse.csc_array, ...]]:
    """Convert list of separable TPMatrix to scipy CSC factors.

    The :attr:`~TPMatrix.scale` is folded into the first factor matrix.

    Args:
        A: List of TPMatrix objects.
        tol: Near-zero elimination tolerance.

    Returns:
        List of tuples of per-axis scipy csc_array matrices.
    """
    from jaxfun.utils.common import eliminate_near_zeros

    result = []
    for a in A:
        scale = a.scale
        factors = []
        for i, mat in enumerate(a.mats):
            dense = eliminate_near_zeros(mat.todense(), tol)
            if i == 0:
                dense = dense * scale
            factors.append(scipy_sparse.csc_array(dense))
        result.append(tuple(factors))
    return result


def tpmats_to_scipy_kron(A: list[TPMatrix], tol: int = 1) -> scipy_sparse.csc_matrix:
    """Return summed global scipy sparse matrix (Kronecker expansion).

    Args:
        A: List of TPMatrix objects.
        tol: Near-zero elimination tolerance.

    Returns:
        scipy.sparse.csc_matrix representing Σ kron(factors).
    """
    a = tpmats_to_scipy_sparse(A, tol=tol)
    if len(a[0]) == 2:
        return np.sum([scipy_sparse.kron(b[0], b[1], format="csc") for b in a])
    else:
        return np.sum(
            [
                scipy_sparse.kron(
                    scipy_sparse.kron(b[0], b[1], format="csc"), b[2], format="csc"
                )
                for b in a
            ]
        )
