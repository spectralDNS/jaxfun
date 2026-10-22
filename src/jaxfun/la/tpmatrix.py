from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple, cast, overload

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array, shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from scipy import linalg as scipy_linalg, sparse as scipy_sparse

from jaxfun.la.diamatrix import (
    DiaMatrix,
    _affine_prefix,
    _lu_banded_kernel,
    _pivoted_forward,
    _use_prefix_substitution,
    diakron,
)
from jaxfun.la.matrix import LUFactors, Matrix
from jaxfun.la.matrixprotocol import (
    BaseMatrix,
    DiaMatrixSolveMethod,
    SolverNotApplicable,
    _CacheBox,
)

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction

type _SparseMatrixCache = _CacheBox[DiaMatrix]
type _DenseMatrixCache = _CacheBox[Matrix]
type _TPMatricesFactors = (
    TPMatricesDenseLUFactors
    | TPMatricesLUFactors
    | TPMatricesWavenumberSolver
    | TPMatricesSchurSolver
)


class TPMatrix(BaseMatrix):  # noqa: B903
    """Rank-d separable tensor product operator A = kron(A0, A1, ...).

    Provides efficient matvec via successive multiplications instead of
    forming the full Kronecker product explicitly.

    Attributes:
        mats: List of per-axis sparse/dense matrices.
        coefficient: Scalar scaling (multiplicative).
        global_indices: Tuple of global index into vectorized expansions.
    """

    is_zero = False

    def __init__(
        self,
        mats: Sequence[BaseMatrix],
        scale: complex | Array,
        global_indices: tuple[int, int] = (0, 0),
    ) -> None:
        self.mats = nnx.List(mats)
        self.coefficient = scale
        self.global_indices = global_indices

    @property
    def dims(self) -> int:
        return len(self.mats)

    @property
    def shape(self) -> tuple[int, int]:
        rows = int(np.prod([mat.shape[0] for mat in self.mats]))
        cols = int(np.prod([mat.shape[1] for mat in self.mats]))
        return rows, cols

    @property
    def dtype(self) -> jnp.dtype:
        dtype = jnp.result_type(self.coefficient)
        for mat in self.mats:
            dtype = jnp.result_type(dtype, mat.dtype)
        return jnp.dtype(dtype)

    def __len__(self) -> int:
        return len(self.mats)

    def scale(self, alpha: complex | Array) -> TPMatrix:
        return _scale_tpmatrix(self, alpha)

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        sparse_box: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
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
        dense_box: _DenseMatrixCache | None = getattr(self, "_dense_cache", None)
        if dense_box is not None:
            return dense_box.value.todense()
        sparse_box: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
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

    @property
    def is_diagonal(self) -> bool:
        """Whether every factor is purely main-diagonal."""
        return self.diagonal_or_none() is not None

    def diagonal_or_none(self) -> Array | None:
        """Return the tensor-product diagonal when every factor is diagonal."""
        if len(self.mats) == 0:
            return None

        diagonals: list[Array] = []
        for mat in self.mats:
            diagonal = mat.diagonal_or_none()
            if diagonal is None or diagonal.ndim != 1:
                return None
            diagonals.append(diagonal)

        diagonal = diagonals[0]
        for axis, factor_diag in enumerate(diagonals[1:], start=1):
            shape = (1,) * axis + (factor_diag.shape[0],)
            diagonal = diagonal[..., None] * factor_diag.reshape(shape)
        return diagonal * jnp.asarray(self.coefficient)

    def _matmul_array(self, w: Array) -> Array:
        result = w
        for i, mat in enumerate(self.mats):
            result = mat.matvec(result, axis=i)
        return result * jnp.asarray(self.coefficient)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply matrix to rank-2 coefficient array u."""
        w = self._as_array(u)
        return self._matmul_array(cast(Array, w))

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def _rmatmul_array(self, w: Array) -> Array:
        result = w
        for i, mat in enumerate(self.mats):
            result = mat.T.matvec(result, axis=i)
        return result * jnp.asarray(self.coefficient)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        w = cast(Array, self._as_array(u))
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
        diagonal = self.diagonal_or_none()
        if diagonal is not None:
            return _solve_diagonal(diagonal, rhs)
        return self.lu_factor().solve(rhs)

    def lu_factor(self) -> TPLUFactors:
        """Pre-compute LU factors for every Kronecker factor.

        Returns a :class:`TPLUFactors` whose :meth:`~TPLUFactors.solve` method
        solves the Kronecker system without rebuilding the factorisation. The
        result is cached, as in :meth:`TPMatrices.lu_factor`, so that repeated
        solves against the same operator reuse one factorisation -- and one set
        of `TPLUFactors`/`LUFactors` instances, which the jit caches of their
        `solve` methods are keyed on.

        The cache is a tracked attribute rather than an opaque `_CacheBox`, so
        the factors are pytree leaves of the operator; see `TPLUFactors`.
        """
        cached: TPLUFactors | None = getattr(self, "_lu_cache", None)
        if cached is not None:
            return cached
        lu_factors = [mat.lu_factor() for mat in self.mats]
        shape = tuple(int(mat.shape[0]) for mat in self.mats)
        self._lu_cache = nnx.data(
            TPLUFactors(lu_factors=lu_factors, scale=self.coefficient, shape=shape)
        )
        return self._lu_cache

    def __add__(self, other):
        if isinstance(other, TPMatrix):
            return TPMatrices([self, other])
        if isinstance(other, TPMatrices):
            return TPMatrices([self, *list(other.tpmats)])
        return NotImplemented


class TPLUFactors(nnx.Pytree):
    """LU factorisation of a :class:`TPMatrix` (Kronecker product).

    Holds the per-factor LU objects and applies them sequentially on their
    respective axes to solve the full tensor-product system.

    A pytree, like the `LUFactors` it is built from, so the factor arrays are
    leaves rather than static payload and `solve` can take `self` as an ordinary
    traced argument.

    Attributes:
        lu_factors: Per-axis LU factorisation objects (DiaMatrix or Matrix).
        scale: Scalar from the parent :class:`TPMatrix`. Static, as
            `TPMatrix.coefficient` is, so it still constant-folds.
        shape: Tuple of per-factor sizes ``(n0, n1, …)``.
    """

    def __init__(
        self, lu_factors: list, scale: complex | Array, shape: tuple[int, ...]
    ) -> None:
        self.lu_factors = nnx.List(lu_factors)
        self.scale = scale
        self.shape = shape

    @jax.jit
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
        LU: Force the factored path (wavenumber, diagonalization or Schur
            solver). Propagates :exc:`ValueError` if the factor-matrix
            structure is not suitable.
        KRON: Force explicit Kronecker product assembly.  The assembled
            :class:`~jaxfun.la.DiaMatrix` or :class:`~jaxfun.la.Matrix` is
            cached; the DIA-matrix solver is selected via *kron_method* in
            :meth:`TPMatrices.solve`.
    """

    AUTO = "auto"
    LU = "lu"
    KRON = "kron"


class DistinctFactors(NamedTuple):
    """A sum of tensor products written over its distinct factor matrices.

    The operator is
    ``sum_k scales[k] * kron(*(matrices[i] for i in indices[k]))``, where no
    two of `matrices` are equal up to a scalar. See
    :meth:`TPMatrices.distinct_factors`.
    """

    matrices: list[BaseMatrix]
    indices: list[tuple[int, ...]]
    scales: list[complex | Array]


def _index_up_to_scalar(
    mat: BaseMatrix, matrices: list[BaseMatrix]
) -> tuple[int, complex]:
    """Index of the matrix in `matrices` that `mat` is a multiple of, and how much.

    `mat` is appended, as its own multiple 1, when it is a multiple of none.
    """
    for r, rep in enumerate(matrices):
        c = mat.proportionality(rep)
        if c is not None:
            return r, c
    matrices.append(mat)
    return len(matrices) - 1, 1.0


class TPMatrices(BaseMatrix):
    """Container for list of TPMatrix bilinear operator tensors."""

    is_zero = False

    def __init__(self, tpmats: list[TPMatrix]) -> None:
        self.tpmats = nnx.List(tpmats)

    @jax.jit
    def _apply_array(self, u: Array) -> Array:
        return jnp.sum(jnp.array([mat._matmul_array(u) for mat in self.tpmats]), axis=0)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply summed tensor product operator to u."""
        w = self._as_array(u)
        return self._apply_array(w)

    def __len__(self) -> int:
        """Return number of TPMatrix terms."""
        return len(self.tpmats)

    @property
    def shape(self) -> tuple[int, int]:
        if len(self.tpmats) == 0:
            return (0, 0)
        return self.tpmats[0].shape

    @property
    def dtype(self) -> jnp.dtype:
        dtype = jnp.float32
        for mat in self.tpmats:
            dtype = jnp.result_type(dtype, mat.dtype)
        return jnp.dtype(dtype)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        w = cast(Array, self._as_array(u))
        return jnp.sum(
            jnp.array([mat._rmatmul_array(w) for mat in self.tpmats]), axis=0
        )

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        sparse_box: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
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
        dense_box: _DenseMatrixCache | None = getattr(self, "_dense_cache", None)
        if dense_box is not None:
            return dense_box.value.todense()
        sparse_box: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
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

    @property
    def is_diagonal(self) -> bool:
        """Whether every summed tensor-product term is purely diagonal."""
        return self.diagonal_or_none() is not None

    def diagonal_or_none(self) -> Array | None:
        """Return the summed tensor-product diagonal when all terms are diagonal."""
        diagonal_sum: Array | None = None
        for mat in self.tpmats:
            diagonal = mat.diagonal_or_none()
            if diagonal is None:
                return None
            diagonal_sum = diagonal if diagonal_sum is None else diagonal_sum + diagonal
        return diagonal_sum

    def scale(self, alpha: complex | Array) -> TPMatrices:
        """Return ``alpha * self`` preserving the summed tensor-product form."""
        return TPMatrices([_scale_tpmatrix(mat, alpha) for mat in self.tpmats])

    def merge(self) -> TPMatrices:
        """Return the same operator with terms that share factors merged.

        Terms whose factors agree, up to a scalar, on all axes but one are
        summed on that axis: ``A ⊗ B + C ⊗ cB`` becomes ``(A + cC) ⊗ B``,
        repeated until no two terms qualify. Merged terms are written over the
        :meth:`distinct_factors`; the others are kept as they are. This
        operator itself when nothing merges, which includes when its values are
        traced and cannot be compared.
        """
        merged = _merge_shared_factors(self.tpmats)
        return self if len(merged) == len(self.tpmats) else TPMatrices(merged)

    def distinct_factors(self) -> DistinctFactors:
        """Return the terms written over factor matrices distinct up to a scalar.

        `inner` builds every term's factors separately and folds constants into
        them, so equal matrices arrive as different objects, and the mass matrix
        of an anisotropic Laplacian as ``M``, ``2M`` and ``3M``. Here every
        factor is one of the distinct matrices times a scalar, and the scalars
        join their term's coefficient. Matrices on different axes are shared
        when equal up to a scalar. The values are compared, so the matrices may
        not be traced.
        """
        matrices: list[BaseMatrix] = []
        placed: dict[int, tuple[int, complex]] = {}  # id(factor) -> (index, scalar)
        indices: list[tuple[int, ...]] = []
        scales: list[complex | Array] = []
        for tp in self.tpmats:
            idx: list[int] = []
            scale = tp.coefficient
            for mat in tp.mats:
                if id(mat) not in placed:
                    placed[id(mat)] = _index_up_to_scalar(mat, matrices)
                r, c = placed[id(mat)]
                idx.append(r)
                scale = scale * c
            indices.append(tuple(idx))
            scales.append(scale)
        return DistinctFactors(matrices, indices, scales)

    def __add__(self, other):
        if isinstance(other, TPMatrix):
            return TPMatrices([*list(self.tpmats), other])
        if isinstance(other, TPMatrices):
            return TPMatrices([*list(self.tpmats), *list(other.tpmats)])
        return NotImplemented

    def lu_factor(self) -> _TPMatricesFactors:
        """Pre-compute factors for repeated fast solves.

        Dispatch order:

        1. If all per-axis factor matrices are dense :class:`~jaxfun.la.Matrix`
           instances, use :func:`tpmats_dense_lu_factor` (simple full Kronecker
           LU — works for any linear system).
        2. Otherwise try :func:`tpmats_wavenumber_factor` (efficient for
           Fourier x polynomial problems).
        3. Then :func:`tpmats_lu_factor` (diagonalization — requires at most two
           distinct factor matrices per axis and well-conditioned eigenvectors).
        4. Fall back to :func:`tpmats_schur_factor` (generalized Schur — 2D
           systems of two terms, after merging shared factors).

        The result is cached as a tracked attribute, so the factor arrays are
        pytree leaves of the operator rather than static payload.

        Returns:
            :class:`TPMatricesDenseLUFactors`, :class:`TPMatricesWavenumberSolver`,
            :class:`TPMatricesLUFactors` or :class:`TPMatricesSchurSolver` for
            repeated fast solves.
        """
        cached: _TPMatricesFactors | None = getattr(self, "_lu_cache", None)
        if cached is not None:
            return cached
        # A previous attempt that found no applicable factored solver is cached
        # too. Whether one applies is a property of the matrix structure, so the
        # answer cannot change -- and re-deciding it is not free: the structural
        # inspection below reads matrix values, which a caller that traces
        # `solve` (the time integrators jit their step) cannot do a second time.
        why: str | None = getattr(self, "_lu_not_applicable", None)
        if why is not None:
            raise SolverNotApplicable(why)
        result: _TPMatricesFactors
        tpmats_list = list(self.tpmats)
        try:
            if all(isinstance(mat, Matrix) for tp in tpmats_list for mat in tp.mats):
                result = tpmats_dense_lu_factor(tpmats_list)
            else:
                try:
                    result = tpmats_wavenumber_factor(tpmats_list)
                except SolverNotApplicable:
                    result = _diagonalize_or_schur(tpmats_list)
        except SolverNotApplicable as e:
            object.__setattr__(self, "_lu_not_applicable", str(e))
            raise
        self._lu_cache = nnx.data(result)
        return result

    def solve(
        self,
        rhs: Array,
        *,
        method: TPSolveMethod | str = TPSolveMethod.AUTO,
        kron_method: DiaMatrixSolveMethod | str = DiaMatrixSolveMethod.AUTO,
        auto_threshold: int = 100,
    ) -> Array:
        """Solve the summed tensor-product system.

        Args:
            rhs: Right-hand side array.
            method: High-level solver selection. One of:

                * ``"auto"`` (default) — tries the factored path
                  (:meth:`lu_factor`) first; falls back to explicit Kronecker
                  product assembly if the factor-matrix structure is not
                  suitable (e.g. not simultaneously diagonalizable).
                * ``"lu"`` — force the factored path (wavenumber,
                  diagonalization or Schur solver). Raises :exc:`ValueError` if
                  the structure is not suitable.
                * ``"kron"`` — force explicit Kronecker product assembly.
                  The assembled matrix is cached for repeated solves.

            kron_method: Solver method forwarded to
                :meth:`~jaxfun.la.DiaMatrix.lu_solve` when the Kronecker-product
                path is used and the assembled matrix is a
                :class:`~jaxfun.la.DiaMatrix` (i.e. all factor matrices are
                sparse).  One of ``"auto"``, ``"banded"``, ``"rcm"``,
                ``"dense"``.  Ignored when the assembled matrix is a dense
                :class:`~jaxfun.la.Matrix`.
                auto_threshold: Threshold for the ``"auto"`` method, trading off
                    banded/RCM solvers against dense.  The banded solver is usually
                    faster for small bandwidth, but compile time grows with bandwidth.
                    RCM can reduce bandwidth and enable the banded solver, but adds
                    overhead and is not guaranteed to help.  Default is 100.

        Returns:
            Solution array with the same shape as *rhs*.

        Raises:
            ValueError: If ``method="lu"`` but the factor-matrix structure is
                not suitable for the factored solver.
        """
        diagonal = self.diagonal_or_none()
        if diagonal is not None:
            return _solve_diagonal(diagonal, rhs)

        method = TPSolveMethod(method)

        def _kron_solve(r: Array) -> Array:
            flat = r.flatten()
            # DiaMatrix path: shared cache with tosparse()
            sparse_box: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
            if sparse_box is not None:
                return sparse_box.value.lu_solve(
                    flat, method=kron_method, auto_threshold=auto_threshold
                ).reshape(r.shape)
            # Dense Matrix path
            dense_box: _DenseMatrixCache | None = getattr(self, "_dense_cache", None)
            if dense_box is not None:
                return dense_box.value.solve(flat).reshape(r.shape)
            # No cache yet: compute and store
            kron = tpmats_to_kron(list(self.tpmats))
            if isinstance(kron, DiaMatrix):
                object.__setattr__(self, "_sparse_cache", _CacheBox(kron))
                return kron.lu_solve(
                    flat, method=kron_method, auto_threshold=auto_threshold
                ).reshape(r.shape)
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


class TPMatricesLUFactors(nnx.Pytree):
    """Diagonalization-based solver for a sum of tensor-product operators.

    Solves

    .. math::

        \\sum_k s_k \\, (A_k^{(0)} \\otimes A_k^{(1)} \\otimes \\cdots)\\, x = f

    by simultaneously diagonalizing the factor matrices on each axis.

    Given per-axis matrices :math:`V` and :math:`W` satisfying
    :math:`W A V = \\Lambda` (diagonal) and :math:`W B V = I`, the system
    reduces to element-wise division in the transformed space — :math:`O(n^d)`
    work after the :math:`O(n^3)` per-axis factorisation. For a
    symmetric-definite pair :math:`W = V^T`.

    For 2D Poisson (``K⊗M + M⊗K``): the denominator is
    :math:`D_{ij} = \\lambda_i + \\lambda_j` and the solution is
    :math:`U = V ((W F W^T) / D) V^T`.
    """

    def __init__(
        self,
        eigvecs: list[Array],
        left_eigvecs: list[Array],
        per_term_eigenvalues: list[list[Array]],
        scales: list[complex | Array],
        shape: tuple[int, ...],
        real_operator: bool,
    ) -> None:
        # `nnx.data` on the array containers, so the eigenbasis is a pytree
        # leaf; see `TPLUFactors`.
        self.eigvecs = nnx.data(eigvecs)  # list of (n_i, n_i) V matrices
        self.left_eigvecs = nnx.data(left_eigvecs)  # list of (n_i, n_i) W matrices
        self.per_term_eigenvalues = nnx.data(
            per_term_eigenvalues
        )  # [term][axis] -> (n_axis,)
        self.scales = scales
        self.shape = shape
        # A real operator can still have complex eigenpairs (Petrov-Galerkin);
        # they come in conjugate pairs, so a real rhs has a real solution.
        self.real_operator = real_operator

    @jax.jit
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

        # Forward transform: apply W_i along each axis i.
        # jnp.tensordot(W, X, axes=[[1],[i]]) contracts W with axis i of X,
        # placing the result at position 0; moveaxis restores it to position i.
        Ftilde = F
        for i, W in enumerate(self.left_eigvecs):
            Ftilde = jnp.tensordot(W, Ftilde, axes=[[1], [i]])
            Ftilde = jnp.moveaxis(Ftilde, 0, i)

        # Denominator: D[i0,i1,...] = sum_k s_k * Λ_k[0][i0] * Λ_k[1][i1] * ...
        # A scale can be complex with real eigenvectors and RHS, e.g. the 1j of
        # K⊗M + (1j M)⊗K, so it takes part in the dtype.
        dtype = jnp.result_type(Ftilde.dtype, jnp.float32, *self.scales)
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

        if self.real_operator and not jnp.iscomplexobj(rhs):
            U = U.real
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


def _make_wavenumber_solve(
    L_offsets: tuple[int, ...],
    U_offsets: tuple[int, ...],
    n_P: int,
    dtype: Any,
) -> Callable[..., Array]:
    """Build the per-wavenumber substitution, sequential or log-depth.

    The two produce the same answer and differ in shape of work. The sequential
    scan is ``2 n_P`` steps of a few multiply-adds each; the prefix form is
    ``O(log n_P)`` steps of ``r x r`` matmuls, more arithmetic for less depth.

    Which wins depends on the backend *and* on the band. A step costs a kernel
    launch on an accelerator, so depth is what is paid for there and the prefix
    form wins; on CPU a step is a loop iteration, the extra arithmetic is the
    whole cost, and the scan wins by a wide margin.

    The band matters because the prefix form carries an ``(n_P, r, r)``
    companion stack per wavenumber, so its traffic grows as ``r**2`` where the
    factors themselves grow as ``r``. That is cheap for a narrow band and
    ruinous for a wide one, and wide is not hypothetical: a Chebyshev stiffness
    matrix assembled Galerkin-style is nearly dense upper-triangular, putting
    ``q`` within a couple of ``n_P``. The same operator in a Petrov-Galerkin
    formulation -- `get_testspace("PG")` -- comes back banded at ``q = 4``.

    So the width is a property of the formulation rather than of the basis,
    which is why this reads the offsets it was actually handed instead of
    reasoning about which family they came from: the prefix form is taken only
    while ``r**2`` stays within a small multiple of the stored band.

    `JAXFUN_LU_SUBSTITUTION_ALGORITHM` (``scan`` | ``prefix`` | ``auto``)
    overrides the choice, which is how the two get compared on new hardware.
    """
    p = max((-o for o in L_offsets if o < 0), default=0)
    q = max((o for o in U_offsets if o > 0), default=0)
    if _use_prefix_substitution(p, q, len(L_offsets) + len(U_offsets), n_P):
        return _make_wavenumber_prefix_solve(L_offsets, U_offsets, n_P, dtype)
    return _make_wavenumber_vmap_solve(L_offsets, U_offsets, n_P, dtype)


def _make_wavenumber_prefix_solve(
    L_offsets: tuple[int, ...],
    U_offsets: tuple[int, ...],
    n_P: int,
    dtype: Any,
) -> Callable[..., Array]:
    """`_make_wavenumber_vmap_solve`'s answer, reached in log depth.

    Same signature, same result; the substitutions go through `_affine_prefix`
    instead of `jax.lax.scan`, so the sequential chain is ``O(log n_P)`` rather
    than ``2 n_P``.
    """
    p = max((-o for o in L_offsets if o < 0), default=0)
    q = max((o for o in U_offsets if o > 0), default=0)
    l_indices: list[int | None] = [
        L_offsets.index(-s) if -s in L_offsets else None for s in range(1, p + 1)
    ]
    U_main_idx: int = U_offsets.index(0)
    u_indices: list[int | None] = [
        U_offsets.index(s) if s in U_offsets else None for s in range(1, q + 1)
    ]
    rev = jnp.arange(n_P - 1, -1, -1)

    def _fwd_elim(L_data: Array, b: Array) -> Array:
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
        return _affine_prefix(l_mat.T, b[:, None], p)[:, 0]

    def _bwd_sub(U_data: Array, y: Array) -> Array:
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
        # Reversed, the back substitution is the same forward recurrence; the
        # division by the main diagonal folds into both sides to match the
        # `rhs - coef . window` form.
        diag_rev = diag_d[rev]
        y_rev = y[rev] / diag_rev
        u_rev = u_mat[:, rev].T / diag_rev[:, None]  # (n_P, q)
        return _affine_prefix(u_rev, y_rev[:, None], q)[:, 0][rev]

    def _solve_one(L_data: Array, U_data: Array, b: Array) -> Array:
        return _bwd_sub(U_data, _fwd_elim(L_data, b))

    return jax.jit(jax.vmap(_solve_one))


def _align_dia_data(
    dia_mat: DiaMatrix, target_offsets: tuple[int, ...], n_P: int
) -> Array:
    """Return ``dia_mat``'s diagonals stacked in ``target_offsets`` order.

    Offsets the matrix does not carry are filled with zeros, so that every
    matrix in a batch ends up with the same row layout and they can be stacked.
    """
    rows: list[Array] = []
    for off in target_offsets:
        if off in dia_mat.offsets:
            rows.append(dia_mat.data[list(dia_mat.offsets).index(off)])
        else:
            rows.append(jnp.zeros(n_P, dtype=dia_mat.data.dtype))
    return jnp.stack(rows)


def _nonzero_diagonals(x: Array) -> np.ndarray:
    """Which diagonals of ``x`` are nonzero somewhere in the *global* batch.

    ``x`` is ``(n_F, n_diags, n_P)`` and may be split over the wavenumber axis,
    so "somewhere" spans devices and processes. Reduced from each addressable
    block and then OR-ed across processes: the result is a handful of booleans,
    and every process has to come out with the same ones -- see
    `_prune_zero_diagonals` for what disagreeing would cost.

    Read from addressable blocks rather than the global array because a sharded
    array cannot be fetched whole from a process that holds only part of it.
    """
    nz = np.zeros(x.shape[1], dtype=bool)
    for shard in x.addressable_shards:
        nz |= np.asarray(jnp.any(jnp.abs(shard.data) > 0, axis=(0, 2)))
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        gathered = multihost_utils.process_allgather(jnp.asarray(nz))
        nz = np.asarray(np.any(np.asarray(gathered), axis=0), dtype=bool)
    return nz


def _prune_zero_diagonals(
    L: Array, U: Array, L_offsets: tuple[int, ...], U_offsets: tuple[int, ...]
) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
    """Drop the diagonals that are zero at *every* wavenumber.

    The LU fill-in pattern is structural -- the same for every k -- so which
    diagonals survive is a property of the operator, not of any one wavenumber.
    That only holds if the mask is computed over the whole batch. Deriving it
    from one process's slice lets two processes prune differently, and a
    differing set of offsets means a differing kernel shape and so a differing
    compiled program, which SPMD cannot survive.

    The mask has to be concrete: it selects how many diagonals the solve kernel
    carries, which is a shape. `_nonzero_diagonals` is what makes it both
    concrete and global.
    """
    L_nz = _nonzero_diagonals(L)
    U_nz = _nonzero_diagonals(U)
    return (
        L[:, L_nz, :],
        U[:, U_nz, :],
        tuple(o for o, nz in zip(L_offsets, L_nz) if nz),
        tuple(o for o, nz in zip(U_offsets, U_nz) if nz),
    )


def _parity_decoupling_enabled() -> bool:
    """Whether to split an even-offset operator into its two parity blocks.

    On by default. `JAXFUN_WAVENUMBER_PARITY=off` keeps the undecoupled path,
    which is what the two get compared against each other with.
    """
    return os.environ.get("JAXFUN_WAVENUMBER_PARITY", "on").lower() != "off"


def _parity_halves_offsets(poly_offsets: tuple[int, ...]) -> bool:
    """Whether every offset is even, so the operator decouples by index parity."""
    return any(o != 0 for o in poly_offsets) and all(o % 2 == 0 for o in poly_offsets)


def _parity_split_dia(
    B_data_batch: Array, poly_offsets: tuple[int, ...], n_P: int
) -> tuple[Array, tuple[int, ...], int]:
    """Split an even-offset DIA batch into its two decoupled parity blocks.

    An operator whose offsets are all even never connects an even index to an
    odd one, so reordering the polynomial axis as ``[0, 2, 4, ..., 1, 3, 5, ...]``
    makes it block diagonal: two independent systems of half the size, each with
    the offsets halved. The blocks join the batch, so the wavenumber axis doubles
    while the sequential axis halves -- depth traded for width, in the direction
    the hardware wants.

    On DIA data the reordering costs nothing. ``data[d][j]`` is the entry in
    *column* ``j``, and column ``2j + c`` is position ``j`` of parity ``c``, so
    the two blocks are the stride-2 slices ``data[d][0::2]`` and
    ``data[d][1::2]``. An odd ``n_P`` leaves the odd block one short; it is
    padded with an identity row, whose solution is zero and is dropped on the way
    out.

    Returns ``(B_split, offsets, m)`` with ``B_split`` shaped
    ``(n_batch * 2, n_diags, m)``, parity varying fastest so each device keeps
    the wavenumbers it already had.
    """
    m = (n_P + 1) // 2
    new_offsets = tuple(o // 2 for o in poly_offsets)

    cols = np.arange(m)[None, :] * 2 + np.arange(2)[:, None]  # (2, m)
    real = cols < n_P
    taken = jnp.transpose(
        B_data_batch[:, :, np.clip(cols, 0, n_P - 1)], (0, 2, 1, 3)
    )  # (n_batch, 2, n_diags, m)

    # A DIA slot is only a matrix entry while its implied row stays in range;
    # the rest are storage padding and have to read as zero, because the odd
    # block inherits slots whose original row lay past the end.
    rows = np.arange(m)[None, :] - np.asarray(new_offsets)[:, None]
    row_ok = (rows >= 0) & (rows < m)  # (n_diags, m)
    keep = real[:, None, :] & row_ok[None, :, :]
    is_main = (np.asarray(new_offsets) == 0)[None, :, None]
    pad_identity = (~real)[:, None, :] & is_main & row_ok[None, :, :]

    split = jnp.where(keep, taken, 0) + jnp.where(pad_identity, 1, 0).astype(
        B_data_batch.dtype
    )
    n_batch = B_data_batch.shape[0]
    return split.reshape(n_batch * 2, len(poly_offsets), m), new_offsets, m


def _banded_lu_batch(
    B_data_batch: Array, poly_offsets: tuple[int, ...]
) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
    """Factorize every wavenumber's banded system in one vmapped XLA call.

    Converts DIA format to band storage, runs all the LU factorisations at
    once, then extracts the *full* band range so that fill-in landing on a gap
    position is not dropped. Nothing is pruned here: which diagonals survive is
    a global property, so the caller prunes once it can see the whole batch.

    Every wavenumber is independent, so this is happy with a per-device block --
    which is how the sharded path uses it, one call per device on its own
    wavenumbers.

    Args:
        B_data_batch: ``(n_F, n_diags, n_P)`` per-wavenumber DIA data.
        poly_offsets: The diagonal offsets ``B_data_batch``'s rows correspond to.

    Returns:
        ``(L, U, L_offsets, U_offsets)`` over the *full* band range, L and U
        shaped ``(n_F, n_offsets, n_P)``.
    """
    from jaxfun.la.diamatrix import _lu_banded_no_pivot_kernel

    n_batch, _n_diags, n_P = B_data_batch.shape
    dtype = B_data_batch.dtype
    p = max((-o for o in poly_offsets if o < 0), default=0)
    q = max((o for o in poly_offsets if o > 0), default=0)
    center, bw = p, p + q + 1

    band_rows = jnp.array([center + off for off in poly_offsets])
    band = (
        jnp.zeros((n_batch, bw, n_P), dtype=dtype).at[:, band_rows, :].set(B_data_batch)
    )
    band_lu = jax.jit(jax.vmap(lambda b: _lu_banded_no_pivot_kernel(b, p, q, center)))(
        band
    )
    # Rows center - p, ..., center - 1 hold offsets -p, ..., -1: none when p = 0,
    # as for an upper-triangular matrix.
    L = band_lu[:, :center, :]
    U = band_lu[:, center:, :]
    return L, U, tuple(range(-p, 0)), tuple(range(0, q + 1))


def _batched_banded_lu(
    B_data_batch: Array, poly_offsets: tuple[int, ...]
) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
    """`_banded_lu_batch` followed by the structural prune."""
    return _prune_zero_diagonals(*_banded_lu_batch(B_data_batch, poly_offsets))


def _sharded_banded_lu(
    B_data_batch: Array, poly_offsets: tuple[int, ...]
) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
    """Factorize per device, then prune on a mask every device agrees on.

    `B_data_batch` is split over the wavenumber axis, so the `shard_map` body
    factors one device's block and nothing is ever assembled whole. The prune
    cannot go inside: it selects how many diagonals the solve kernel carries,
    which is a shape, and shapes have to match across devices. So the mapped
    body returns the full band range and `_prune_zero_diagonals` -- reading
    addressable blocks and OR-ing across processes -- decides afterwards.
    """
    p = max((-o for o in poly_offsets if o < 0), default=0)
    q = max((o for o in poly_offsets if o > 0), default=0)

    def _local(B_loc: Array) -> tuple[Array, Array]:
        L_loc, U_loc, _, _ = _banded_lu_batch(B_loc, poly_offsets)
        return L_loc, U_loc

    L, U = jax.jit(
        shard_map(
            _local,
            mesh=_sharding().spmd_mesh,
            in_specs=(P("k"),),
            out_specs=(P("k"), P("k")),
            check_vma=False,
        )
    )(B_data_batch)
    return _prune_zero_diagonals(L, U, tuple(range(-p, 0)), tuple(range(0, q + 1)))


def _looped_banded_lu(
    B_matrices: list,
) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
    """Factorize per wavenumber in a Python loop, one `DiaMatrix.lu_factor` each.

    `_batched_banded_lu` supersedes this for anything built by
    `tpmats_wavenumber_factor`; this is kept for callers that hand over
    ready-made per-wavenumber matrices.
    """
    n_P = B_matrices[0].shape[0]
    all_offsets = sorted({off for B in B_matrices for off in B.offsets})
    p = max((-o for o in all_offsets if o < 0), default=0)
    q = max((o for o in all_offsets if o > 0), default=0)
    # Full contiguous range, so LU fill-in within [-p, q] is not dropped.
    L_offsets, U_offsets = tuple(range(-p, 0)), tuple(range(0, q + 1))
    lus = [B.lu_factor() for B in B_matrices]
    L = jnp.stack([_align_dia_data(lu.L, L_offsets, n_P) for lu in lus])
    U = jnp.stack([_align_dia_data(lu.U, U_offsets, n_P) for lu in lus])
    return _prune_zero_diagonals(L, U, L_offsets, U_offsets)


def _check_shardable(poly_axis: int, shape: tuple[int, ...], n_total: int) -> None:
    """Reject a layout the wavenumber sharding cannot express.

    Both conditions are properties of the operator, so they are checked once, at
    construction. Left to fail later they surface as an `IndivisibleError` from
    deep inside array construction, or -- worse -- as a
    `TracerBoolConversionError` thousands of lines away, because a failed
    warm-up leaves the factorization cache cold and it is retried from inside
    the jitted step.
    """
    if poly_axis == 0:
        raise ValueError(
            "Multi-device solve requires axis 0 to be a Fourier axis "
            f"(poly_axis=0 not supported). Got shape={shape}, poly_axis={poly_axis}."
        )
    # Axis 0, not the product of every Fourier extent: the spectral sharding
    # partitions that one axis, and for more than one Fourier axis the two
    # differ -- a (3, 4, n_P) operator has 12 wavenumbers, which two devices
    # divide, across a leading axis of 3, which they do not.
    n_0 = shape[0]
    if n_0 % n_total:
        lower = (n_0 // n_total) * n_total
        raise ValueError(
            f"The leading Fourier axis carries {n_0} wavenumbers, which "
            f"{n_total} devices cannot split evenly: the spectral sharding "
            "partitions axis 0, so that count must be a multiple of the device "
            f"count. Nearest workable counts are {lower} and {lower + n_total}. "
            "A half spectrum (`TensorProduct(..., real=True)`) stores M // 2 + 1 "
            "coefficients, which is odd for every power-of-two M, and pads them "
            "up to a multiple of the device count for exactly this reason -- so "
            "reaching this means the padding was turned off with n_extra, or the "
            "space was built before the other processes' devices were visible. A "
            "full Fourier spectrum stores n_F = M and needs no padding."
        )


class TPMatricesWavenumberSolver(nnx.Pytree):
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
    pre-factorised once, outside jit -- which factorisation applies and which
    diagonals survive are decided by inspecting matrix *values*, and inside a
    traced step those are tracers.

    Args:
        poly_axis: Index of the polynomial axis in the full tensor.
        shape: Full solution shape ``(n_0, n_1, ...)``.
        B_matrices: Per-wavenumber :class:`~jaxfun.la.DiaMatrix` objects, length
            ``n_F`` (product of all Fourier-axis sizes). Superseded by
            ``B_data_batch``; see `_looped_banded_lu`.
        B_data_batch: ``(n_F, n_diags, n_P)`` per-wavenumber DIA data, the fast
            path.
        poly_offsets: Diagonal offsets ``B_data_batch``'s rows correspond to.
    """

    def __init__(
        self,
        poly_axis: int,
        shape: tuple[int, ...],
        B_matrices: list | None = None,
        B_data_batch: Array | None = None,
        poly_offsets: tuple[int, ...] | None = None,
    ) -> None:
        if isinstance(B_data_batch, jax.core.Tracer):
            raise SolverNotApplicable(
                "TPMatricesWavenumberSolver has to be built from concrete matrix "
                "values -- which factorisation applies and which diagonals survive "
                "are read off the values themselves -- but it was reached while "
                "tracing, so there is no second chance to look. Warm the operator "
                "before the jitted step"
            )

        self.poly_axis = poly_axis
        self.shape = shape

        n_total = len(jax.devices())
        if n_total > 1:
            _check_shardable(poly_axis, shape, n_total)

        parity = False
        if (
            B_data_batch is not None
            and poly_offsets is not None
            and _parity_decoupling_enabled()
            and _parity_halves_offsets(poly_offsets)
        ):
            # Before the factorisation, not after: halving the band makes the LU
            # itself cheaper, shrinks what the factors store, and both
            # substitutions inherit it without knowing it happened.
            n_P_full = B_data_batch.shape[-1]
            B_data_batch, poly_offsets, _m_parity = _parity_split_dia(
                B_data_batch, poly_offsets, n_P_full
            )
            parity = True

        if B_data_batch is not None and poly_offsets is not None:
            _, _n_diags, n_P_local = B_data_batch.shape
            # Factor each device's own wavenumbers. `B_data_batch` arrives on
            # `spectral_sharding`, so under `shard_map` the body sees one
            # device's block and never the whole batch -- which is the point:
            # the factors are the largest thing the operator holds, and this is
            # what keeps them O(n_F / n_devices) per device instead of O(n_F).
            # Pruning is deliberately left out of the mapped body; it decides a
            # shape, and a shape has to be agreed globally.
            if n_total > 1:
                L_all, U_all, L_offsets, U_offsets = _sharded_banded_lu(
                    B_data_batch, poly_offsets
                )
            else:
                L_all, U_all, L_offsets, U_offsets = _batched_banded_lu(
                    B_data_batch, poly_offsets
                )
        else:
            assert B_matrices is not None, (
                "Either B_data_batch+poly_offsets or B_matrices must be provided."
            )
            n_P_local = B_matrices[0].shape[0]
            L_all, U_all, L_offsets, U_offsets = _looped_banded_lu(B_matrices)

        self._parity: bool = parity
        self.L_offsets: tuple[int, ...] = L_offsets
        self.U_offsets: tuple[int, ...] = U_offsets
        self.L = L_all
        self.U = U_all
        self._vmap_solve = _make_wavenumber_solve(
            L_offsets, U_offsets, n_P_local, L_all.dtype
        )

        # Geometry of the transposed (Fourier..., polynomial) layout the solve
        # works in, and the permutation back out of it.
        _ndim = len(shape)
        _fourier_axes = [a for a in range(_ndim) if a != poly_axis]
        _fourier_shape = tuple(shape[a] for a in _fourier_axes)
        _n_F = int(np.prod(_fourier_shape)) if _fourier_shape else 1
        _n_P = shape[poly_axis]
        _axes_order = _fourier_axes + [poly_axis]
        _inv_perm = [0] * _ndim
        for _new_pos, _old_pos in enumerate(_axes_order):
            _inv_perm[_old_pos] = _new_pos
        _vmap_fn = self._vmap_solve

        _m = (_n_P + 1) // 2  # parity block length; unused when not decoupled

        def _to_parity(rhs_2d: Array, n_F: int) -> Array:
            """``(n_F, n_P)`` -> ``(n_F * 2, m)``, parity fastest."""
            r = jnp.pad(rhs_2d, ((0, 0), (0, 2 * _m - _n_P)))
            return r.reshape(n_F, _m, 2).transpose(0, 2, 1).reshape(n_F * 2, _m)

        def _from_parity(sol: Array, n_F: int) -> Array:
            """``(n_F * 2, m)`` -> ``(n_F, n_P)``, dropping the identity padding."""
            interleaved = sol.reshape(n_F, 2, _m).transpose(0, 2, 1)
            return interleaved.reshape(n_F, 2 * _m)[:, :_n_P]

        def _local_solve(
            L: Array, U: Array, rhs: Array, n_F: int, out_shape: tuple[int, ...]
        ) -> Array:
            """The whole solve, on whatever block of wavenumbers it is handed."""
            rhs_2d = jnp.transpose(rhs, _axes_order).reshape(n_F, _n_P)
            if parity:
                sol_2d = _from_parity(_vmap_fn(L, U, _to_parity(rhs_2d, n_F)), n_F)
            else:
                sol_2d = _vmap_fn(L, U, rhs_2d)
            sol_perm = sol_2d.reshape(out_shape + (_n_P,))
            return jnp.transpose(sol_perm, _inv_perm)

        if n_total == 1:

            @jax.jit
            def _solve_jit(L: Array, U: Array, rhs: Array) -> Array:
                return _local_solve(L, U, rhs, _n_F, _fourier_shape)
        else:
            # Every wavenumber's banded solve is independent and contracts only
            # the polynomial axis, which is never the split one. `shard_map`
            # states that rather than leaving it to the partitioner to notice:
            # each device gets its own wavenumbers of `rhs` *and* the factors
            # for exactly those, so the body has nothing to fetch and no
            # collective to arrange.
            _n_F_local = _n_F // n_total
            _local_fourier_shape = (_fourier_shape[0] // n_total,) + _fourier_shape[1:]

            @jax.jit
            def _solve_jit(L: Array, U: Array, rhs: Array) -> Array:
                return shard_map(
                    lambda L_loc, U_loc, rhs_loc: _local_solve(
                        L_loc, U_loc, rhs_loc, _n_F_local, _local_fourier_shape
                    ),
                    mesh=_sharding().spmd_mesh,
                    in_specs=(P("k"), P("k"), P("k")),
                    out_specs=P("k"),
                    check_vma=False,
                )(L, U, rhs)

        self._solve_jit = _solve_jit

    def solve(self, rhs: Array) -> Array:
        """Solve the wavenumber-loop system for RHS ``rhs``.

        Safe to call from inside an enclosing trace, which is how the time
        integrators reach it, and on any number of devices.

        Args:
            rhs: Right-hand side shaped ``self.shape``.

        Returns:
            Solution with the same shape and sharding as ``rhs``.
        """
        return self._solve_jit(self.L, self.U, rhs)


class TPMatricesDenseLUFactors(nnx.Pytree):
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

    @jax.jit
    def solve(self, rhs: Array) -> Array:
        """Solve the summed tensor-product system for RHS ``rhs``.

        Args:
            rhs: Right-hand side, flat ``(n,)`` or shaped ``(n0, n1, …)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        return self.lu.solve(rhs.ravel()).reshape(rhs.shape)


def tpmats_dense_lu_factor(
    A: TPMatrix | Sequence[TPMatrix],
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


def _sharding():
    """Return `jaxfun.sharding`, imported on use rather than at module scope.

    `jaxfun.sharding` imports `jaxfun.typing`, which imports `jaxfun.la`, which
    is this package -- so importing it from here at module scope closes a cycle.
    Every caller below is on a multi-device path, long after imports settle.
    """
    from jaxfun import sharding

    return sharding


def _solve_diagonal(diagonal: Array, rhs: Array) -> Array:
    """Solve a diagonal system while preserving the RHS shape."""
    if diagonal.shape == rhs.shape:
        return rhs / diagonal
    return (rhs.reshape((-1,)) / diagonal.reshape((-1,))).reshape(rhs.shape)


def _scale_tpmatrix(tp: TPMatrix, alpha: complex | Array) -> TPMatrix:
    return TPMatrix(
        list(tp.mats),
        scale=tp.coefficient * alpha,
        global_indices=tp.global_indices,
    )


# Largest condition number of a non-Hermitian pencil's (unit-column) eigenvector
# matrix that `tpmats_lu_factor` accepts. Close-to-normal pencils, such as Poisson
# in Chebyshev with PG/GR test spaces, stay in the tens up to N = 256.
# Far-from-normal ones grow exponentially with N: advection-dominated operators,
# or a first derivative with a single boundary condition, are past 1e3 by N = 12.
# A Hermitian-definite pencil is exempt: its eigenvectors are orthonormal in the
# inner product of the definite matrix, however their Euclidean norms spread.
_EIGVEC_COND_MAX = 1e3


def _merge_shared_factors(tpmats: Sequence[TPMatrix]) -> list[TPMatrix]:
    """Combine terms whose factors agree, up to a scalar, on all axes but one.

    ``A ⊗ B + C ⊗ cB`` becomes ``(A + cC) ⊗ B``, repeated until no two terms
    share all but one factor. Only terms of the same block, with equal
    `global_indices`, combine; the result keeps the blocks in order of first
    appearance. Merged terms are written over the distinct factors (see
    `TPMatrices.distinct_factors`); the others are returned as they are.
    """
    terms = list(tpmats)
    # Which terms merge is decided by their values, and traced values cannot be
    # read. Left unmerged, the terms are still the same operator.
    if any(
        isinstance(x, jax.core.Tracer)
        for tp in terms
        for x in (tp.coefficient, *(getattr(m, "data", None) for m in tp.mats))
    ):
        return terms
    blocks: dict[tuple[int, int], list[TPMatrix]] = {}
    for tp in terms:
        blocks.setdefault(tp.global_indices, []).append(tp)
    # The inputs are concrete, but under jit the sums of factors formed while
    # merging would be traced, and they are compared next. Evaluated eagerly,
    # they enter the trace as constants.
    with jax.ensure_compile_time_eval():
        return [tp for block in blocks.values() for tp in _merge_block(block)]


def _merge_block(tpmats: list[TPMatrix]) -> list[TPMatrix]:
    """`_merge_shared_factors` for the terms of one block."""
    form = TPMatrices(tpmats).distinct_factors()
    matrices = form.matrices
    # (factor indices, scale, original term or None once merged)
    terms: list[tuple[list[int], complex | Array, TPMatrix | None]] = [
        (list(idx), s, tp)
        for idx, s, tp in zip(form.indices, form.scales, tpmats, strict=True)
    ]
    while True:
        pair = next(
            (
                (a, b)
                for a, b in itertools.combinations(range(len(terms)), 2)
                if sum(x != y for x, y in zip(terms[a][0], terms[b][0])) <= 1
            ),
            None,
        )
        if pair is None:
            break
        a, b = pair
        (ia, sa, _), (ib, sb, _) = terms[a], terms[b]
        j = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), 0)
        r, c = _index_up_to_scalar(
            matrices[ia[j]].scale(sa) + matrices[ib[j]].scale(sb), matrices
        )
        terms[a] = ([*ia[:j], r, *ia[j + 1 :]], c, None)
        del terms[b]
    indices = tpmats[0].global_indices
    return [
        tp
        if tp is not None
        else TPMatrix([matrices[i] for i in idx], scale=s, global_indices=indices)
        for idx, s, tp in terms
    ]


def _diagonalize_or_schur(
    tpmats: list[TPMatrix],
) -> TPMatricesLUFactors | TPMatricesSchurSolver:
    """`tpmats_lu_factor`, or `tpmats_schur_factor` where it does not apply."""
    try:
        return tpmats_lu_factor(tpmats)
    except SolverNotApplicable as diag_err:
        try:
            return tpmats_schur_factor(tpmats)
        except SolverNotApplicable as schur_err:
            raise SolverNotApplicable(
                f"Diagonalization: {diag_err} Schur: {schur_err}"
            ) from None


def _is_hermitian(A: np.ndarray) -> bool:
    tol = 100 * np.finfo(A.dtype).eps * np.abs(A).max()
    return bool(np.all(np.abs(A - A.conj().T) <= tol))


def _diagonalize_axis(
    mats: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Simultaneously diagonalize the one or two factor matrices of an axis.

    Returns ``(V, W, evals)`` with ``W @ mats[j] @ V == diag(evals[j])``; one
    matrix of a pair gets the identity (``evals[j] == 1``).
    """
    # A single matrix A is the pair (A, I).
    pair = list(mats) if len(mats) == 2 else [mats[0], np.eye(len(mats[0]))]
    if _is_hermitian(pair[0]) and _is_hermitian(pair[1]):
        # eigh needs its second matrix positive definite, which only one of
        # the pair may be (e.g. a stiffness matrix entering with a minus sign).
        for a, b in ((0, 1), (1, 0)):
            try:
                lam, V = scipy_linalg.eigh(pair[a], pair[b])
            except scipy_linalg.LinAlgError:
                continue
            evals = [lam, lam]
            evals[b] = np.ones_like(lam)
            return V, V.conj().T, evals[: len(mats)]
    # Weighted inner products (Chebyshev, Jacobi) and Petrov-Galerkin test
    # spaces give non-symmetric matrices; eigh would silently use one triangle.
    for a, b in ((0, 1), (1, 0)):
        lam, V = scipy_linalg.eig(pair[a], pair[b])
        if np.all(np.isfinite(lam)):
            break
    else:
        raise SolverNotApplicable("Both factor matrices of an axis are singular.")
    cond = float(np.linalg.cond(V))
    if not cond <= _EIGVEC_COND_MAX:
        raise SolverNotApplicable(
            f"An eigenvector matrix has condition number {cond:.1e} (limit "
            f"{_EIGVEC_COND_MAX:.0e}), so diagonalization would lose that factor in "
            "accuracy; the factor matrices are far from normal."
        )
    # A complex pencil can have a real spectrum but complex eigenvectors.
    if np.all(lam.imag == 0) and np.all(V.imag == 0):
        lam, V = lam.real, V.real
    W = np.linalg.inv(pair[b] @ V)
    evals = [lam, lam]
    evals[b] = np.ones_like(lam)
    return V, W, evals[: len(mats)]


def tpmats_lu_factor(A: TPMatrix | Sequence[TPMatrix]) -> TPMatricesLUFactors:
    """Compute diagonalization-based LU factors for a sum of :class:`TPMatrix`.

    Simultaneously diagonalizes the factor matrices on each axis so that the
    full Kronecker-sum system reduces to element-wise division in the
    transformed space.

    **Algorithm** (2D, generalises to any number of dims):

    Given a list of TPMatrices representing :math:`\\sum_k s_k A_k \\otimes B_k`,
    find :math:`V` and :math:`W` such that :math:`W A V = \\Lambda_A` and
    :math:`W B V = I` (generalized eigenproblem :math:`A v = \\lambda B v`,
    with :math:`W = (B V)^{-1}`, or :math:`W = V^T` for a symmetric-definite
    pair). Then:

    .. math::

        \\tilde{F} = W F W^T, \\quad
        D_{ij} = \\textstyle\\sum_k s_k \\lambda_k^{(0)}{}_i \\lambda_k^{(1)}{}_j,
        \\quad U = V (\\tilde{F} / D) V^T.

    **Requirement**: each axis has at most 2 distinct factor matrices (e.g.
    stiffness K and mass M from the same 1D function space), counted up to a
    scalar, so that M, 2M and 3M count once. Terms that share their factors,
    up to a scalar, on all axes but one are merged first, so
    ``K⊗M + M⊗K + c M⊗M`` qualifies. The matrices need not be symmetric, so
    weighted bases (Chebyshev, Jacobi) and Petrov-Galerkin test spaces are
    supported; complex eigenpairs are handled in complex arithmetic. Axes that
    share the same unordered matrix pair reuse the same eigenvectors.

    The eigenvectors of a non-Hermitian pair must also be well conditioned.
    They are not for pencils far from normal, such as advection-dominated
    operators or a first derivative with a single boundary condition;
    :func:`tpmats_schur_factor` solves those in 2D.

    Args:
        A: Single :class:`TPMatrix` or list of :class:`TPMatrix` objects (as
            returned by :func:`~jaxfun.galerkin.inner.inner`).

    Returns:
        :class:`TPMatricesLUFactors` whose :meth:`~TPMatricesLUFactors.solve`
        method solves the system without re-factorising.

    Raises:
        SolverNotApplicable: if any axis has more than 2 distinct factor
            matrices after merging, both matrices of an axis are singular, or
            a non-Hermitian axis has ill-conditioned eigenvectors.
    """
    if isinstance(A, TPMatrix):
        A = [A]
    tpmats = list(A)
    ndim = tpmats[0].dims

    # Factors equal up to a scalar count once, so M, 2M and 3M on an axis are
    # one matrix; only the distinct ones are diagonalized.
    form = TPMatrices(tpmats).distinct_factors()
    # Merging only when needed leaves systems that already qualify untouched.
    if any(len({idx[i] for idx in form.indices}) > 2 for i in range(ndim)):
        tpmats = _merge_shared_factors(tpmats)
        form = TPMatrices(tpmats).distinct_factors()

    # --- per-axis (V, W, {matrix: eigenvalues}) ------------------------------
    # Axes that share the same unordered pair of matrices reuse the result. The
    # eigenvalues are kept per axis: a matrix paired differently on two axes
    # (M with K on one, with P on another) has different eigenvalues on each.
    pair_cache: dict[frozenset[int], tuple[Array, Array, dict[int, Array]]] = {}
    eigvecs: list[Array] = []
    left_eigvecs: list[Array] = []
    axis_eigenvalues: list[dict[int, Array]] = []
    for i in range(ndim):
        axis_mats = list(dict.fromkeys(idx[i] for idx in form.indices))
        pair_key = frozenset(axis_mats)
        if pair_key not in pair_cache:
            if len(axis_mats) > 2:
                raise SolverNotApplicable(
                    f"Axis {i} has {len(axis_mats)} distinct factor matrices; "
                    "simultaneous diagonalization requires ≤ 2 distinct matrices "
                    "per axis."
                )
            V, W, evals = _diagonalize_axis(
                [np.asarray(form.matrices[r].todense()) for r in axis_mats]
            )
            pair_cache[pair_key] = (
                jnp.asarray(V),
                jnp.asarray(W),
                {r: jnp.asarray(ev) for r, ev in zip(axis_mats, evals, strict=True)},
            )
        V, W, evals_map = pair_cache[pair_key]
        eigvecs.append(V)
        left_eigvecs.append(W)
        axis_eigenvalues.append(evals_map)

    per_term_eigenvalues = [
        [axis_eigenvalues[i][idx[i]] for i in range(ndim)] for idx in form.indices
    ]
    scales = form.scales
    real_operator = not any(
        jnp.issubdtype(m.dtype, jnp.complexfloating) for m in form.matrices
    ) and not any(np.iscomplexobj(s) for s in scales)
    shape: tuple[int, ...] = tuple(int(tpmats[0].mats[i].shape[0]) for i in range(ndim))
    return TPMatricesLUFactors(
        eigvecs=eigvecs,
        left_eigvecs=left_eigvecs,
        per_term_eigenvalues=per_term_eigenvalues,
        scales=scales,
        shape=shape,
        real_operator=real_operator,
    )


class TPMatricesSchurSolver(nnx.Pytree):
    """Generalized Schur (QZ) solver for a 2D sum of two tensor-product operators.

    Solves

    .. math::

        (X_0 \\otimes Y_0 + X_1 \\otimes Y_1)\\, x = f

    by the Bartels-Stewart method. A QZ decomposition :math:`Y_0 = Q S Z^H`,
    :math:`Y_1 = Q T Z^H` of the pair on one axis, the *Schur axis*, makes
    :math:`S` and :math:`T` upper triangular. The solution then follows one
    Schur index :math:`j` at a time, last to first, each a banded solve along
    the other axis with the matrix :math:`S_{jj} X_0 + T_{jj} X_1`, factorised
    with partial pivoting.

    Only unitary transforms enter, so unlike :class:`TPMatricesLUFactors` the
    solver stays accurate when the factor matrices are far from normal, such as
    advection-dominated operators or a first derivative with a single boundary
    condition. The price is a sequential loop over the Schur axis.

    Args:
        schur_axis: Axis carrying the QZ decomposition.
        shape: Solution shape ``(n_0, n_1)``.
        X: The two banded factors along the other axis, coefficients folded in.
        Q: Left unitary QZ factor, ``(n, n)`` for ``n = shape[schur_axis]``.
        Z: Right unitary QZ factor.
        S: Upper triangular QZ factor of :math:`Y_0`.
        T: Upper triangular QZ factor of :math:`Y_1`.
        real_operator: Whether the operator is real, so that a real right-hand
            side has a real solution.
    """

    def __init__(
        self,
        schur_axis: int,
        shape: tuple[int, int],
        X: Sequence[DiaMatrix],
        Q: Array,
        Z: Array,
        S: Array,
        T: Array,
        real_operator: bool,
    ) -> None:
        n_p = shape[1 - schur_axis]
        offsets = tuple(sorted(set(X[0].offsets) | set(X[1].offsets)))
        p = max((-o for o in offsets if o < 0), default=0)
        q = max((o for o in offsets if o > 0), default=0)
        X0, X1 = (_align_dia_data(x, offsets, n_p).astype(Q.dtype) for x in X)
        # S[j, j] X_0 + T[j, j] X_1 for every j, factorised in one batch with
        # partial pivoting within the band, as LAPACK's gbtrf. These matrices are
        # not diagonally dominant in general, and an unpivoted LU of them can
        # lose all accuracy without meeting a zero pivot. In the band layout of
        # `DiaMatrix.lu_factor`: L's multipliers below the center, U above,
        # widened by p for the fill from row swaps.
        B = jnp.diagonal(S)[:, None, None] * X0 + jnp.diagonal(T)[:, None, None] * X1
        center = p
        band = (
            jnp.zeros((B.shape[0], center + p + q + 1, n_p), dtype=B.dtype)
            .at[:, jnp.array([center + o for o in offsets]), :]
            .set(B)
        )
        band_lu, pivots = jax.vmap(lambda b: _lu_banded_kernel(b, p, p + q, center))(
            band
        )
        diagonal = band_lu[:, center]
        if not bool(jnp.all(jnp.isfinite(band_lu))) or bool(jnp.any(diagonal == 0)):
            raise SolverNotApplicable(
                "A banded matrix S[j, j] X_0 + T[j, j] X_1 is singular."
            )
        # Multipliers as (n, n_p, p), column t - 1 for the row t below.
        multipliers = jnp.flip(band_lu[:, :center], axis=1).transpose(0, 2, 1)
        _, U, _, U_offsets = _prune_zero_diagonals(
            band_lu[:, :0], band_lu[:, center:], (), tuple(range(p + q + 1))
        )
        self.schur_axis = schur_axis
        self.shape = shape
        self.X = nnx.List(X)
        self.Q = Q
        self.Z = Z
        self.S_upper = jnp.triu(S, 1)
        self.T_upper = jnp.triu(T, 1)
        self.multipliers = multipliers
        self.pivots = pivots
        self.U = U
        self.real_operator = real_operator
        self._backward = _make_wavenumber_solve((), U_offsets, n_p, U.dtype)

    @jax.jit
    def solve(self, rhs: Array) -> Array:
        """Solve the two-term tensor-product system for RHS ``rhs``.

        Args:
            rhs: Right-hand side, flat ``(n,)`` or shaped ``(n0, n1)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        q = self.schur_axis
        n = self.shape[q]
        X0, X1 = self.X
        # Rows of R are Schur indices. Once row j is solved, its contribution
        # to every row above it is subtracted, so R[j] is complete when reached.
        R = self.Q.conj().T @ jnp.moveaxis(rhs.reshape(self.shape), q, 0)

        def step(R: Array, xs: tuple) -> tuple[Array, Array]:
            j, m_j, piv_j, U_j, s_j, t_j = xs
            y = _pivoted_forward(m_j, piv_j, R[j][:, None])[:, 0]
            v = self._backward(U_j[None, :0], U_j[None], y[None])[0]
            R = R - jnp.outer(s_j, X0.matvec(v)) - jnp.outer(t_j, X1.matvec(v))
            return R, v

        xs = (
            jnp.arange(n),
            self.multipliers,
            self.pivots,
            self.U,
            self.S_upper.T,
            self.T_upper.T,
        )
        _, V = jax.lax.scan(step, R, xs, reverse=True)
        u = jnp.moveaxis(self.Z @ V, 0, q)
        if self.real_operator and not jnp.iscomplexobj(rhs):
            u = u.real
        return u.reshape(rhs.shape)


def tpmats_schur_factor(
    A: TPMatrix | Sequence[TPMatrix] | TPMatrices,
) -> TPMatricesSchurSolver:
    """Pre-factorize a 2D :class:`TPMatrices` system by QZ decomposition.

    The system must reduce to two terms, :math:`X_0 \\otimes Y_0 + X_1 \\otimes
    Y_1`, once terms that share their factors, up to a scalar, on all axes but
    one are merged. Any constant-coefficient operator that is a sum of pure
    x-derivatives and pure y-derivatives does, for example ``B⊗D + K⊗B + C⊗B``
    for :math:`u_y - \\nu u_{xx} - u_x`.

    The QZ decomposition goes on an axis with dense factor matrices if there is
    one, and otherwise on the shorter axis, whose length is that of the
    sequential loop in :meth:`TPMatricesSchurSolver.solve`. The other axis needs
    banded :class:`~jaxfun.la.DiaMatrix` factors.

    Args:
        A: :class:`TPMatrix`, list of :class:`TPMatrix` or :class:`TPMatrices`.

    Returns:
        :class:`TPMatricesSchurSolver` for repeated fast solves.

    Raises:
        SolverNotApplicable: if the system is not 2D, does not reduce to two
            terms, has no axis with banded factors, or one of the banded
            matrices is singular.
    """
    if isinstance(A, TPMatrices):
        tpmats = list(A.tpmats)
    elif isinstance(A, TPMatrix):
        tpmats = [A]
    else:
        tpmats = list(A)
    if tpmats[0].dims != 2:
        raise SolverNotApplicable(
            f"The Schur solver handles 2D systems; this one is {tpmats[0].dims}D."
        )
    if len(tpmats) > 2:
        tpmats = _merge_shared_factors(tpmats)
    if len(tpmats) != 2:
        raise SolverNotApplicable(
            f"The system reduces to {len(tpmats)} tensor-product terms; the Schur "
            "solver needs exactly 2."
        )
    shape = (int(tpmats[0].mats[0].shape[0]), int(tpmats[0].mats[1].shape[0]))
    banded = [all(isinstance(tp.mats[i], DiaMatrix) for tp in tpmats) for i in (0, 1)]
    if all(banded):
        schur_axis = 0 if shape[0] < shape[1] else 1
    elif any(banded):
        schur_axis = banded.index(False)
    else:
        raise SolverNotApplicable(
            "The Schur solver needs banded DiaMatrix factors on one axis."
        )
    X = [
        cast(DiaMatrix, tp.mats[1 - schur_axis]).scale(tp.coefficient) for tp in tpmats
    ]
    Y = [np.asarray(tp.mats[schur_axis].todense()) for tp in tpmats]
    S, T, Q, Z = scipy_linalg.qz(Y[0], Y[1], output="complex")
    real_operator = not any(np.iscomplexobj(np.asarray(x.data)) for x in X) and not any(
        np.iscomplexobj(y) for y in Y
    )
    return TPMatricesSchurSolver(
        schur_axis,
        shape,
        X,
        jnp.asarray(Q),
        jnp.asarray(Z),
        jnp.asarray(S),
        jnp.asarray(T),
        real_operator,
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
        SolverNotApplicable: If a factor is not a :class:`~jaxfun.la.DiaMatrix`,
            or the structure does not have exactly one non-diagonal
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
    if not all(isinstance(m, DiaMatrix) for tp in tpmats for m in tp.mats):
        raise SolverNotApplicable(
            "tpmats_wavenumber_factor requires DiaMatrix factors on every axis."
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

    # Working dtype, from *every* axis and the coefficients -- not the
    # polynomial axis alone. A Fourier diagonal is complex whenever the operator
    # carries an odd derivative along that direction (continuity is the case in
    # hand: its diagonal is purely imaginary), and taking the dtype from the
    # polynomial axis casts that diagonal to real, which silently zeroes every
    # k != 0 block and factorises a singular matrix. Nothing caught it because
    # every operator previously routed here is Helmholtz-like, whose diagonal is
    # real even when it is stored complex.
    _dtype = jnp.result_type(
        *[m.data.dtype for tp in tpmats for m in tp.mats],
        *[jnp.asarray(tp.coefficient).dtype for tp in tpmats],
    )
    # Stored complex is not the same as complex-valued. Demote when nothing
    # actually carries an imaginary part, so the usual real operators keep their
    # real factors instead of doubling their width for zeros.
    if jnp.issubdtype(_dtype, jnp.complexfloating) and not any(
        bool(jnp.any(jnp.abs(jnp.imag(jnp.asarray(part))) > 0))
        for tp in tpmats
        for part in ([m.data for m in tp.mats] + [jnp.asarray(tp.coefficient)])
    ):
        _dtype = jnp.finfo(_dtype).dtype

    # Build weight matrix W[i, k] = scale_i * prod_a(diag(F_i^(a))[k_a]).
    # The flat Fourier index k varies in C-order (last Fourier axis fastest),
    # matching the transposed layout used in TPMatricesWavenumberSolver.solve.
    W_list: list[Array] = []
    for tp in tpmats:
        w: Array = jnp.asarray(tp.coefficient, dtype=_dtype).reshape(1)
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
    n_total = len(jax.devices())
    if n_total > 1:
        # `W` is (n_terms, n_F) of Fourier diagonals -- kilobytes -- and
        # `P_data_stack` does not depend on the wavenumber at all, so both are
        # cheap to hold whole. `B_data_batch` is the first array here that
        # scales as n_F * n_diags * n_P, and so the first that must be born
        # split rather than split afterwards. The einsum is elementwise in the
        # wavenumber index, so placing `W`'s f axis places the output with it
        # and no wavenumber's data goes anywhere.
        _check_shardable(poly_axis, shape, n_total)
        _sh = _sharding()
        W = jax.device_put(W, NamedSharding(_sh.spmd_mesh, P(None, "k")))
        B_data_batch = jax.jit(
            lambda w, pd: jnp.einsum("tf,tdp->fdp", w, pd),
            out_shardings=_sh.spectral_sharding,
        )(W, P_data_stack)
    else:
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
            a0 = a0 * jnp.asarray(tpm.coefficient)
            for m in tpm.mats[1:]:
                a0 = jnp.kron(a0, m.todense())
            result = a0 if result is None else result + a0
        assert result is not None
        return Matrix(result)

    def _get_dia(mat: BaseMatrix) -> DiaMatrix:
        if isinstance(mat, Matrix):
            return DiaMatrix.from_dense(mat.todense(), tol=tol)
        assert isinstance(mat, DiaMatrix)
        return mat

    result: DiaMatrix | None = None
    for tpm in A:
        dmat: DiaMatrix = _get_dia(tpm.mats[0]) * jnp.asarray(tpm.coefficient)
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

    The :attr:`~TPMatrix.coefficient` is folded into the first factor matrix.

    Args:
        A: List of TPMatrix objects.
        tol: Near-zero elimination tolerance.

    Returns:
        List of tuples of per-axis scipy csc_array matrices.
    """
    from jaxfun.utils.common import eliminate_near_zeros

    result = []
    for a in A:
        scale = a.coefficient
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
