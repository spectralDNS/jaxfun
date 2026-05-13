from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from jaxfun.la.diamatrix import DiaMatrix
from jaxfun.la.matrix import Matrix
from jaxfun.la.matrixprotocol import _CacheBox
from jaxfun.la.tpmatrix import TPMatrices, TPMatrix, tpmats_to_kron

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction
    from jaxfun.galerkin.tensorproductspace import VectorTensorProductSpace


class BlockTPMatrix(nnx.Pytree):
    """Block matrix of TPMatrix objects.

    Args:
        tpmats: List of TPMatrix objects.
        test_space: VectorTensorProductSpace descriptor for the test space.
        trial_space: VectorTensorProductSpace descriptor for the trial space.

    Attributes:
        blocks: list of TPMatrix objects.
        test_space, trial_space: VectorTensorProductSpace descriptors.
    """

    def __init__(
        self,
        tpmats: list[TPMatrix],
        test_space: VectorTensorProductSpace,
        trial_space: VectorTensorProductSpace,
    ) -> None:
        self.tpmats = nnx.List(tpmats)
        self.test_space = test_space
        self.trial_space = trial_space
        self.shape = (self.test_space.dim, self.trial_space.dim)
        self.test_block_sizes: tuple[int, ...] = tuple(
            int(self.test_space[i].dim) for i in range(self.test_space.dims)
        )
        self.trial_block_sizes: tuple[int, ...] = tuple(
            int(self.trial_space[i].dim) for i in range(self.trial_space.dims)
        )
        # Pre-compute one combined matrix per (test_block, trial_block) pair so
        # that matvec, todense and tosparse each iterate over at most one entry
        # per block instead of one entry per contributing TPMatrix.
        _grouped: dict[str, TPMatrices] = {}
        for mat in tpmats:
            idx = f"{mat.global_indices[0]},{mat.global_indices[1]}"
            if idx not in _grouped:
                _grouped[idx] = TPMatrices([mat])
            else:
                _grouped[idx].tpmats.append(mat)
        self._combined_blocks = nnx.Dict(
            {idx: tpmats_to_kron(list(tpm.tpmats)) for idx, tpm in _grouped.items()}
        )

    @jax.jit
    def _matmul_array(self, w: Array) -> Array:
        out = jnp.zeros_like(w)
        for s, block_mat in self._combined_blocks.items():
            bi, bj = map(int, s.split(","))
            out = out.at[bi].add((block_mat @ w[bj].ravel()).reshape(out[bi].shape))
        return out

    def slice(self, indices: tuple[int, ...]) -> tuple[slice, ...]:  # ty:ignore[invalid-type-form]
        """Return slice object for block matrix indices."""
        N = self.test_block_sizes
        M = self.trial_block_sizes
        return (
            slice(sum(N[: indices[0]]), sum(N[: indices[0] + 1])),
            slice(sum(M[: indices[1]]), sum(M[: indices[1] + 1])),
        )

    def todense(self) -> Array:
        """Return dense array."""
        out = jnp.zeros(self.shape)
        for s, block_mat in self._combined_blocks.items():
            bi, bj = map(int, s.split(","))
            out = out.at[self.slice((bi, bj))].add(block_mat.todense())
        return out

    def to_matrix(self) -> Matrix:
        """Return dense matrix object."""
        return Matrix(self.todense())

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        """Assemble global :class:`~jaxfun.la.DiaMatrix` from all blocks.

        Works for both :class:`~jaxfun.la.DiaMatrix` and dense
        :class:`~jaxfun.la.Matrix` factor blocks.  Dense blocks are converted
        to :class:`~jaxfun.la.DiaMatrix` via
        :meth:`~jaxfun.la.DiaMatrix.from_dense` before assembly.

        The assembled matrix is cached on the instance so repeated calls are
        free.

        Args:
            tol: Near-zero elimination tolerance passed to
                :meth:`~jaxfun.la.DiaMatrix.from_dense` when any block is
                dense.  Entries smaller than ``ulp(tol)`` times the maximum
                entry are treated as zero.  Default 100.

        Returns:
            :class:`~jaxfun.la.DiaMatrix` of shape ``(total_rows, total_cols)``.
        """
        cached: _CacheBox[DiaMatrix] | None = getattr(self, "_sparse_cache", None)
        if cached is not None:
            return cached.value

        test_block_sizes = [int(s) for s in self.test_block_sizes]
        trial_block_sizes = [int(s) for s in self.trial_block_sizes]
        test_starts = [sum(test_block_sizes[:i]) for i in range(len(test_block_sizes))]
        trial_starts = [
            sum(trial_block_sizes[:i]) for i in range(len(trial_block_sizes))
        ]
        total_rows = sum(test_block_sizes)
        total_cols = sum(trial_block_sizes)

        # Build per-block DiaMatrix and collect all global diagonal offsets.
        block_dias: dict[tuple[int, int], DiaMatrix] = {}
        global_offsets: set[int] = set()
        for s, block_kron in self._combined_blocks.items():
            bi, bj = map(int, s.split(","))
            if isinstance(block_kron, Matrix):
                block_dia = DiaMatrix.from_dense(block_kron.todense(), tol=tol)
            else:
                block_dia = block_kron
            block_dias[(bi, bj)] = block_dia
            shift = trial_starts[bj] - test_starts[bi]
            for k in block_dia.offsets:
                global_offsets.add(int(k) + shift)

        sorted_offsets = tuple(sorted(global_offsets))
        offset_to_idx = {k: i for i, k in enumerate(sorted_offsets)}

        # Allocate global DIA data array and scatter each block's diagonals.
        dtype = jnp.result_type(*[d.data.dtype for d in block_dias.values()])
        global_data = jnp.zeros((len(sorted_offsets), total_cols), dtype=dtype)
        for (bi, bj), block_dia in block_dias.items():
            col_start = trial_starts[bj]
            shift = col_start - test_starts[bi]
            n_cols_block = trial_block_sizes[bj]
            for d_b, k_b in enumerate(block_dia.offsets):
                d_g = offset_to_idx[int(k_b) + shift]
                global_data = global_data.at[
                    d_g, col_start : col_start + n_cols_block
                ].add(block_dia.data[d_b])

        result = DiaMatrix(
            data=global_data, offsets=sorted_offsets, shape=(total_rows, total_cols)
        )
        object.__setattr__(self, "_sparse_cache", _CacheBox(result))
        return result

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply block matrix to coefficient array u.

        Uses the RCM-permuted :class:`~jaxfun.la.DiaMatrix` for a single
        global matvec when the sparse cache has been populated (e.g. after
        :meth:`solve` or an explicit :meth:`tosparse` call); otherwise falls
        back to the per-block path.
        """
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        sparse_box: _CacheBox[DiaMatrix] | None = getattr(self, "_sparse_cache", None)
        if sparse_box is not None:
            sparse = sparse_box.value
            if getattr(sparse, "_rcm_cache", None) is not None:
                A_perm, perm, inv_perm = sparse.rcm()
                return A_perm.matvec(w.ravel()[perm])[inv_perm].reshape(w.shape)
            return sparse.matvec(w.ravel()).reshape(w.shape)
        return self._matmul_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def __len__(self) -> int:
        """Return number of rows (test space dimension)."""
        return self.shape[0]

    def solve(self, b: Array) -> Array:
        """Solve ``M x = b``.

        When all factor matrices are :class:`~jaxfun.la.DiaMatrix` instances,
        assembles the global sparse matrix via :meth:`tosparse`, applies the
        reverse Cuthill-McKee permutation to minimise bandwidth, and solves
        with the banded LU.  The permuted matrix and permutation vectors are
        cached so repeated solves are cheap.

        Falls back to a dense factorisation when any factor is a
        :class:`~jaxfun.la.Matrix`.
        """
        all_dia = all(isinstance(m, DiaMatrix) for m in self._combined_blocks.values())
        if all_dia:
            A_perm, perm, inv_perm = self.tosparse().rcm()
            x_perm = A_perm.solve(b.ravel()[perm])
            return x_perm[inv_perm].reshape(b.shape)
        M = self.to_matrix()
        return M.solve(b.ravel()).reshape(b.shape)
