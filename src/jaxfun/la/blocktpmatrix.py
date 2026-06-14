from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from jaxfun.la.diamatrix import DiaMatrix
from jaxfun.la.matrix import Matrix
from jaxfun.la.matrixprotocol import BaseMatrix, IndexedArray, _CacheBox
from jaxfun.la.tpmatrix import TPMatrices, TPMatrix, tpmats_to_kron

if TYPE_CHECKING:
    from jaxfun.galerkin import (
        CartesianProductSpace,
        JAXFunction,
    )

type _SparseMatrixCache = _CacheBox[DiaMatrix]


class BlockTPMatrix(BaseMatrix):
    """Block matrix of TPMatrix objects.

    Args:
        tpmats: List of TPMatrix objects.
        test_space: descriptor for the (leaf) test space.
        trial_space: descriptor for the (leaf) trial space.

    Attributes:
        blocks: list of TPMatrix objects.
        test_space, trial_space: VectorTensorProductSpace descriptors.
    """

    def __init__(
        self,
        tpmats: list[TPMatrix],
        test_space: CartesianProductSpace,
        trial_space: CartesianProductSpace,
    ) -> None:
        self.tpmats = nnx.List(tpmats)
        self.test_space = test_space
        self.trial_space = trial_space
        self.shape = (self.test_space.dim, self.trial_space.dim)
        self.test_block_sizes: tuple[int, ...] = self.test_space.block_sizes
        self.trial_block_sizes: tuple[int, ...] = self.trial_space.block_sizes

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

    @property
    def dtype(self) -> jnp.dtype:
        dtype = jnp.float32
        for block_mat in self._combined_blocks.values():
            dtype = jnp.result_type(dtype, block_mat.dtype)
        return jnp.dtype(dtype)

    def scale(self, alpha: complex | Array) -> BlockTPMatrix:
        return BlockTPMatrix(
            [mat.scale(alpha) for mat in self.tpmats],
            self.test_space,
            self.trial_space,
        )

    def __add__(self, other: BlockTPMatrix) -> BlockTPMatrix:
        return BlockTPMatrix(
            list(self.tpmats) + list(other.tpmats),
            self.test_space,
            self.trial_space,
        )

    def __sub__(self, other: BlockTPMatrix) -> BlockTPMatrix:
        return BlockTPMatrix(
            list(self.tpmats) + [mat.scale(-1) for mat in other.tpmats],
            self.test_space,
            self.trial_space,
        )

    @jax.jit
    def _matmul_array(self, w: tuple[Array, ...]) -> tuple[Array, ...]:
        out = []
        flat_spaces = self.test_space.flatten()
        for s, block_mat in self._combined_blocks.items():
            bi, bj = map(int, s.split(","))
            out.append((block_mat @ w[bj].ravel()).reshape(flat_spaces[bi].num_dofs))
        return tuple(out)

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
        cached: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
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

    def __call__(self, u: BlockArray | JAXFunction) -> BlockArray:
        """Apply block matrix to coefficient array u.

        Uses the RCM-permuted :class:`~jaxfun.la.DiaMatrix` for a single
        global matvec when the sparse cache has been populated (e.g. after
        :meth:`solve` or an explicit :meth:`tosparse` call); otherwise falls
        back to the per-block path.
        """
        if not isinstance(u, BlockArray):
            w = BlockArray(self.test_space, tuple_array=u.get_array())
        else:
            w = u

        sparse_box: _SparseMatrixCache | None = getattr(self, "_sparse_cache", None)
        if sparse_box is not None:
            wf = w.flatten()
            sparse = sparse_box.value
            if getattr(sparse, "_rcm_cache", None) is not None:
                A_perm, perm, inv_perm = sparse.rcm()
                z = A_perm.matvec(wf[perm])[inv_perm]
            else:
                z = sparse.matvec(wf)
            return BlockArray(self.test_space, flat_array=z)

        return BlockArray(self.test_space, tuple_array=self._matmul_array(w.array))

    def __matmul__(self, u: BlockArray | JAXFunction) -> BlockArray:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def __len__(self) -> int:
        """Return number of rows (test space dimension)."""
        return self.shape[0]

    def solve(self, b: BlockArray) -> BlockArray:  # ty:ignore[invalid-method-override]
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
        x = b.flatten()
        if all_dia:
            A_perm, perm, inv_perm = self.tosparse().rcm()
            x_perm = A_perm.solve(x[perm])
            return BlockArray(b.cartspace, flat_array=x_perm[inv_perm])
        M = self.to_matrix()
        return BlockArray(b.cartspace, flat_array=M.solve(x))


class BlockArray(nnx.Pytree):
    """Block of Array objects.

    Args:
        cartspace: descriptor for the (leaf) space.
        indexed_arrays: List of IndexedArray objects.
        flat_array: 1D Array of length cartspace.dim.

    Attributes:
        data: nnx.List of Arrays representing each block.
        block_sizes: tuple of ints, dim of each block.
        num_dofs: tuple of tuple of ints, num_dof of each block.
        cartspace: CartesianProductSpace.
    """

    def __init__(
        self,
        cartspace: CartesianProductSpace,
        indexed_arrays: list[IndexedArray] | None = None,
        flat_array: Array | None = None,
        tuple_array: tuple[Array, ...] | None = None,
    ) -> None:
        self.data: nnx.List[Array] = nnx.List([])
        self.cartspace = nnx.static(cartspace)
        self.block_sizes: tuple[int, ...] = nnx.static(self.cartspace.block_sizes)
        self.num_dofs: tuple[tuple[int, ...], ...] = nnx.static(self.cartspace.num_dofs)
        for _ in range(cartspace.num_components):
            self.data.append(jnp.zeros(()))
        if indexed_arrays is not None:
            self.accumulate(indexed_arrays)
        if flat_array is not None:
            self.accumulate(self._list_from_flat_array(flat_array))
        if tuple_array is not None:
            self.accumulate(self._list_from_tuple_array(tuple_array))

    def _broadcast_data(self) -> tuple[Array, ...]:
        return tuple(
            d if d.shape != () else jnp.broadcast_to(d, self.num_dofs[i])
            for i, d in enumerate(self.data)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.cartspace.dim,)

    @property
    def array(self) -> tuple[Array, ...]:
        return self._broadcast_data()

    def flatten(self) -> Array:
        a = [d.ravel() for d in self._broadcast_data()]
        return jnp.concatenate(tuple(a))

    def _list_from_flat_array(self, x: Array) -> list[IndexedArray]:
        s0: int = 0
        d: list[IndexedArray] = []
        for i, s1 in enumerate(self.block_sizes):
            d.append(IndexedArray(i, x[slice(s0, s0 + s1)].reshape(self.num_dofs[i])))
            s0 += s1
        return d

    def _list_from_tuple_array(self, x: tuple[Array, ...]) -> list[IndexedArray]:
        return [IndexedArray(i, xi) for i, xi in enumerate(x)]

    def accumulate(self, b: IndexedArray | list[IndexedArray]) -> None:
        b_list: list[IndexedArray] = [b] if isinstance(b, IndexedArray) else b
        for bi in b_list:
            self.data[bi.i] += bi.data

    def __len__(self) -> int:
        return self.cartspace.num_components

    def __add__(self, b: BlockArray) -> BlockArray:
        return BlockArray(
            self.cartspace, tuple_array=tuple(x + y for x, y in zip(self, b))
        )

    def __sub__(self, b: BlockArray) -> BlockArray:
        return BlockArray(
            self.cartspace, tuple_array=tuple(x - y for x, y in zip(self, b))
        )

    def __neg__(self) -> BlockArray:
        return BlockArray(self.cartspace, tuple_array=tuple(-x for x in self))

    def __mul__(self, scalar: float | complex | Array) -> BlockArray:
        return BlockArray(self.cartspace, tuple_array=tuple(scalar * x for x in self))

    def __rmul__(self, scalar: float | complex | Array) -> BlockArray:
        return BlockArray(self.cartspace, tuple_array=tuple(scalar * x for x in self))

    def __iter__(self):
        return iter(self._broadcast_data())

    def __getitem__(self, i: int) -> Array:
        return self._broadcast_data()[i]
