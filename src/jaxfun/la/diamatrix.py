from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

Array = jax.Array


@nnx.dataclass
class DiaMatrix(nnx.Pytree):
    """Diagonal storage (DIA) sparse matrix, compatible with scipy.sparse.dia.

    Format:
        - ``offsets`` : 1D int array of diagonal offsets (0 = main, +k = k-th
          superdiagonal, -k = k-th subdiagonal).
        - ``data`` : float array of shape ``(n_diags, n_cols)`` with diagonals
          stored column-aligned. For diagonal with offset ``k``, the entry in
          column ``j`` is ``A[j - k, j]``.  Columns where the diagonal falls
          outside the matrix are stored as zero.

    Column-aligned storage rules:
        * ``k >= 0`` : valid columns are ``k .. min(m, n+k)-1``;
          ``data[i, k:k+length] = diag_values``.
        * ``k < 0``  : valid columns are ``0 .. min(m, n+k)-1``;
          ``data[i, 0:length]   = diag_values``.

    Example (tridiagonal 3x3):
    >>> import jax.numpy as jnp
    >>> a = jnp.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    >>> A = DiaMatrix.from_dense(a, offsets=(-1, 0, 1))
    >>> A.matvec(jnp.ones(3))
    Array([1., 0., 1.], dtype=float32)

    Example (non-square 4x6, tridiagonal-like):
    >>> A = diags(
    ...     [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
    ...     offsets=(-1, 0, 1),
    ...     shape=(4, 6),
    ... )
    >>> A.data.shape
    (3, 6)
    """

    data: Array
    offsets: tuple[int, ...]
    shape: tuple[int, int]

    @classmethod
    def from_dense(
        cls,
        a: Array,
        offsets: tuple[int, ...] | None = None,
    ) -> DiaMatrix:
        """Build a DiaMatrix from a dense 2-D array.

        Args:
            a: Dense matrix of shape ``(n, m)``.
            offsets: Which diagonals to store.  Defaults to all diagonals
                with at least one non-zero entry.
        """
        a = jnp.asarray(a)
        n, m = a.shape

        if offsets is None:
            offsets = tuple(k for k in range(-(n - 1), m) if jnp.any(jnp.diag(a, k)))

        offsets_arr = jnp.asarray(offsets, dtype=jnp.int32)
        j = jnp.arange(m)

        def _extract(k: Array) -> Array:
            row = j - k
            valid = (row >= 0) & (row < n)
            safe_row = jnp.where(valid, row, 0)
            return jnp.where(valid, a[safe_row, j], jnp.zeros((), dtype=a.dtype))

        sdata = jax.vmap(_extract)(offsets_arr)  # (n_diags, m)
        return cls(data=sdata, offsets=offsets, shape=(n, m))

    @jax.jit(static_argnums=(2,))
    def matvec(self, x: Array, axis: int = 0) -> Array:
        """Multiply ``A`` along ``axis`` of ``x``.

        Args:
            x:    Input array.  ``x.shape[axis]`` must equal ``m``.
            axis: The axis of ``x`` along which the matrix acts.  The output
                  has the same shape as ``x`` except ``shape[axis]`` becomes
                  ``n``.

        For a 1-D ``x`` the ``axis`` argument is ignored and the result is
        the plain matrix-vector product ``A @ x`` of shape ``(n,)``.

        Examples::

            A.matvec(x)  # x shape (m,)      → (n,)
            A.matvec(X, axis=0)  # X shape (m, k)    → (n, k)
            A.matvec(X, axis=1)  # X shape (k, m)    → (k, n)
            A.matvec(T, axis=2)  # T shape (a, b, m) → (a, b, n)
        """
        n, m = self.shape
        offsets_arr = jnp.array(self.offsets, dtype=jnp.int32)

        if x.ndim == 1:
            # v[p, j] = data[p, j] * x[j]  contributes to output row  i = j - k_p,
            # i.e.  y_p[i] = v[p, i + k_p].
            # Pad left and right by (n-1) zeros so that a window starting at
            # index  k + (n-1)  of size n always lands inside the padded array
            # for any valid offset k ∈ [-(n-1), m-1].
            v = self.data * x[None, :]  # (n_diags, m)
            v_pad = jnp.pad(v, ((0, 0), (n - 1, n - 1)))  # (n_diags, m+2n-2)

            def _extract_1d(args: tuple) -> Array:
                v_row, k = args
                return jax.lax.dynamic_slice(v_row, (k + n - 1,), (n,))

            contribs = jax.vmap(_extract_1d)((v_pad, offsets_arr))  # (n_diags, n)
            return contribs.sum(axis=0)

        axis = axis % x.ndim
        x_moved = jnp.moveaxis(x, axis, 0)
        rest_shape = x_moved.shape[1:]
        ncols_rest = x_moved.size // m
        x2d = x_moved.reshape(m, ncols_rest)  # (m, batch)

        # Same idea in 2-D: v[p, j, b] = data[p, j] * x2d[j, b].
        v = self.data[:, :, None] * x2d[None, :, :]  # (n_diags, m, batch)
        v_pad = jnp.pad(v, ((0, 0), (n - 1, n - 1), (0, 0)))  # (n_diags, m+2n-2, batch)

        def _extract_nd(args: tuple) -> Array:
            v_row, k = args  # (m+2n-2, batch), scalar
            return jax.lax.dynamic_slice(v_row, (k + n - 1, 0), (n, ncols_rest))

        contribs = jax.vmap(_extract_nd)((v_pad, offsets_arr))  # (n_diags, n, batch)
        y2d = contribs.sum(axis=0)  # (n, batch)
        y_moved = y2d.reshape((n,) + rest_shape)
        return jnp.moveaxis(y_moved, 0, axis)

    @jax.jit(static_argnums=(2,))
    def matmat(self, x: Array, axis: int = 0) -> Array:
        """Multiply ``A @ x`` where ``x.shape[axis] == m``.

        This is a convenience alias for :meth:`matvec` that makes it
        explicit that ``x`` is treated as a collection of column vectors.
        Equivalent to ``self.matvec(x, axis=axis)``.
        """
        return self.matvec(x, axis=axis)

    @jax.jit(static_argnums=(2,))
    def apply(self, x: Array, axis: int = 0) -> Array:
        """Apply ``A`` along ``axis`` of ``x`` (alias for :meth:`matvec`)."""
        return self.matvec(x, axis=axis)

    @jax.jit
    def to_dense(self) -> Array:
        """Return the equivalent dense ``(n, m)`` array."""
        # return self.matvec(jnp.eye(self.shape[1], dtype=self.data.dtype), axis=0) # expensive  # noqa: E501

        n, m = self.shape
        j = jnp.arange(m)

        def _add_diag(acc: Array, args: tuple) -> tuple:
            d, k = args
            row = j - k
            valid = (row >= 0) & (row < n)
            safe_row = jnp.where(valid, row, 0)
            vals = jnp.where(valid, d, jnp.zeros((), dtype=self.data.dtype))
            return acc.at[safe_row, j].add(vals), None

        A, _ = jax.lax.scan(
            _add_diag,
            jnp.zeros((n, m), dtype=self.data.dtype),
            (self.data, jnp.array(self.offsets, dtype=jnp.int32)),
        )
        return A

    def lu_factor(self, *, pivot: bool = False) -> LUFactors:
        """Compute a banded LU factorisation of this (square) matrix.

        Args:
            pivot: If ``False`` (default), no row interchanges are performed.
                This is faster and sufficient for diagonally-dominant or
                positive-definite matrices — the common case.  ``lu.perm`` will
                be ``None``.

                If ``True``, restricted partial pivoting is used: at each step
                the row with the largest absolute value in the pivot column,
                among rows ``k … k+p``, is swapped into position.  This handles
                zero diagonals and improves stability for general banded
                matrices.  Fill-in may extend U's bandwidth to ``p + q``.

        The elimination is compiled via ``jax.lax.scan`` and ``jax.jit``, so
        the first call traces and compiles the kernel (cached for the same
        ``(n, p, q, pivot)`` combination); subsequent calls on arrays of the
        same size are fast.  It is an *O(n·(p + q))* operation.

        Returns:
            :class:`LUFactors` containing ``L``, ``U``, ``perm``, and the
            original shape.

        Raises:
            ValueError: if the matrix is not square or the system is singular.
        """
        n, m = self.shape
        if n != m:
            raise ValueError(
                f"lu_factor requires a square matrix, got shape {self.shape}"
            )

        offsets = self.offsets
        p = max((-k for k in offsets if k < 0), default=0)  # lower bandwidth
        q = max((k for k in offsets if k > 0), default=0)  # upper bandwidth

        if not pivot:
            # ---- fast path: no row swaps --------------------------------
            # center = p, bw = p + q + 1  (no fill, no extra rows needed)
            center = p
            q_eff = q
            bw = p + q + 1
            band = jnp.zeros((bw, n), dtype=self.data.dtype)
            for d_idx, off in enumerate(offsets):
                s = center + off
                if 0 <= s < bw:
                    band = band.at[s, :].set(self.data[d_idx])

            band_lu = _lu_banded_no_pivot_kernel(band, p, q, center)

            if float(jnp.min(jnp.abs(band_lu[center]))) == 0.0:
                raise ValueError("Matrix is singular: zero pivot in LU factorisation.")

            l_offsets = tuple(
                off
                for off in range(-p, 1)
                if off == 0 or bool(jnp.any(band_lu[center + off] != 0))
            )
            l_data_rows: list[Array] = []
            for off in l_offsets:
                if off == 0:
                    l_data_rows.append(jnp.ones(n, dtype=self.data.dtype))
                else:
                    l_data_rows.append(band_lu[center + off].astype(self.data.dtype))
            L = DiaMatrix(data=jnp.stack(l_data_rows), offsets=l_offsets, shape=(n, n))

            u_offsets = tuple(range(0, q + 1))
            u_data_rows = [
                band_lu[center + u].astype(self.data.dtype) for u in range(q + 1)
            ]
            U = DiaMatrix(data=jnp.stack(u_data_rows), offsets=u_offsets, shape=(n, n))

            return LUFactors(L=L, U=U, shape=(n, n), perm=None)

        # ---- pivoting path ----------------------------------------------
        # Row pivoting within the band can introduce fill: U bandwidth ≤ p + q.
        q_eff = q + p

        # Build a compact band array of shape (bw, n) — O(n·bw) rather than O(n²).
        #
        # Convention (LAPACK-style with extra kl rows for fill):
        #   band[center + off, j] = A[j - off, j]  (column-aligned DIA)
        #   center = 2p
        #   bw     = 2p + q_eff + 1 = 3p + q + 1
        #
        # The extra 'p' rows above the original p sub-diagonals store multipliers
        # that were displaced when earlier rows got swapped down — allowing L fill
        # at offsets as deep as -2p without growing the array.
        center = 2 * p
        bw = center + q_eff + 1
        band = jnp.zeros((bw, n), dtype=self.data.dtype)
        for d_idx, off in enumerate(offsets):
            s = center + off  # band row index
            if 0 <= s < bw:
                band = band.at[s, :].set(self.data[d_idx])

        band_lu, perm_arr = _lu_banded_kernel(band, p, q_eff, center)

        # Singularity check: the main diagonal of U lives at band row `center`.
        if float(jnp.min(jnp.abs(band_lu[center]))) == 0.0:
            raise ValueError("Matrix is singular: zero pivot in LU factorisation.")

        # ---------- extract L -------------------------------------------
        # band_lu[center - t, :] is already the column-aligned DIA row for
        # offset -t (L[i, i-t] = band_lu[center-t, i-t]).  Skip all-zero rows.
        l_offsets = tuple(
            off
            for off in range(-2 * p, 1)
            if off == 0 or bool(jnp.any(band_lu[center + off] != 0))
        )
        l_data_rows: list[Array] = []
        for off in l_offsets:
            if off == 0:
                l_data_rows.append(jnp.ones(n, dtype=self.data.dtype))
            else:
                l_data_rows.append(band_lu[center + off].astype(self.data.dtype))
        L = DiaMatrix(
            data=jnp.stack(l_data_rows),
            offsets=l_offsets,
            shape=(n, n),
        )

        # ---------- extract U -------------------------------------------
        # band_lu[center + u, :] is the column-aligned DIA row for offset +u.
        u_offsets = tuple(range(0, q_eff + 1))
        u_data_rows = [
            band_lu[center + u].astype(self.data.dtype) for u in range(q_eff + 1)
        ]
        U = DiaMatrix(
            data=jnp.stack(u_data_rows),
            offsets=u_offsets,
            shape=(n, n),
        )

        # Use None when no rows were actually swapped (identity permutation).
        perm = (
            None
            if bool(jnp.all(perm_arr == jnp.arange(n, dtype=jnp.int32)))
            else perm_arr
        )
        return LUFactors(L=L, U=U, shape=(n, n), perm=perm)

    def solve(self, b: Array, axis: int = 0, *, pivot: bool = False) -> Array:
        """Solve ``A x = b`` directly via LU factorisation.

        Args:
            b:     Right-hand side array.  ``b.shape[axis]`` must equal ``n``.
            axis:  The axis of ``b`` along which the system is solved.  All
                   other axes are treated as independent batch dimensions.
                   The output has the same shape as ``b``.
            pivot: Passed to :meth:`lu_factor`.  Use ``True`` for matrices
                   with zero or near-zero diagonal entries.

        Returns:
            Solution array with the same shape as ``b``.

        Examples::

            A.solve(b)  # b shape (n,)       → (n,)
            A.solve(B, axis=0)  # B shape (n, k)     → (n, k)
            A.solve(B, axis=1)  # B shape (k, n)     → (k, n)
            A.solve(T, axis=2)  # T shape (a, b, n)  → (a, b, n)
        """
        return self.lu_factor(pivot=pivot).solve(b, axis=axis)

    @property
    def T(self) -> DiaMatrix:
        """Return the transpose ``A^T`` as a new :class:`DiaMatrix`.

        Shape becomes ``(m, n)``.  Each diagonal at offset ``k`` maps to
        offset ``-k`` in the transpose.  The data is re-aligned to the new
        column count ``n``.
        """
        n, m = self.shape
        new_shape = (m, n)

        def _transpose_row(d: Array, k: Array) -> Array:
            # Original: data[d] has length m, valid at columns [max(0,k), max(0,k)+length).  # noqa: E501
            # Transposed offset is -k; new data has length n.
            # new_data[i] = old_data[i + k]  for 0 <= i < n and 0 <= i+k < m.
            i = jnp.arange(n)
            j = i + k  # original column index
            valid = (j >= 0) & (j < m)
            safe_j = jnp.where(valid, j, 0)
            return jnp.where(valid, d[safe_j], jnp.zeros((), dtype=d.dtype))

        offsets_arr = jnp.asarray(self.offsets, dtype=jnp.int32)
        new_data = jax.vmap(_transpose_row)(self.data, offsets_arr)  # (n_diags, n)
        return DiaMatrix(
            data=new_data, offsets=tuple((-offsets_arr).tolist()), shape=new_shape
        )

    def diagonal(self, k: int = 0) -> Array:
        """Return the ``k``-th diagonal as a 1-D array.

        The length is ``min(n - max(0, -k), m - max(0, k))``.  Returns a
        zero array if ``k`` is not among the stored offsets.

        Args:
            k: Diagonal offset (0 = main, positive = upper, negative = lower).
        """
        n, m = self.shape
        length = min(n - max(0, -k), m - max(0, k))
        if length <= 0:
            return jnp.zeros(0, dtype=self.data.dtype)

        zero_diag = jnp.zeros(length, dtype=self.data.dtype)

        # Scan over stored diagonals; pick the one that matches k.
        def _pick(acc: Array, args: tuple) -> tuple:
            d, off = args
            col_start = jnp.maximum(0, off)
            vals = jax.lax.dynamic_slice(d, (col_start,), (length,))
            return jnp.where(off == k, vals, acc), None

        result, _ = jax.lax.scan(
            _pick, zero_diag, (self.data, jnp.array(self.offsets, dtype=jnp.int32))
        )
        return result

    def scale(self, alpha: float | Array) -> DiaMatrix:
        """Return ``alpha * A`` as a new :class:`DiaMatrix`."""
        return DiaMatrix(data=self.data * alpha, offsets=self.offsets, shape=self.shape)

    def __mul__(self, other: float | Array) -> DiaMatrix:
        return self.scale(other)

    def __rmul__(self, other: float | Array) -> DiaMatrix:
        return self.scale(other)

    def __neg__(self) -> DiaMatrix:
        return self.scale(-1)

    def _merge(self, other: DiaMatrix, sign: float) -> DiaMatrix:
        """Add ``self + sign * other`` in DIA format."""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        # Build the sum as a dense matrix and re-detect non-zero diagonals.
        # Addition is rarely on the hot path so the dense round-trip is fine.
        A = self.to_dense() + sign * other.to_dense()
        return DiaMatrix.from_dense(A)

    def __add__(self, other: DiaMatrix) -> DiaMatrix:
        return self._merge(other, +1.0)

    def __sub__(self, other: DiaMatrix) -> DiaMatrix:
        return self._merge(other, -1.0)

    @jax.jit
    def _matmul_compute(self, other: DiaMatrix) -> tuple[Array, Array, Array]:
        n, m = self.shape
        _, l = other.shape
        if m != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for matrix product: ({n}, {m}) @ {other.shape}"
            )

        j = jnp.arange(l)

        # Accumulate contributions per output diagonal offset.
        # Using a plain Python dict; number of diagonals is small.
        accum: dict[int, Array] = {}
        p_offs = self.offsets
        q_offs = other.offsets

        for p_idx, p_off in enumerate(p_offs):
            da = self.data[p_idx]  # shape (m,)
            for q_idx, q_off in enumerate(q_offs):
                db = other.data[q_idx]  # shape (l,)
                r_off = p_off + q_off

                # At output column j, A-column index is j - q_off
                a_col = j - q_off  # (l,)
                valid = (a_col >= 0) & (a_col < m)
                safe_a_col = jnp.where(valid, a_col, 0)

                contrib = jnp.where(
                    valid, da[safe_a_col] * db, jnp.zeros((), dtype=da.dtype)
                )

                if r_off in accum:
                    accum[r_off] = accum[r_off] + contrib
                else:
                    accum[r_off] = contrib

        if not accum:
            offsets_out = (0,)
            data_out = jnp.zeros((1, l), dtype=self.data.dtype)
            return data_out, jnp.array(offsets_out), jnp.array((n, l))

        offsets_list = sorted(accum.keys())
        return (
            jnp.stack([accum[r] for r in offsets_list]),
            jnp.array(offsets_list, dtype=jnp.int32),
            jnp.array((n, l), dtype=jnp.int32),
        )

    def __matmul__(self, other: Array | DiaMatrix) -> Array | DiaMatrix:
        """Support ``A @ x`` (vector/matrix) and ``A @ B`` (DiaMatrix).

        DiaMatrix x DiaMatrix is computed purely in DIA format without
        materialising either operand as a dense array.

        """
        if not isinstance(other, DiaMatrix):
            return self.apply(other)

        data_out, offsets_out, shape_out = self._matmul_compute(other)
        return DiaMatrix(
            data=data_out,
            offsets=tuple(offsets_out.tolist()),
            shape=tuple(shape_out.tolist()),
        )

    def __rmatmul__(self, other: Array) -> Array:
        """Support ``x @ A`` (row-vector or matrix on the left).

        ``x @ A == (A^T @ x^T)^T``.  Works for 1-D and 2-D ``other``.
        """
        if other.ndim == 1:
            return self.T.matvec(other)
        # other: (k, m)  →  A^T @ other^T  is (m_T==n, k)  →  transpose to (k, n)
        return self.T.matvec(other.T, axis=0).T

    def astype(self, dtype: jnp.dtype) -> DiaMatrix:
        """Return a copy with data cast to ``dtype``."""
        return DiaMatrix(
            data=self.data.astype(dtype), offsets=self.offsets, shape=self.shape
        )

    @property
    def nnz(self) -> int:
        """Number of explicitly stored (structurally non-zero) entries."""
        n, m = self.shape
        total = 0
        for k in self.offsets:
            total += min(n - max(0, -k), m - max(0, k))
        return total

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    def __repr__(self) -> str:
        n, m = self.shape
        nd = len(self.offsets)
        return (
            f"DiaMatrix(shape=({n}, {m}), dtype={self.dtype}, "
            f"n_diags={nd}, nnz={self.nnz}, offsets={self.offsets})"
        )


class LUFactors:
    """Result of :meth:`DiaMatrix.lu_factor`.

    Holds the unit-lower-triangular factor ``L`` and the upper-triangular
    factor ``U`` as :class:`DiaMatrix` objects so that ``A ≈ L @ U``.

    Example::

        A = diags([jnp.ones(4), -2 * jnp.ones(5), jnp.ones(4)], (-1, 0, 1))
        lu = A.lu_factor()
        x = lu.solve(jnp.ones(5))
    """

    def __init__(
        self,
        L: DiaMatrix,
        U: DiaMatrix,
        shape: tuple[int, int],
        perm: Array | None = None,
    ):
        self.L = L
        self.U = U
        self.shape = shape
        # perm[i] = original row index that was placed at row i after pivoting.
        # None means the identity permutation (no row swaps occurred).
        self.perm = perm

    def solve(self, b: Array, axis: int = 0) -> Array:
        """Solve ``A x = b`` using forward then backward substitution.

        The row permutation recorded during factorisation is applied to ``b``
        before forward elimination (i.e. solves ``L U x = P b``).

        Args:
            b:    Right-hand side array.  ``b.shape[axis]`` must equal ``n``.
            axis: The axis of ``b`` along which the system is solved.  All
                  other axes are treated as independent batch dimensions.
                  The output has the same shape as ``b``.

        Examples::

            lu.solve(b)  # b shape (n,)        → (n,)
            lu.solve(B, axis=0)  # B shape (n, k)      → (n, k)
            lu.solve(B, axis=1)  # B shape (k, n)      → (k, n)
            lu.solve(T, axis=2)  # T shape (a, b, n)   → (a, b, n)

        Returns:
            Solution array with the same shape as ``b``.
        """
        n = self.shape[0]

        if b.ndim == 1:
            if self.perm is not None:
                b = b[self.perm]
            y = _forward_elimination(self.L, b)
            return _backward_substitution(self.U, y)

        # N-D case: normalise the solved axis to position 0.
        axis = axis % b.ndim
        b_moved = jnp.moveaxis(b, axis, 0)  # (n, *rest)
        rest_shape = b_moved.shape[1:]
        batch = b_moved.size // n
        b2d = b_moved.reshape(n, batch)  # (n, batch)

        if self.perm is not None:
            b2d = b2d[self.perm, :]

        y2d = _forward_elimination(self.L, b2d)
        x2d = _backward_substitution(self.U, y2d)  # (n, batch)

        x_moved = x2d.reshape((n,) + rest_shape)
        return jnp.moveaxis(x_moved, 0, axis)

    def __repr__(self) -> str:
        n, m = self.shape
        return f"LUFactors(shape=({n}, {m}), L_offsets={self.L.offsets}, U_offsets={self.U.offsets})"  # noqa: E501


@jax.jit(static_argnums=(1, 2, 3))
def _lu_banded_no_pivot_kernel(band: Array, p: int, q: int, center: int) -> Array:
    """JIT-compiled banded LU *without* pivoting via ``jax.lax.scan``.

    Suitable for diagonally-dominant or positive-definite matrices.  Skips
    all pivot-search and row-swap operations, making each scan step cheaper
    than :func:`_lu_banded_kernel` by a factor of roughly ``bw``.

    Band convention: ``band[center + off, j] = A[j - off, j]``, ``center = p``,
    ``bw = p + q + 1``.

    Returns the in-place factored band (same shape as input).
    """
    n = band.shape[1]

    def elim_step(band: Array, k: Array) -> tuple[Array, None]:
        k = k.astype(jnp.int32)
        pivot = band[center, k]

        for s in range(1, p + 1):
            in_i = (k + s) < n
            factor = jnp.where(in_i, band[center - s, k] / pivot, 0.0)
            band = band.at[center - s, k].set(
                jnp.where(in_i, factor, band[center - s, k])
            )
            for u in range(1, q + 1):
                j = k + u
                in_j = j < n
                s_idx = center + u - s
                safe_j = jnp.where(in_j, j, 0)
                band = band.at[s_idx, safe_j].add(
                    jnp.where(in_i & in_j, -factor * band[center + u, safe_j], 0.0)
                )

        return band, None

    band_lu, _ = jax.lax.scan(elim_step, band, jnp.arange(n, dtype=jnp.int32))
    return band_lu


@jax.jit(static_argnums=(1, 2, 3))
def _lu_banded_kernel(
    band: Array, p: int, q_eff: int, center: int
) -> tuple[Array, Array]:
    """JIT-compiled banded LU with partial pivoting via ``jax.lax.scan``.

    Operates on the compact ``(bw, n)`` band array described in
    :meth:`DiaMatrix.lu_factor` rather than a dense ``(n, n)`` matrix,
    reducing both memory and compute from O(n²) to O(n·bw).

    Band convention:  ``band[center + off, j] = A[j - off, j]``
    (column-aligned DIA for diagonal offset ``off``).  ``center = 2p`` gives
    p extra rows above the original sub-diagonals for fill from row swaps.

    Args:
        band:    ``(bw, n)`` band array with ``bw = center + q_eff + 1``.
        p:       Lower bandwidth (static — determines loop unrolling).
        q_eff:   Effective upper bandwidth ``= q + p`` (static).
        center:  Band index of the main diagonal ``= 2p`` (static).

    Returns:
        ``(band_lu, perm)`` — the in-place factored band and the int32
        row-permutation array.
    """
    bw, n = band.shape

    def elim_step(
        carry: tuple[Array, Array], k: Array
    ) -> tuple[tuple[Array, Array], None]:
        band, perm = carry
        k = k.astype(jnp.int32)

        # --- partial pivot: find argmax |A[k..k+p, k]| ---
        # A[k+t, k] lives at band[center - t, k]  (t static, k traced)
        pivot_vals = jnp.stack([band[center - t, k] for t in range(p + 1)])
        row_inds = k + jnp.arange(p + 1, dtype=jnp.int32)
        masked = jnp.where(row_inds < n, jnp.abs(pivot_vals), 0.0)
        r_rel = jnp.argmax(masked).astype(jnp.int32)
        r = k + r_rel

        # --- swap rows k ↔ r in DIA band storage ---
        #
        # In column-aligned DIA format:  band[center + off, j] = A[j - off, j].
        # For a given original-matrix column j, the entry in row i is stored at
        # band row  s = center + j - i  and band column  j.
        # Swapping rows k and r means, for each column j:
        #   swap  band[center + j - k, j]  with  band[center + j - r, j].
        #
        # With dj = j - k (static loop variable) and dr = r_rel (traced):
        #   s_k  = center + dj          (static)
        #   s_r  = center + dj - r_rel  (dynamic — enumerate dr statically)
        for dj in range(-center, q_eff + 1):
            j = k + dj
            in_j = (j >= 0) & (j < n)
            s_k = center + dj  # static band-row for row k at column j
            if not (0 <= s_k < bw):
                continue
            safe_j = jnp.where(in_j, j, 0)
            for dr in range(1, p + 1):
                s_r = s_k - dr  # static band-row for row r=k+dr at column j
                if not (0 <= s_r < bw):
                    continue
                do_swap = (r_rel == dr) & in_j
                v_k = band[s_k, safe_j]
                v_r = band[s_r, safe_j]
                band = band.at[s_k, safe_j].set(
                    jnp.where(do_swap, v_r, band[s_k, safe_j])
                )
                band = band.at[s_r, safe_j].set(
                    jnp.where(do_swap, v_k, band[s_r, safe_j])
                )

        pk, pr = perm[k], perm[r]
        perm = perm.at[k].set(pr)
        perm = perm.at[r].set(pk)

        pivot = band[center, k]  # A[k, k] after swap

        # --- eliminate rows k+1 .. k+p (unrolled at trace time) ---
        for s in range(1, p + 1):
            in_i = (k + s) < n
            # Multiplier: A[k+s, k] = band[center - s, k]
            factor = jnp.where(in_i, band[center - s, k] / pivot, 0.0)
            band = band.at[center - s, k].set(
                jnp.where(in_i, factor, band[center - s, k])
            )
            for u in range(1, q_eff + 1):
                j = k + u
                in_j = j < n
                # A[k+s, k+u] = band[center + u - s, k+u]  (s_idx = center+u-s, static)
                s_idx = center + u - s
                safe_j = jnp.where(in_j, j, 0)
                band = band.at[s_idx, safe_j].add(
                    jnp.where(
                        in_i & in_j,
                        -factor * band[center + u, safe_j],
                        0.0,
                    )
                )

        return (band, perm), None

    perm0 = jnp.arange(n, dtype=jnp.int32)
    (band_lu, perm), _ = jax.lax.scan(
        elim_step, (band, perm0), jnp.arange(n, dtype=jnp.int32)
    )
    return band_lu, perm


@jax.jit
def _forward_elimination(L: DiaMatrix, b: Array) -> Array:
    """Solve ``L y = b`` where ``L`` is unit-lower-triangular (DiaMatrix).

    The scan carry is a sliding window of the last ``p`` computed values
    (bandwidth ``p``), so carry shape is ``(p, k)`` rather than ``(n, k)``.
    Each computed ``y[i]`` is emitted as the scan output and stacked into the
    full solution array automatically by ``jax.lax.scan``.

    Column-aligned DIA: for sub-diagonal at offset ``-s``,
    ``data[idx][i-s] = L[i, i-s]``.  Pre-shifting the data row by ``s``
    positions gives ``l_s[i] = L[i, i-s]`` directly.
    """
    n = L.shape[0]
    offsets = L.offsets

    scalar = b.ndim == 1
    b2d = b[:, None] if scalar else b  # (n, k)
    k = b2d.shape[1]

    # Build l_mat[j, i] = L[i, i-(j+1)]  for j = 0 .. p-1.
    l_rows: list[Array] = []
    for off in sorted(
        (o for o in offsets if o < 0), reverse=True
    ):  # -1, -2, ... (s=1,2,...)
        s = -off
        idx = offsets.index(off)
        d = L.data[idx]
        # d[i-s] = L[i, i-s]; shift right by s: zeros(s) ++ d[:n-s]
        l_rows.append(jnp.concatenate([jnp.zeros(s, dtype=d.dtype), d[: n - s]]))

    p = len(l_rows)
    if p == 0:
        return b  # L is identity

    l_mat = jnp.stack(l_rows)  # (p, n);  l_mat[j, i] = L[i, i-(j+1)]

    # carry: window[:, :] where window[j] = y[i-1-j] — the last p y-values.
    # xs at each step i: (b2d[i], l_mat[:, i])

    def step(window: Array, xs: tuple) -> tuple[Array, Array]:
        bi, l_i = xs  # (k,), (p,)
        yi = bi - jnp.einsum("p,pk->k", l_i, window)
        new_window = jnp.concatenate([yi[None, :], window[:-1, :]], axis=0)
        return new_window, yi  # carry (p, k);  output (k,)

    window0 = jnp.zeros((p, k), dtype=b.dtype)
    _, ys = jax.lax.scan(step, window0, (b2d, l_mat.T))  # ys: (n, k)
    return ys[:, 0] if scalar else ys


@jax.jit
def _backward_substitution(U: DiaMatrix, b: Array) -> Array:
    """Solve ``U x = b`` where ``U`` is upper-triangular (DiaMatrix).

    The scan carry is a sliding window of the ``q`` most-recently computed
    values (upper bandwidth ``q``); the scan runs in reverse order
    ``i = n-1, n-2, ..., 0``.

    Column-aligned DIA: for super-diagonal at offset ``+s``,
    ``data[idx][i+s] = U[i, i+s]``.  Shifting the data row left by ``s``
    gives ``u_s[i] = U[i, i+s]`` directly.
    """
    n = U.shape[0]
    offsets = U.offsets

    scalar = b.ndim == 1
    b2d = b[:, None] if scalar else b  # (n, k)
    k = b2d.shape[1]

    main_idx = offsets.index(0)
    diag_d = U.data[main_idx]  # diag_d[i] = U[i, i]

    # Build u_mat[j, i] = U[i, i+(j+1)]  for j = 0 .. q-1.
    u_rows: list[Array] = []
    for off in sorted(o for o in offsets if o > 0):  # 1, 2, ...
        s = off
        idx = offsets.index(off)
        d = U.data[idx]
        # d[i+s] = U[i, i+s]; shift left by s: d[s:] ++ zeros(s)
        u_rows.append(jnp.concatenate([d[s:n], jnp.zeros(s, dtype=d.dtype)]))

    q = len(u_rows)

    # Reverse all xs so the scan runs from i = n-1 down to 0.
    rev = jnp.arange(n - 1, -1, -1)
    b_rev = b2d[rev]  # (n, k)
    diag_rev = diag_d[rev]  # (n,)

    if q == 0:
        # Pure diagonal: no coupling, no window needed.
        xs_rev = b_rev / diag_rev[:, None]  # (n, k)
        x = xs_rev[rev]  # un-reverse
        return x[:, 0] if scalar else x

    u_mat = jnp.stack(u_rows)  # (q, n)
    u_mat_rev = u_mat[:, rev].T  # (n, q)

    # carry: window[j] = x[i+1+j] — next q solution values.
    def step(window: Array, xs: tuple) -> tuple[Array, Array]:
        bi, u_i, dii = xs  # (k,), (q,), ()
        xi = (bi - jnp.einsum("q,qk->k", u_i, window)) / dii
        new_window = jnp.concatenate([xi[None, :], window[:-1, :]], axis=0)
        return new_window, xi  # carry (q, k);  output (k,)

    window0 = jnp.zeros((q, k), dtype=b.dtype)
    _, xs_out = jax.lax.scan(step, window0, (b_rev, u_mat_rev, diag_rev))
    # xs_out[t] = x[n-1-t]; un-reverse to get x[0..n-1].
    x = xs_out[rev]  # (n, k)
    return x[:, 0] if scalar else x


def diags(
    diagonals: list[Array],
    offsets: tuple[int, ...],
    shape: tuple[int, int] | None = None,
    dtype: jnp.dtype | None = None,
) -> DiaMatrix:
    """Construct a :class:`DiaMatrix` from diagonal arrays.

    Mirrors ``scipy.sparse.diags``.

    Args:
        diagonals: Sequence of 1-D arrays, one per diagonal.
        offsets: Diagonal offset(s) matching ``diagonals``.
        shape: ``(n, m)`` shape of the output matrix.  Inferred from diagonal
            lengths when omitted (produces a square matrix).
        dtype: Optional dtype.  Defaults to the common dtype of ``diagonals``.

    Returns:
        :class:`DiaMatrix` instance.

    Example:
        >>> A = diags(
        ...     [jnp.ones(4), -2 * jnp.ones(5), jnp.ones(4)], offsets=(-1, 0, 1)
        ... )  # 5x5 tridiagonal
        >>> A = diags(
        ...     [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
        ...     offsets=(-1, 0, 1),
        ...     shape=(4, 6),
        ... )
    """
    diagonals = [jnp.asarray(d) for d in diagonals]

    if len(offsets) != len(diagonals):
        raise ValueError("offsets and diagonals must have the same length.")

    if shape is None:
        # Infer square shape: each diagonal of length L at offset k fits in
        # an (L + |k|) × (L + |k|) matrix.
        inferred = max(len(d) + abs(k) for d, k in zip(diagonals, offsets))
        n = m = inferred
    else:
        n, m = shape

    common_dtype = jnp.result_type(*diagonals) if dtype is None else dtype

    data_rows: list[Array] = []
    for d, k in zip(diagonals, offsets):
        d = d.astype(common_dtype)
        row = jnp.zeros(m, dtype=common_dtype)
        col_start = max(0, k)
        row_start = max(0, -k)
        length = min(len(d), m - col_start, n - row_start)
        if length > 0:
            # d[0] maps to A[row_start, col_start], i.e., column col_start
            row = row.at[col_start : col_start + length].set(d[:length])
        data_rows.append(row)

    data = jnp.stack(data_rows)  # (n_diags, m)
    return DiaMatrix(data=data, offsets=offsets, shape=(n, m))
