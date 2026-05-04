from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, overload

import jax
import jax.numpy as jnp
from flax import nnx

from jaxfun.la.matrixprotocol import _CacheBox

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction
    from jaxfun.la import Matrix, MatrixProtocol
    from jaxfun.la.pinned import PinnedSystem

Array = jax.Array


class DenseIndexingWarning(UserWarning):
    """Warning raised when DiaMatrix indexing must materialize dense output."""


def _normalize_index(idx, size: int):
    """Normalize an int/slice/array index into either scalar or 1-D JAX indices."""
    if isinstance(idx, int):
        idx = idx + size if idx < 0 else idx
        return min(max(idx, 0), size - 1)

    if isinstance(idx, slice):
        return jnp.arange(size, dtype=jnp.int32)[idx]

    # Array / traced scalar / advanced index.
    idx = jnp.asarray(idx, dtype=jnp.int32)
    idx = jnp.where(idx < 0, idx + size, idx)
    # jax clips out-of-bounds indices to the edge of the array.
    return jnp.clip(idx, 0, size - 1)


def _is_full_slice(key) -> bool:
    return (
        isinstance(key, slice)
        and key.start is None
        and key.stop is None
        and key.step is None
    )


def _warn_dense_indexing() -> None:
    warnings.warn(
        "DiaMatrix indexing materializes dense output for this key.",
        DenseIndexingWarning,
        stacklevel=3,
    )


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

    Attributes:
        data: Float array of shape ``(n_diags, n_cols)`` holding the stored
            diagonals in column-aligned format.
        offsets: Tuple of diagonal offsets corresponding to rows in ``data``.
        shape: ``(n, m)`` shape of the represented matrix.
        _lu_cache: Private ``dict[bool, LUFactors]`` populated on the first
            call to :meth:`lu_factor`.  Keyed by the ``pivot`` flag so the
            no-pivot and pivoting factorisations are cached independently.
            Stored via ``object.__setattr__`` so it is invisible to JAX's
            pytree machinery and does not affect JIT tracing or compilation.

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

    def __post_init__(self) -> None:
        n, m = self.shape
        total = 0
        for k in self.offsets:
            total += min(n - max(0, -k), m - max(0, k))
        self._size = total

    @classmethod
    def from_dense(
        cls,
        a: Array,
        offsets: tuple[int, ...] | None = None,
        tol: float = 1e-12,
    ) -> DiaMatrix:
        """Build a DiaMatrix from a dense 2-D array.

        Args:
            a: Dense matrix of shape ``(n, m)``.
            offsets: Which diagonals to store.  Defaults to all diagonals
                with at least one non-zero entry.
        """
        import numpy as np

        a = jnp.asarray(a)
        n, m = a.shape

        if offsets is None:
            # Use NumPy for the detection loop to avoid ~(n+m) separate JAX
            # device dispatches, which dominate runtime for large matrices.
            a_np = np.asarray(a)
            offsets = tuple(
                k for k in range(-(n - 1), m) if np.any(np.abs(a_np.diagonal(k)) > tol)
            )

        if not offsets:
            empty = jnp.zeros((0, m), dtype=a.dtype)
            return cls(data=empty, offsets=(), shape=(n, m))

        offsets_arr = jnp.asarray(offsets, dtype=int)
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
        offsets_arr = jnp.array(self.offsets, dtype=int)

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
    def rmatvec(self, x: Array, axis: int = -1) -> Array:
        """Multiply ``x @ A`` contracting ``axis`` of ``x`` against rows of ``A``.

        Symmetric dual of :meth:`matvec`: where ``matvec`` maps column indices
        of ``A`` to row indices, ``rmatvec`` maps row indices of ``A`` to
        column indices.

        Args:
            x: Input array.  ``x.shape[axis]`` must equal ``n``
                   (number of rows of ``A``).
            axis:  The axis of ``x`` to contract.  The output has the same
                   shape as ``x`` except ``shape[axis]`` becomes ``m``
                   (number of columns of ``A``).

        For a 1-D ``x`` the ``axis`` argument is ignored.

        Examples::

            A.rmatvec(x)  # x shape (n,)      → (m,)
            A.rmatvec(X, axis=-1)  # X shape (k, n)    → (k, m)
            A.rmatvec(X, axis=0)  # X shape (n, k)    → (m, k)

        Implementation note:
            ``y[j] = Σ_p data[p, j] * x[j - k_p]``.  Pad ``x`` by
            ``(m-1, m-1)`` zeros so that
            ``dynamic_slice(x_pad, (m-1) - k, m)`` always lands in-bounds
            for any offset ``k ∈ [-(n-1), m-1]``.
        """
        n, m = self.shape
        offsets_arr = jnp.array(self.offsets, dtype=int)

        if x.ndim == 1:
            # y[j] = sum_p data[p,j] * x[j - k_p]
            x_pad = jnp.pad(x, (m - 1, m - 1))  # length n + 2(m-1)

            def _extract_1d(args: tuple) -> Array:
                v_row, k = args
                sliced = jax.lax.dynamic_slice(x_pad, ((m - 1) - k,), (m,))
                return v_row * sliced

            contribs = jax.vmap(_extract_1d)((self.data, offsets_arr))  # (n_diags, m)
            return contribs.sum(axis=0)

        axis = axis % x.ndim
        x_moved = jnp.moveaxis(x, axis, -1)  # (..., n)
        rest_shape = x_moved.shape[:-1]
        batch = x_moved.size // n
        x2d = x_moved.reshape(batch, n)  # (batch, n)

        # Pad last axis: (batch, n) → (batch, n + 2(m-1))
        x_pad = jnp.pad(x2d, ((0, 0), (m - 1, m - 1)))

        def _extract_nd(args: tuple) -> Array:
            v_row, k = args  # v_row: (m,), k: scalar
            sliced = jax.lax.dynamic_slice(x_pad, (0, (m - 1) - k), (batch, m))
            return v_row[None, :] * sliced  # (batch, m)

        contribs = jax.vmap(_extract_nd)(
            (self.data, offsets_arr)
        )  # (n_diags, batch, m)
        result2d = contribs.sum(axis=0)  # (batch, m)
        result_moved = result2d.reshape(rest_shape + (m,))
        return jnp.moveaxis(result_moved, -1, axis)

    @jax.jit
    def todense(self) -> Array:
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
            (self.data, jnp.array(self.offsets, dtype=int)),
        )
        return A

    # Alias so DiaMatrix satisfies MatrixProtocol (which uses to_dense).
    to_dense = todense

    def to_Matrix(self) -> Matrix:
        """Convert this DiaMatrix to a dense Matrix."""
        from .matrix import Matrix

        return Matrix(self.todense())

    def lu_factor(self, *, pivot: bool = False) -> LUFactors:
        """Compute a banded LU factorisation of this (square) matrix.

        The result is cached on the matrix instance (keyed by ``pivot``) so
        that repeated calls — including from :meth:`solve` — pay the
        factorisation cost only once.  The cache is stored as a plain Python
        attribute (``_lu_cache``) so it is invisible to JAX's pytree machinery.

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
        # --- lazy cache -------------------------------------------------
        _box: _CacheBox[dict[bool | str, LUFactors]] | None = getattr(
            self, "_lu_cache", None
        )
        if _box is None:
            _box = _CacheBox({})
            object.__setattr__(self, "_lu_cache", _box)
        cache: dict[bool | str, LUFactors] = _box.value
        if pivot in cache:
            return cache[pivot]

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
            valid = [
                (d_idx, center + off)
                for d_idx, off in enumerate(offsets)
                if 0 <= center + off < bw
            ]
            if valid:
                src_idx, dst_rows = zip(*valid)
                band = band.at[jnp.array(dst_rows), :].set(
                    self.data[jnp.array(src_idx)]
                )

            band_lu = _lu_banded_no_pivot_kernel(band, p, q, center)

            if float(jnp.min(jnp.abs(band_lu[center]))) == 0.0:
                raise ValueError(
                    "Matrix is singular: zero pivot in LU factorisation. "
                    "Consider pinning (:meth:`DiaMatrix.pin`) additional DOFs or "
                    "enabling pivoting for this matrix."
                )

            _lu_tol = (
                float(jnp.finfo(self.data.dtype).eps)
                * float(jnp.max(jnp.abs(band_lu)))
                * n
            )

            l_offsets = tuple(
                off
                for off in range(-p, 1)
                if off == 0 or float(jnp.max(jnp.abs(band_lu[center + off]))) > _lu_tol
            )
            l_data_rows: list[Array] = []
            for off in l_offsets:
                if off == 0:
                    l_data_rows.append(jnp.ones(n, dtype=self.data.dtype))
                else:
                    l_data_rows.append(band_lu[center + off].astype(self.data.dtype))
            L = DiaMatrix(data=jnp.stack(l_data_rows), offsets=l_offsets, shape=(n, n))

            u_offsets = tuple(
                off
                for off in range(0, q + 1)
                if off == 0 or float(jnp.max(jnp.abs(band_lu[center + off]))) > _lu_tol
            )
            u_data_rows = [
                band_lu[center + u].astype(self.data.dtype) for u in u_offsets
            ]
            U = DiaMatrix(data=jnp.stack(u_data_rows), offsets=u_offsets, shape=(n, n))

            result = LUFactors(L=L, U=U, shape=(n, n), perm=None)
            cache[pivot] = result
            return result

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
        valid = [
            (d_idx, center + off)
            for d_idx, off in enumerate(offsets)
            if 0 <= center + off < bw
        ]
        if valid:
            src_idx, dst_rows = zip(*valid)
            band = band.at[jnp.array(dst_rows), :].set(self.data[jnp.array(src_idx)])

        band_lu, perm_arr = _lu_banded_kernel(band, p, q_eff, center)

        # Singularity check: the main diagonal of U lives at band row `center`.
        if float(jnp.min(jnp.abs(band_lu[center]))) == 0.0:
            raise ValueError(
                "Matrix is singular: zero pivot in LU factorisation. "
                "Consider pinning (:meth:`DiaMatrix.pin`) additional DOFs "
                "for this matrix."
            )

        # ---------- extract L -------------------------------------------
        # band_lu[center - t, :] is already the column-aligned DIA row for
        # offset -t (L[i, i-t] = band_lu[center-t, i-t]).  Skip near-zero rows
        # using a relative tolerance (floating-point residuals from elimination
        # can leave structurally-zero diagonals with tiny non-zero entries).
        _lu_tol = (
            float(jnp.finfo(self.data.dtype).eps) * float(jnp.max(jnp.abs(band_lu))) * n
        )

        l_offsets = tuple(
            off
            for off in range(-2 * p, 1)
            if off == 0 or float(jnp.max(jnp.abs(band_lu[center + off]))) > _lu_tol
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
        u_offsets = tuple(
            off
            for off in range(0, q_eff + 1)
            if off == 0 or float(jnp.max(jnp.abs(band_lu[center + off]))) > _lu_tol
        )
        u_data_rows = [band_lu[center + u].astype(self.data.dtype) for u in u_offsets]
        U = DiaMatrix(
            data=jnp.stack(u_data_rows),
            offsets=u_offsets,
            shape=(n, n),
        )

        # Use None when no rows were actually swapped (identity permutation).
        perm = None if bool(jnp.all(perm_arr == jnp.arange(n, dtype=int))) else perm_arr
        result = LUFactors(L=L, U=U, shape=(n, n), perm=perm)
        cache[pivot] = result
        return result

    def lu_solve(
        self,
        b: Array,
        axis: int = 0,
        *,
        pivot: bool = False,
        dense_threshold: int = 100,
    ) -> Array:
        """Solve ``A x = b``.

        Uses the banded LU factorisation (:meth:`lu_factor`) when the
        bandwidth is narrow enough that the JIT-compiled kernel compiles
        quickly.  Falls back to ``jnp.linalg.solve`` on the dense matrix when
        the product ``p * (q + 1)`` (number of unrolled loop iterations in the
        LU kernel) exceeds ``dense_threshold``, avoiding multi-minute XLA
        compile times for wide-banded matrices.

        Args:
            b: Right-hand side array.
            axis: Axis of ``b`` along which the system is solved.
            pivot: Passed to :meth:`lu_factor` (banded path only).
            dense_threshold: Maximum bandwidth ``p * (q + 1)`` before
                switching to a dense solver. Default 100.

        Returns:
            Solution array with the same shape as ``b``.
        """
        offsets = self.offsets
        p = max((-k for k in offsets if k < 0), default=0)
        q = max((k for k in offsets if k > 0), default=0)

        # If the banded LU is already cached (e.g. from a prior lu_factor()
        # call), use it regardless of bandwidth — the expensive compilation
        # already happened.
        _box: _CacheBox[dict[bool | str, LUFactors]] | None = getattr(
            self, "_lu_cache", None
        )
        if _box is not None and pivot in _box.value:
            return _box.value[pivot].solve(b, axis=axis)

        if p * (q + 1) > dense_threshold:
            # Dense path: XLA compiles jnp.linalg.solve in milliseconds.
            # Cache the dense LUFactors so repeated calls pay only the
            # forward/back substitution cost (same as the banded path).
            from jaxfun.la.matrix import Matrix  # local import avoids circular

            _box: _CacheBox[dict[bool | str, LUFactors]] | None = getattr(
                self, "_lu_cache", None
            )
            if _box is None:
                _box = _CacheBox({})
                object.__setattr__(self, "_lu_cache", _box)
            cache: dict[bool | str, LUFactors] = _box.value
            _dense_key = "_dense"
            if _dense_key not in cache:
                cache[_dense_key] = Matrix(self.todense()).lu_factor()  # type: ignore[index]
            return cache[_dense_key].solve(b, axis=axis)

        return self.lu_factor(pivot=pivot).solve(b, axis=axis)

    solve = lu_solve

    @property
    def T(self) -> DiaMatrix:
        """Return the transpose ``A^T`` as a new :class:`DiaMatrix`.

        Shape becomes ``(m, n)``.  Each diagonal at offset ``k`` maps to
        offset ``-k`` in the transpose.  The data is re-aligned to the new
        column count ``n``.
        """
        n, m = self.shape
        new_offsets = tuple(-k for k in self.offsets)
        new_shape = (m, n)

        def _transpose_row(d: Array, k: Array) -> Array:
            i = jnp.arange(n)
            j = i + k  # original column index
            valid = (j >= 0) & (j < m)
            safe_j = jnp.where(valid, j, 0)
            return jnp.where(valid, d[safe_j], jnp.zeros((), dtype=d.dtype))

        offsets_arr = jnp.asarray(self.offsets, dtype=int)
        new_data = jax.vmap(_transpose_row)(self.data, offsets_arr)
        return DiaMatrix(data=new_data, offsets=new_offsets, shape=new_shape)

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
            _pick, zero_diag, (self.data, jnp.array(self.offsets, dtype=int))
        )
        return result

    def get_row(self, i: int | Array) -> Array:
        """Return row ``i`` of the matrix as a dense 1-D array of length ``m``.

        In DIA format each stored diagonal at offset ``k`` contributes to row
        ``i`` at column ``j = i + k``, provided ``0 <= j < m``.  The value is
        ``data[p, i + k]`` (column-aligned storage).

        Args:
            i: Row index (0-based).  May be a traced JAX scalar so the method
               is usable inside :func:`jax.jit` / :func:`jax.vmap`.

        Returns:
            Dense array of shape ``(m,)`` with the row values; columns not
            covered by any stored diagonal are zero.

        Examples:

            >>> import jax.numpy as jnp
            >>> from jaxfun.la import diags
            >>> A = diags([jnp.ones(4), -2 * jnp.ones(5), jnp.ones(4)], (-1, 0, 1))
            >>> bool(jnp.allclose(A.get_row(0), jnp.array([-2.0, 1.0, 0.0, 0.0, 0.0])))
            True
            >>> bool(jnp.allclose(A.get_row(2), jnp.array([0.0, 1.0, -2.0, 1.0, 0.0])))
            True
        """
        _, m = self.shape

        i = jnp.asarray(i, dtype=jnp.int32)
        offsets = jnp.asarray(self.offsets, dtype=jnp.int32)

        cols = i + offsets
        in_bounds = (cols >= 0) & (cols < m)
        safe_cols = jnp.where(in_bounds, cols, 0)

        diag_ids = jnp.arange(self.data.shape[0])
        vals = self.data[diag_ids, safe_cols]
        vals = jnp.where(in_bounds, vals, jnp.zeros((), dtype=self.data.dtype))

        row = jnp.zeros((m,), dtype=self.data.dtype)
        return row.at[safe_cols].add(vals)

    def get_column(self, j: int | Array) -> Array:
        """Return column ``j`` of the matrix as a dense 1-D array of length ``n``.

        In DIA format each stored diagonal at offset ``k`` contributes to
        column ``j`` at row ``i = j - k``, provided ``0 <= i < n``.  The
        value is ``data[p, j]`` (column-aligned storage, so ``data[p, j] =
        A[j - k, j]``).

        Args:
            j: Column index.  May be a traced JAX scalar so the method is usable inside
            :func:`jax.jit` / :func:`jax.vmap`.

        Returns:
            Dense array of shape ``(n,)`` with the column values; rows not
            covered by any stored diagonal are zero.

        Examples:

            >>> import jax.numpy as jnp
            >>> from jaxfun.la import diags
            >>> A = diags([jnp.ones(4), -2 * jnp.ones(5), jnp.ones(4)], (-1, 0, 1))
            >>> bool(
            ...     jnp.allclose(A.get_column(0), jnp.array([-2.0, 1.0, 0.0, 0.0, 0.0]))
            ... )
            True
            >>> bool(
            ...     jnp.allclose(A.get_column(2), jnp.array([0.0, 1.0, -2.0, 1.0, 0.0]))
            ... )
            True
        """
        n, m = self.shape

        j = jnp.asarray(j, dtype=jnp.int32)
        offsets = jnp.asarray(self.offsets, dtype=jnp.int32)

        rows = j - offsets
        in_bounds = (rows >= 0) & (rows < n) & (j >= 0) & (j < m)
        safe_rows = jnp.where(in_bounds, rows, 0)
        safe_j = jnp.where((j >= 0) & (j < m), j, 0)

        vals = self.data[:, safe_j]
        vals = jnp.where(in_bounds, vals, jnp.zeros((), dtype=self.data.dtype))

        col = jnp.zeros((n,), dtype=self.data.dtype)
        return col.at[safe_rows].add(vals)

    def scale(self, alpha: float | Array) -> DiaMatrix:
        """Return ``alpha * A`` as a new :class:`DiaMatrix`."""
        return DiaMatrix(data=self.data * alpha, offsets=self.offsets, shape=self.shape)

    @property
    def size(self) -> int:
        """Return the total number of entries in the matrix (including zeros)."""
        return self._size

    def __mul__(self, other: float | Array) -> DiaMatrix:
        return self.scale(other)

    def __rmul__(self, other: float | Array) -> DiaMatrix:
        return self.scale(other)

    def __neg__(self) -> DiaMatrix:
        return self.scale(-1)

    def __len__(self) -> int:
        return min(self.shape)

    def __getitem__(self, key: int | slice | tuple | Array) -> Array:
        if not isinstance(key, tuple):
            return self._get_single_axis(key)

        key = self._expand_tuple_key(key)
        axis_keys = [item for item in key if item is not None]
        if len(axis_keys) != 2:
            raise TypeError("DiaMatrix indexing expects A[i, j].")

        row_key, col_key = axis_keys
        result, axis_counts = self._get_tuple_axes(row_key, col_key)
        return self._apply_newaxes(result, key, axis_counts)

    def _expand_tuple_key(self, key: tuple) -> tuple:
        items = list(key)
        ellipsis_count = sum(item is Ellipsis for item in items)
        if ellipsis_count > 1:
            raise TypeError("DiaMatrix indexing expects at most one ellipsis.")

        if ellipsis_count:
            ellipsis_pos = next(i for i, item in enumerate(items) if item is Ellipsis)
            used_axes = sum(item is not None and item is not Ellipsis for item in items)
            fill = 2 - used_axes
            if fill < 0:
                raise TypeError("DiaMatrix indexing expects A[i, j].")
            items[ellipsis_pos : ellipsis_pos + 1] = [slice(None)] * fill

        used_axes = sum(item is not None for item in items)
        if used_axes > 2:
            raise TypeError("DiaMatrix indexing expects A[i, j].")

        items.extend([slice(None)] * (2 - used_axes))
        return tuple(items)

    def _axis_indices(self, key, size: int) -> tuple[Array, bool, bool]:
        if isinstance(key, slice):
            return jnp.arange(size, dtype=jnp.int32)[key], False, False

        if isinstance(key, int) and not isinstance(key, bool):
            idx = jnp.asarray(_normalize_index(key, size), dtype=jnp.int32)
            return idx, True, False

        idx = jnp.asarray(key)
        if idx.dtype == jnp.bool_:
            if idx.ndim != 1:
                raise TypeError("DiaMatrix tuple boolean masks must be 1-D.")
            return jnp.nonzero(idx)[0].astype(jnp.int32), False, True

        idx = jnp.asarray(_normalize_index(idx, size), dtype=jnp.int32)
        return idx, idx.ndim == 0, idx.ndim > 0

    def _gather_rows(self, rows: Array) -> Array:
        original_shape = rows.shape
        rows = rows.reshape((-1,))
        selected = jax.vmap(self.get_row)(rows)
        return selected.reshape(original_shape + selected.shape[1:])

    def _gather_columns(self, cols: Array) -> Array:
        original_shape = cols.shape
        cols = cols.reshape((-1,))
        selected = jax.vmap(self.get_column)(cols)
        selected = selected.reshape(original_shape + selected.shape[1:])
        return jnp.moveaxis(selected, -1, 0)

    def _get_tuple_axes(self, row_key, col_key) -> tuple[Array, tuple[int, int]]:
        n, m = self.shape
        rows, row_scalar, row_advanced = self._axis_indices(row_key, n)
        cols, col_scalar, col_advanced = self._axis_indices(col_key, m)

        if row_scalar and col_scalar:
            return self._get_scalar(rows, cols), (0, 0)

        if row_scalar:
            return self.get_row(rows)[cols], (0, cols.ndim)

        if col_scalar:
            return self.get_column(cols)[rows], (rows.ndim, 0)

        if row_advanced and col_advanced:
            rows, cols = jnp.broadcast_arrays(rows, cols)
            return self._get_entries(rows, cols), (rows.ndim, 0)

        if _is_full_slice(col_key) and row_advanced:
            return self._gather_rows(rows), (rows.ndim, 1)

        if _is_full_slice(row_key) and col_advanced:
            return self._gather_columns(cols), (1, cols.ndim)

        row_grid = rows.reshape(rows.shape + (1,) * cols.ndim)
        col_grid = cols.reshape((1,) * rows.ndim + cols.shape)
        return self._get_entries(row_grid, col_grid), (rows.ndim, cols.ndim)

    def _apply_newaxes(
        self, result: Array, key: tuple, axis_counts: tuple[int, int]
    ) -> Array:
        if None not in key:
            return result

        final_shape = []
        result_axis = 0
        consumed_axis = 0

        for item in key:
            if item is None:
                final_shape.append(1)
                continue

            count = axis_counts[consumed_axis]
            final_shape.extend(result.shape[result_axis : result_axis + count])
            result_axis += count
            consumed_axis += 1

        return result.reshape(tuple(final_shape))

    def _get_single_axis(self, key: int | slice | Array) -> Array:
        n, m = self.shape

        if isinstance(key, list):
            raise TypeError(
                "Using a non-tuple sequence for multidimensional indexing is "
                "not allowed; use arr[array(seq)] instead of arr[seq]."
            )

        if isinstance(key, slice):
            rows = jnp.arange(n, dtype=jnp.int32)[key]
            return jax.vmap(self.get_row)(rows)

        if isinstance(key, int) and not isinstance(key, bool):
            return self.get_row(_normalize_index(key, n))

        if key is Ellipsis or key is None or isinstance(key, bool):
            if key is False:
                return jnp.zeros((0, n, m), dtype=self.data.dtype)
            _warn_dense_indexing()
            A = self.todense()
            if key is Ellipsis:
                return A
            return A[None, ...]

        idx = jnp.asarray(key)

        if idx.dtype == jnp.bool_:
            if idx.ndim == 0:
                if not bool(idx):
                    return jnp.zeros((0, n, m), dtype=self.data.dtype)
                _warn_dense_indexing()
                return self.todense()[None, ...]

            rows = jnp.arange(n, dtype=jnp.int32)
            if idx.ndim == 1:
                return jax.vmap(self.get_row)(rows[idx])

            row_idx, col_idx = jnp.nonzero(idx)
            return jax.vmap(self._get_scalar)(row_idx, col_idx)

        rows = _normalize_index(idx, n)
        if rows.ndim == 0:
            return self.get_row(rows)

        original_shape = rows.shape
        rows = rows.reshape((-1,))
        selected = jax.vmap(self.get_row)(rows)
        return selected.reshape(original_shape + selected.shape[1:])

    def _get_scalar(self, i: int | Array, j: int | Array) -> Array:
        n, m = self.shape

        i = jnp.asarray(i, dtype=jnp.int32)
        j = jnp.asarray(j, dtype=jnp.int32)

        offsets = jnp.asarray(self.offsets, dtype=jnp.int32)
        k = j - i

        ij_in_bounds = (i >= 0) & (i < n) & (j >= 0) & (j < m)
        safe_j = jnp.where((j >= 0) & (j < m), j, 0)

        vals = self.data[:, safe_j]
        matches = offsets == k

        # We never check for repeated offsets, so sum over all matches just in case.
        val = jnp.sum(jnp.where(matches, vals, jnp.zeros((), dtype=self.data.dtype)))

        return jnp.where(ij_in_bounds, val, jnp.zeros((), dtype=self.data.dtype))

    def _get_entries(self, i: int | Array, j: int | Array) -> Array:
        """Return A[i, j] for broadcast-compatible i and j.

        Supports scalar, vector, and broadcasted matrix-shaped ``i``/``j``.
        """
        n, m = self.shape

        i = jnp.asarray(i, dtype=jnp.int32)
        j = jnp.asarray(j, dtype=jnp.int32)

        offsets = jnp.asarray(self.offsets, dtype=jnp.int32)
        k = j - i

        in_bounds = (i >= 0) & (i < n) & (j >= 0) & (j < m)
        safe_j = jnp.where((j >= 0) & (j < m), j, 0)

        vals = jnp.take(self.data, safe_j, axis=1)
        vals = jnp.moveaxis(vals, 0, -1)
        matches = k[..., None] == offsets

        # Sum as offsets may be repeated
        out = jnp.sum(
            jnp.where(matches, vals, jnp.zeros((), dtype=self.data.dtype)), axis=-1
        )

        return jnp.where(in_bounds, out, jnp.zeros((), dtype=self.data.dtype))

    def _merge(self, other: DiaMatrix, sign: float) -> DiaMatrix:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        n, m = self.shape

        # ── fast path: identical offsets ────────────────────────────────────
        if self.offsets == other.offsets:
            return DiaMatrix(
                data=self.data + sign * other.data,
                offsets=self.offsets,
                shape=(n, m),
            )

        # ── general path: different offsets ─────────────────────────────────
        all_offsets = tuple(sorted(set(self.offsets) | set(other.offsets)))
        result_data = _merge_jit_kernel(
            self.data,
            other.data,
            self.offsets,
            other.offsets,
            float(sign),
        )
        return DiaMatrix(data=result_data, offsets=all_offsets, shape=(n, m))

    def __add__(self, other: DiaMatrix) -> DiaMatrix:
        return self._merge(other, +1.0)

    def __sub__(self, other: DiaMatrix) -> DiaMatrix:
        return self._merge(other, -1.0)

    @jax.jit
    def _matmul_compute(
        self, other: DiaMatrix
    ) -> tuple[Array, tuple[int, ...], tuple[int, int]]:
        n, m = self.shape
        _, l = other.shape
        if m != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for matrix product: ({n}, {m}) @ {other.shape}"
            )

        p_offs = self.offsets
        q_offs = other.offsets

        # Build all (Q, l) column-index arrays in one shot.
        # a_cols[q, j] = j - q_off[q]  — the A-column (= B-row) needed at output col j
        j = jnp.arange(l)
        p_offs_arr = jnp.array(p_offs)  # (P,)
        a_cols = j[None, :] - jnp.array(q_offs)[:, None]  # (Q, l)
        col_valid = (a_cols >= 0) & (a_cols < m)  # (Q, l): B-row / A-column in bounds
        safe_a_cols = jnp.where(col_valid, a_cols, 0)  # (Q, l)

        # Row-validity: self.data[p, a_col] represents A[a_col - p_off, a_col].
        # That row must also be in [0, n) — padding values outside the band must not
        # contribute even if they are non-zero.
        # a_rows[p, q, j] = a_cols[q, j] - p_off[p]  — row index of A accessed
        a_rows = a_cols[None, :, :] - p_offs_arr[:, None, None]  # (P, Q, l)
        row_valid = (a_rows >= 0) & (a_rows < n)  # (P, Q, l)

        # Gather: contribs[p, q, j] = self.data[p, safe_a_cols[q, j]] * other.data[q, j]
        # self.data[:, safe_a_cols] has shape (P, Q, l) via advanced indexing.
        contribs = jnp.where(
            col_valid[None] & row_valid,
            self.data[:, safe_a_cols] * other.data[None],
            jnp.zeros((), dtype=self.data.dtype),
        )  # (P, Q, l)

        # Group output-diagonal offsets in Python (static, no JAX graph nodes).
        # Only include offsets within the valid range for an (n, l) matrix.
        from collections import defaultdict

        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for pi, po in enumerate(p_offs):
            for qi, qo in enumerate(q_offs):
                out_off = po + qo
                if -(n - 1) <= out_off <= l - 1:
                    groups[out_off].append((pi, qi))

        if not groups:
            return jnp.zeros((1, l), dtype=self.data.dtype), (0,), (n, l)

        offsets_list = tuple(sorted(groups.keys()))
        data_out = jnp.stack(
            [
                jnp.sum(jnp.stack([contribs[pi, qi] for pi, qi in groups[r]]), axis=0)
                for r in offsets_list
            ]
        )
        return data_out, offsets_list, (n, l)

    @overload
    def __matmul__(self, other: Array) -> Array: ...
    @overload
    def __matmul__(self, other: JAXFunction) -> Array: ...
    @overload
    def __matmul__(self, other: DiaMatrix) -> DiaMatrix: ...
    @overload
    def __matmul__(self, other: Matrix) -> Matrix: ...
    @overload
    def __matmul__(self, other: MatrixProtocol) -> MatrixProtocol: ...
    def __matmul__(self, other: Array | JAXFunction | MatrixProtocol) -> Any:
        """Support ``A @ x`` (vector/matrix) and ``A @ B`` (DiaMatrix).

        DiaMatrix x DiaMatrix is computed purely in DIA format without
        materialising either operand as a dense array.

        """
        from jaxfun.galerkin import JAXFunction
        from jaxfun.la import Matrix

        if isinstance(other, JAXFunction):
            return self.matvec(other.array)

        if isinstance(other, Array):
            return self.matvec(other)

        if isinstance(other, Matrix):
            return Matrix(self.matvec(other.data))

        assert isinstance(other, DiaMatrix)

        data_out, offsets_out, shape_out = self._matmul_compute(other)
        return DiaMatrix(
            data=data_out,
            offsets=tuple(int(k) for k in offsets_out),
            shape=(int(shape_out[0]), int(shape_out[1])),
        )

    @overload
    def __rmatmul__(self, other: Array) -> Array: ...
    @overload
    def __rmatmul__(self, other: JAXFunction) -> Array: ...
    @overload
    def __rmatmul__(self, other: DiaMatrix) -> DiaMatrix: ...
    @overload
    def __rmatmul__(self, other: Matrix) -> Matrix: ...
    @overload
    def __rmatmul__(self, other: MatrixProtocol) -> MatrixProtocol: ...
    def __rmatmul__(self, other: Array | JAXFunction | MatrixProtocol) -> Any:
        """Support ``x @ A`` (row-vector or matrix on the left).

        Delegates to :meth:`rmatvec` which contracts the last axis of ``other``
        against the rows of ``A`` using the same sparse ``dynamic_slice + vmap``
        technique as :meth:`matvec`, without materialising a dense matrix or
        computing the transpose.
        """
        from jaxfun.galerkin import JAXFunction
        from jaxfun.la import Matrix

        if isinstance(other, JAXFunction):
            return self.rmatvec(other.array, axis=-1)

        if isinstance(other, Array):
            return self.rmatvec(other, axis=-1)

        # NOTE: unreachable via @ when both operands are concrete DiaMatrix/Matrix
        # instances — Matrix.__matmul__ and DiaMatrix.__matmul__ both handle those
        # cases first. These branches act as fallbacks for subclasses that do not
        # override __matmul__.
        if isinstance(other, Matrix):
            return Matrix(other.data @ self)

        assert isinstance(other, DiaMatrix)
        data_out, offsets_out, shape_out = other._matmul_compute(self)
        return DiaMatrix(
            data=data_out,
            offsets=tuple(int(k) for k in offsets_out),
            shape=(int(shape_out[0]), int(shape_out[1])),
        )

    def astype(self, dtype: jnp.dtype) -> DiaMatrix:
        """Return a copy with data cast to ``dtype``."""
        return DiaMatrix(
            data=self.data.astype(dtype), offsets=self.offsets, shape=self.shape
        )

    @property
    def nnz(self) -> int:
        """Number of explicitly stored (structurally non-zero) entries."""
        return self._size

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    def pin(
        self, constraints: dict[int, float] | tuple[tuple[int, float], ...]
    ) -> PinnedSystem:
        """Return a :class:`PinnedSystem` with the given DOFs fixed."""
        from jaxfun.la.pinned import PinnedSystem

        if isinstance(constraints, dict):
            constraints = tuple(sorted(constraints.items()))
        norm_constraints, new_offsets, data = self._pin(constraints)
        pinned_matrix = DiaMatrix(
            data=data, offsets=tuple(int(i) for i in new_offsets), shape=self.shape
        )
        return PinnedSystem(
            pinned_matrix, tuple((int(i), float(j)) for i, j in norm_constraints)
        )

    @jax.jit(static_argnums=(1,))
    def _pin(
        self, constraints: tuple[tuple[int, float], ...]
    ) -> tuple[list[tuple[int, float]], list[int], Array]:
        """Return a :class:`PinnedSystem` with the given DOFs fixed.

        For each ``(row_index, value)`` pair in *constraints* the corresponding
        row of the matrix is replaced by the identity row ``e_{row_index}``.
        The original sparsity pattern is preserved; if the main diagonal
        (offset 0) is not already stored it is added automatically.

        The LU factorisation of the modified matrix is cached on the returned
        :class:`PinnedSystem` so that repeated :meth:`~PinnedSystem.solve`
        calls are cheap.

        Args:
            constraints: Mapping from DOF index to pinned value, e.g.
                ``{0: 0.0}`` or ``{0: 0.0, -1: 1.0}``.  Negative indices
                are supported (Python-style, relative to ``n``).  Positive
                indices must be in ``[0, n)``, otherwise :exc:`IndexError`
                is raised.

        Returns:
            :class:`PinnedSystem` whose :meth:`~PinnedSystem.solve` method
            modifies the RHS and solves in one call.

        Example::

            A_sys = A.pin({0: 0.0})  # singular Fourier system
            x = A_sys.solve(b)
        """
        n, m = self.shape
        data = self.data  # keep as JAX array — no device→host transfer
        offsets: list[int] = list(self.offsets)

        # Ensure the main diagonal is present so we can write A[i,i] = 1.
        # The structural decision (whether to insert) uses only static ints.
        if 0 not in offsets:
            insert_pos = next(
                (k for k, off in enumerate(offsets) if off > 0), len(offsets)
            )
            offsets.insert(insert_pos, 0)
            zero_row = jnp.zeros((1, m), dtype=data.dtype)
            data = jnp.concatenate(
                [data[:insert_pos], zero_row, data[insert_pos:]], axis=0
            )

        main_idx = offsets.index(0)

        # Normalise negative indices (Python-style); reject out-of-range positives.
        norm_constraints: list[tuple[int, float]] = []
        for idx, val in constraints:
            if idx < -n or idx >= n:
                raise IndexError(
                    f"Constraint index {idx} is out of range for matrix of size {n}"
                )
            norm_constraints.append((idx % n, float(val)))

        # Zero every stored entry in each pinned row, then set diagonal to 1.
        # All indices (di, j) are Python ints — static at trace time — so
        # .at[...].set() is fully traceable under jax.jit / jax.vmap.
        for row_idx, _val in norm_constraints:
            for di, k in enumerate(offsets):
                j = row_idx + k
                if 0 <= j < m:
                    data = data.at[di, j].set(0.0)
            data = data.at[main_idx, row_idx].set(1.0)

        return norm_constraints, offsets, data

    def __repr__(self) -> str:
        n, m = self.shape
        nd = len(self.offsets)
        return (
            f"DiaMatrix(shape=({n}, {m}), dtype={self.dtype}, "
            f"n_diags={nd}, nnz={self.nnz}, offsets={self.offsets})"
        )


@jax.jit(static_argnums=(2, 3, 4))
def _merge_jit_kernel(
    data_a: Array,
    data_b: Array,
    self_offsets: tuple[int, ...],
    other_offsets: tuple[int, ...],
    sign: float,
) -> Array:
    """JIT-compiled core of :meth:`DiaMatrix._merge`.

    Keyed on the static offset tuples and sign so that repeated additions of
    matrices with the same sparsity pattern hit the JIT cache and run with
    zero Python overhead and zero host↔device transfers.

    Returns a ``(len(all_offsets), m)`` array of the merged diagonal data,
    where ``all_offsets = sorted(self_offsets | other_offsets)``.
    """
    all_offsets: tuple[int, ...] = tuple(sorted(set(self_offsets) | set(other_offsets)))
    self_map = {off: i for i, off in enumerate(self_offsets)}
    other_map = {off: i for i, off in enumerate(other_offsets)}

    # The loop is over static offsets so it unrolls completely at trace time.
    rows: list[Array] = []
    for k in all_offsets:
        in_self = k in self_map
        in_other = k in other_map
        if in_self and in_other:
            row = data_a[self_map[k]] + sign * data_b[other_map[k]]
        elif in_self:
            row = data_a[self_map[k]]
        else:
            row = sign * data_b[other_map[k]]
        rows.append(row)

    return jnp.stack(rows)  # (n_out, m)


class LUFactors(nnx.Pytree):
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
        self.L = nnx.data(L)
        self.U = nnx.data(U)
        self.shape = shape
        # perm[i] = original row index that was placed at row i after pivoting.
        # None means the identity permutation (no row swaps occurred).
        self.perm = perm

    @jax.jit(static_argnums=(2,))
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

    def get_pivots(self) -> Array | None:
        """Return the pivot permutation array, or None if no pivoting was done."""
        return self.perm

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
        k = k.astype(int)
        pivot = band[center, k]

        def s_step(s: Array, band: Array) -> Array:
            in_i = (k + s) < n
            f = jnp.where(in_i, band[center - s, k] / pivot, 0.0)
            band = band.at[center - s, k].set(jnp.where(in_i, f, band[center - s, k]))

            def u_step(u: Array, band: Array) -> Array:
                j = k + u
                in_j = j < n
                safe_j = jnp.where(in_j, j, 0)
                return band.at[center + u - s, safe_j].add(
                    jnp.where(in_i & in_j, -f * band[center + u, safe_j], 0.0)
                )

            return jax.lax.fori_loop(1, q + 1, u_step, band)

        return jax.lax.fori_loop(1, p + 1, s_step, band), None

    band_lu, _ = jax.lax.scan(elim_step, band, jnp.arange(n, dtype=int))
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
        k = k.astype(int)

        # --- partial pivot: find argmax |A[k..k+p, k]| ---
        # A[k+t, k] lives at band[center - t, k].  Use dynamic_slice to read
        # p+1 entries from column k without a Python loop.
        pivot_col = jax.lax.dynamic_slice(band[:, k], (center - p,), (p + 1,))
        pivot_vals = pivot_col[::-1]  # pivot_vals[t] = band[center-t, k]
        row_inds = k + jnp.arange(p + 1, dtype=int)
        masked = jnp.where(row_inds < n, jnp.abs(pivot_vals), 0.0)
        r_rel = jnp.argmax(masked).astype(int)
        r = k + r_rel

        # --- swap rows k ↔ r in DIA band storage ---
        #
        # In column-aligned DIA format:  band[center + off, j] = A[j - off, j].
        # Swapping rows k and r: for each band column j (= k+dj), exchange
        # band[center+dj, j] ↔ band[center+dj-r_rel, j].
        # fori_loop over the bw columns in the band window — avoids O(bw*p)
        # static unrolling.  dj ranges over [-center, q_eff] so that
        # s_k = center+dj ∈ [0, bw-1] is always in bounds.  s_r = s_k-r_rel
        # may be negative; we clamp it and mask the write so stray updates
        # never escape.
        def swap_col(dj: Array, band: Array) -> Array:
            j = k + dj
            in_j = (j >= 0) & (j < n)
            safe_j = jnp.where(in_j, j, 0)
            s_k = center + dj
            s_r = s_k - r_rel
            in_band_r = (s_r >= 0) & (s_r < bw)
            do_swap = in_j & in_band_r & (r_rel > 0)
            safe_s_r = jnp.maximum(s_r, 0)
            v_k = band[s_k, safe_j]
            v_r = band[safe_s_r, safe_j]
            band = band.at[s_k, safe_j].set(jnp.where(do_swap, v_r, v_k))
            band = band.at[safe_s_r, safe_j].set(jnp.where(do_swap, v_k, v_r))
            return band

        band = jax.lax.fori_loop(-center, q_eff + 1, swap_col, band)

        pk, pr = perm[k], perm[r]
        perm = perm.at[k].set(pr)
        perm = perm.at[r].set(pk)

        pivot = band[center, k]  # A[k, k] after swap

        # --- eliminate rows k+1 .. k+p via fori_loop (avoids O(p*q) unrolling) ---
        def s_step(s: Array, band: Array) -> Array:
            in_i = (k + s) < n
            f = jnp.where(in_i, band[center - s, k] / pivot, 0.0)
            band = band.at[center - s, k].set(jnp.where(in_i, f, band[center - s, k]))

            def u_step(u: Array, band: Array) -> Array:
                j = k + u
                in_j = j < n
                safe_j = jnp.where(in_j, j, 0)
                return band.at[center + u - s, safe_j].add(
                    jnp.where(in_i & in_j, -f * band[center + u, safe_j], 0.0)
                )

            return jax.lax.fori_loop(1, q_eff + 1, u_step, band)

        band = jax.lax.fori_loop(1, p + 1, s_step, band)

        return (band, perm), None

    perm0 = jnp.arange(n, dtype=int)
    (band_lu, perm), _ = jax.lax.scan(
        elim_step, (band, perm0), jnp.arange(n, dtype=int)
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
    # Important: iterate over the FULL range 1..p_max so that j always maps
    # to window slot (j+1)-th sub-diagonal.  Missing offsets get zero rows.
    p = max((-o for o in offsets if o < 0), default=0)
    if p == 0:
        return b  # L is identity

    l_rows: list[Array] = []
    for s in range(1, p + 1):  # s = 1, 2, ..., p
        off = -s
        if off in offsets and s < n:
            idx = offsets.index(off)
            d = L.data[idx]
            # d[j] = L[j+s, j]; shift right by s → l_rows[-1][i] = L[i, i-s]
            l_rows.append(jnp.concatenate([jnp.zeros(s, dtype=d.dtype), d[: n - s]]))
        else:
            l_rows.append(jnp.zeros(n, dtype=L.data.dtype))

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
    # Iterate over the FULL range 1..q_max so j always maps to the correct
    # window slot.  Missing offsets get zero rows.
    q = max((o for o in offsets if o > 0), default=0)

    # Reverse all xs so the scan runs from i = n-1 down to 0.
    rev = jnp.arange(n - 1, -1, -1)
    b_rev = b2d[rev]  # (n, k)
    diag_rev = diag_d[rev]  # (n,)

    if q == 0:
        # Pure diagonal: no coupling, no window needed.
        xs_rev = b_rev / diag_rev[:, None]  # (n, k)
        x = xs_rev[rev]  # un-reverse
        return x[:, 0] if scalar else x

    u_rows: list[Array] = []
    for s in range(1, q + 1):  # s = 1, 2, ..., q
        off = s
        if off in offsets and s < n:
            idx = offsets.index(off)
            d = U.data[idx]
            # d[i+s] = U[i, i+s]; shift left by s: d[s:n] ++ zeros(s)
            u_rows.append(jnp.concatenate([d[s:n], jnp.zeros(s, dtype=d.dtype)]))
        else:
            u_rows.append(jnp.zeros(n, dtype=U.data.dtype))

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
        diag_len = min(m - col_start, n - row_start)
        if len(d) == 1 and diag_len > 1:
            d = jnp.broadcast_to(d, (diag_len,))
        length = min(len(d), diag_len)
        if length > 0:
            # d[0] maps to A[row_start, col_start], i.e., column col_start
            row = row.at[col_start : col_start + length].set(d[:length])
        data_rows.append(row)

    data = jnp.stack(data_rows)  # (n_diags, m)
    return DiaMatrix(data=data, offsets=offsets, shape=(n, m))


def diakron(A: DiaMatrix, B: DiaMatrix) -> DiaMatrix:
    """Kronecker (tensor) product ``A ⊗ B`` of two DIA sparse matrices.

    When ``B`` is square (``p == q``), each pair of diagonals ``(k_a, k_b)``
    produces a single diagonal at offset ``k_a * p + k_b`` in the result, and
    the column-aligned data is the ravel of the outer product of the two
    row buffers.  When ``B`` is not square the method falls back to dense
    intermediates via :func:`jnp.kron`.

    Args:
        A: First DIA matrix of shape ``(m, n)``.
        B: Second DIA matrix of shape ``(p, q)``.

    Returns:
        A new :class:`DiaMatrix` of shape ``(m * p, n * q)`` representing the
        Kronecker product.

    Example:
        >>> import jax.numpy as jnp
        >>> I2 = diags([jnp.ones(2)], offsets=(0,), shape=(2, 2))
        >>> T3 = diags(
        ...     [-jnp.ones(2), 2 * jnp.ones(3), -jnp.ones(2)],
        ...     offsets=(-1, 0, 1),
        ... )
        >>> K = diakron(I2, T3)
        >>> K.shape
        (6, 6)
    """
    m, n = A.shape
    p, q = B.shape

    if p != q:
        # Non-square B: construct via dense intermediates.
        return DiaMatrix.from_dense(jnp.kron(A.todense(), B.todense()))

    # Fast DIA path — B is square (p == q).
    # For offset k_a in A and k_b in B the Kronecker product has a single
    # diagonal at K = k_a*p + k_b (result shape m*p × n*p).
    # Column j of the result splits as j = j0*p + j1 (j0 ∈ [0,n), j1 ∈ [0,p)),
    # and result.data[K_idx, j] = A.data[ka_idx, j0] * B.data[kb_idx, j1],
    # i.e. the ravel of the outer product A.data[ka, :] ⊗ B.data[kb, :].
    result_shape = (m * p, n * p)
    result_rows = m * p  # valid negative-offset bound: -(result_rows - 1)
    result_cols = n * p  # valid positive-offset bound:  (result_cols - 1)
    accumulated: dict[int, Array] = {}

    for ka_idx, k_a in enumerate(A.offsets):
        a_row = A.data[ka_idx]  # shape (n,)
        for kb_idx, k_b in enumerate(B.offsets):
            b_row = B.data[kb_idx]  # shape (p,)
            K = int(k_a) * p + int(k_b)
            # Skip diagonals entirely outside the result matrix.
            # The result has shape (result_rows, result_cols), so valid
            # offsets are [-(result_rows - 1), result_cols - 1].
            if result_cols <= K or -result_rows >= K:
                continue
            col_data: Array = (a_row[:, None] * b_row[None, :]).ravel()
            if K in accumulated:
                accumulated[K] = accumulated[K] + col_data
            else:
                accumulated[K] = col_data

    sorted_offsets = tuple(sorted(accumulated))
    result_data = jnp.stack([accumulated[k] for k in sorted_offsets])
    return DiaMatrix(data=result_data, offsets=sorted_offsets, shape=result_shape)


def _tridiag(n: int) -> tuple[Matrix, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    import numpy as np

    from .matrix import Matrix

    a = Matrix(
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
    )
    A = DiaMatrix.from_dense(a.data, offsets=(-1, 0, 1))
    return a, A
