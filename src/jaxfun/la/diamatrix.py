from __future__ import annotations

from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction

Array = jax.Array


class _CacheBox[T]:
    """Thin wrapper that provides identity-based equality and hashing.

    Flax NNX captures all instance ``__dict__`` entries as pytree aux_data
    (metadata).  Metadata is compared by equality on every JIT cache lookup.
    Storing a :class:`DiaMatrix` or a :class:`LUFactors` containing JAX arrays
    directly would trigger array equality checks and crash.  Wrapping the
    cached value in ``_CacheBox`` makes the comparison use ``is`` (identity),
    so the same cached object always compares equal to itself.
    """

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return type(other) is _CacheBox and self.value is other.value

    def __hash__(self) -> int:
        return id(self.value)

    def __repr__(self) -> str:
        return f"_CacheBox({self.value!r})"


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
        _T_cache: Private ``_CacheBox`` wrapping the transposed DiaMatrix,
            populated on the first call to the :attr:`T` property.
            Stored via ``object.__setattr__``, invisible to JAX's pytree
            machinery.

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
        _box: _CacheBox[dict[bool, LUFactors]] | None = getattr(self, "_lu_cache", None)
        if _box is None:
            _box = _CacheBox({})
            object.__setattr__(self, "_lu_cache", _box)
        cache: dict[bool, LUFactors] = _box.value
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
        perm = None if bool(jnp.all(perm_arr == jnp.arange(n, dtype=int))) else perm_arr
        result = LUFactors(L=L, U=U, shape=(n, n), perm=perm)
        cache[pivot] = result
        return result

    def solve(
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
        compile times for wide-banded Kronecker matrices.

        Args:
            b:               Right-hand side array.
            axis:            Axis of ``b`` along which the system is solved.
            pivot:           Passed to :meth:`lu_factor` (banded path only).
            dense_threshold: Maximum ``p * (q + 1)`` before switching to the
                             dense solver.  Default 100.

        Returns:
            Solution array with the same shape as ``b``.
        """
        offsets = self.offsets
        p = max((-k for k in offsets if k < 0), default=0)
        q = max((k for k in offsets if k > 0), default=0)

        if p * (q + 1) > dense_threshold:
            # Dense path: XLA compiles jnp.linalg.solve in milliseconds.
            n = self.shape[0]
            axis = axis % b.ndim
            if b.ndim == 1:
                return jnp.linalg.solve(self.todense(), b)
            b_moved = jnp.moveaxis(b, axis, 0)  # (n, *rest)
            rest_shape = b_moved.shape[1:]
            batch = b_moved.size // n
            b2d = b_moved.reshape(n, batch)  # (n, batch)
            x2d = jnp.linalg.solve(self.todense(), b2d)  # (n, batch)
            return jnp.moveaxis(x2d.reshape((n,) + rest_shape), 0, axis)

        return self.lu_factor(pivot=pivot).solve(b, axis=axis)

    @property
    def T(self) -> DiaMatrix:
        """Return the transpose ``A^T`` as a new :class:`DiaMatrix`.

        Shape becomes ``(m, n)``.  Each diagonal at offset ``k`` maps to
        offset ``-k`` in the transpose.  The data is re-aligned to the new
        column count ``n``.

        The result is cached as ``_T_cache`` (a :class:`_CacheBox`) via
        ``object.__setattr__`` so it is invisible to JAX's pytree machinery.
        Repeated calls always return the same transposed instance.
        """
        cached: DiaMatrix | None = getattr(self, "_T_cache", None)
        if cached is not None:
            return cached.value

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

        offsets_arr = jnp.asarray(self.offsets, dtype=int)
        new_data = jax.vmap(_transpose_row)(self.data, offsets_arr)  # (n_diags, n)
        new_offsets = tuple(-k for k in self.offsets)
        result = DiaMatrix(data=new_data, offsets=new_offsets, shape=new_shape)
        object.__setattr__(self, "_T_cache", _CacheBox(result))
        return result

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
        n, m = self.shape
        i = jnp.asarray(i, dtype=int)
        row = jnp.zeros(m, dtype=self.data.dtype)

        offsets_arr = jnp.array(self.offsets, dtype=int)

        def _place(row: Array, args: tuple) -> tuple[Array, None]:
            d, k = args  # d: (m,),  k: scalar int32
            j = i + k  # column where this diagonal hits row i
            in_bounds = (j >= 0) & (j < m)
            safe_j = jnp.where(in_bounds, j, 0)
            val = jnp.where(in_bounds, d[safe_j], jnp.zeros((), dtype=d.dtype))
            return row.at[safe_j].add(jnp.where(in_bounds, val, 0.0)), None

        row, _ = jax.lax.scan(_place, row, (self.data, offsets_arr))
        return row

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
        j = jnp.asarray(j, dtype=int)
        col = jnp.zeros(n, dtype=self.data.dtype)

        offsets_arr = jnp.array(self.offsets, dtype=int)

        def _place(col: Array, args: tuple) -> tuple[Array, None]:
            d, k = args  # d: (m,),  k: scalar int
            i = j - k  # row where this diagonal hits column j
            in_bounds = (i >= 0) & (i < n) & (j >= 0) & (j < m)
            safe_i = jnp.where(in_bounds, i, 0)
            safe_j = jnp.where(in_bounds, j, 0)
            val = jnp.where(in_bounds, d[safe_j], jnp.zeros((), dtype=d.dtype))
            return col.at[safe_i].add(val), None

        col, _ = jax.lax.scan(_place, col, (self.data, offsets_arr))
        return col

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

    def __getitem__(self, key: tuple[int, int]) -> Array:
        i, j = key
        return self.get_row(i)[j]

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
            return data_out, offsets_out, (n, l)

        offsets_list = tuple(sorted(accum.keys()))
        return (
            jnp.stack([accum[r] for r in offsets_list]),
            offsets_list,
            (n, l),
        )

    @overload
    def __matmul__(self, other: Array) -> Array: ...
    @overload
    def __matmul__(self, other: DiaMatrix) -> DiaMatrix: ...
    @overload
    def __matmul__(self, other: JAXFunction) -> Array: ...
    def __matmul__(self, other: Array | DiaMatrix | JAXFunction) -> Array | DiaMatrix:
        """Support ``A @ x`` (vector/matrix) and ``A @ B`` (DiaMatrix).

        DiaMatrix x DiaMatrix is computed purely in DIA format without
        materialising either operand as a dense array.

        """
        from jaxfun.galerkin import JAXFunction as _JAXFunction

        if not isinstance(other, DiaMatrix):
            return self.apply(other)

        if isinstance(other, _JAXFunction):
            return self.apply(other.array)

        data_out, offsets_out, shape_out = self._matmul_compute(other)
        return DiaMatrix(
            data=data_out,
            offsets=tuple(int(k) for k in offsets_out),
            shape=(int(shape_out[0]), int(shape_out[1])),
        )

    def __rmatmul__(self, other: Array) -> Array:
        """Support ``x @ A`` (row-vector or matrix on the left).

        Computes ``x @ A`` as an explicit transpose-matvec directly from this
        matrix's DIA storage, avoiding the ``.T`` cache and any pytree metadata
        side-effects (important when called inside a ``jax.jit``-traced method).

        For diagonal at offset ``k``:  ``A[j-k, j] = data[k_idx, j]``, so the
        contribution to output column ``j`` is ``data[k_idx, j] * other[..., j-k]``.
        Works for 1-D and 2-D ``other``.
        """
        n, m = self.shape
        dtype = jnp.result_type(other.dtype, self.data.dtype)
        j = jnp.arange(m)

        if other.ndim == 1:
            # other shape (n,) → result shape (m,)
            result = jnp.zeros(m, dtype=dtype)
            for k_idx, k in enumerate(self.offsets):
                i = j - k  # row index for column j
                valid = (i >= 0) & (i < n)
                safe_i = jnp.where(valid, i, 0)
                contrib = jnp.where(
                    valid, self.data[k_idx] * other[safe_i], jnp.zeros((), dtype=dtype)
                )
                result = result + contrib
            return result

        # other shape (..., n) → result shape (..., m)
        # Normalise: bring the contracting axis (n) to position 0 → (n, batch)
        batch_shape = other.shape[:-1]
        batch = other[..., 0].size  # number of batch elements
        x2d = other.reshape(batch, n).T  # (n, batch)
        res2d = jnp.zeros((m, batch), dtype=dtype)
        for k_idx, k in enumerate(self.offsets):
            i = j - k
            valid = (i >= 0) & (i < n)
            safe_i = jnp.where(valid, i, 0)
            # data[k_idx, j] scalar per j; x2d[safe_i, :] is (m, batch)
            scale = self.data[k_idx]  # (m,)
            vals = x2d[safe_i, :] * scale[:, None]  # (m, batch)
            vals = jnp.where(valid[:, None], vals, jnp.zeros((), dtype=dtype))
            res2d = res2d + vals
        return res2d.T.reshape(batch_shape + (m,))

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
        if off in offsets:
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
        if off in offsets:
            idx = offsets.index(off)
            d = U.data[idx]
            # d[i+s] = U[i, i+s]; shift left by s: d[s:] ++ zeros(s)
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
    accumulated: dict[int, Array] = {}

    for ka_idx, k_a in enumerate(A.offsets):
        a_row = A.data[ka_idx]  # shape (n,)
        for kb_idx, k_b in enumerate(B.offsets):
            b_row = B.data[kb_idx]  # shape (p,)
            K = int(k_a) * p + int(k_b)
            col_data: Array = (a_row[:, None] * b_row[None, :]).ravel()
            if K in accumulated:
                accumulated[K] = accumulated[K] + col_data
            else:
                accumulated[K] = col_data

    sorted_offsets = tuple(sorted(accumulated))
    result_data = jnp.stack([accumulated[k] for k in sorted_offsets])
    return DiaMatrix(data=result_data, offsets=sorted_offsets, shape=result_shape)


def _tridiag(n: int) -> tuple:
    """Return a symmetric tridiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    import numpy as np

    a = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
    )
    A = DiaMatrix.from_dense(jnp.array(a), offsets=(-1, 0, 1))
    return a, A
