from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction

Array = jax.Array


@nnx.dataclass
class Matrix(nnx.Pytree):
    """Dense matrix class backed by a JAX 2-D array.

    Mirrors the interface of :class:`~jaxfun.la.sparsemat.DiaMatrix` so that
    dense and sparse matrices can be used interchangeably.

    Attributes:
        data: The underlying ``(n, m)`` JAX array.
        _lu_cache: Private ``LUFactors | None`` populated on the first call to
            :meth:`lu_factor`.  Subsequent calls return the cached object at
            no extra cost.  Stored via ``object.__setattr__`` so it is
            invisible to JAX's pytree machinery and does not affect JIT
            tracing or compilation.

    Example::

        A = Matrix(jnp.eye(4))
        y = A.matvec(jnp.ones(4))  # shape (4,)
        Y = A.matvec(jnp.ones((4, 3)))  # shape (4, 3)  — acts on axis 0
        Y = A.matvec(jnp.ones((3, 4)), axis=1)  # shape (3, 4)
    """

    data: Array

    def __init__(self, data: Array):
        self.data = jnp.asarray(data)
        if self.data.ndim != 2:
            raise ValueError(
                f"Matrix requires a 2-D array, got shape {self.data.shape}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """``(n, m)`` shape of the matrix."""
        return cast(tuple[int, int], self.data.shape)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        """Return total number of elements (``n * m``)."""
        n, m = self.shape
        return n * m

    @jax.jit(static_argnums=(2,))
    def matvec(self, x: Array, axis: int = 0) -> Array:
        """Multiply ``A`` along ``axis`` of ``x``.

        Args:
            x:    Input array.  ``x.shape[axis]`` must equal ``m``.
            axis: The axis of ``x`` along which the matrix acts.  The output
                  has the same shape as ``x`` except ``shape[axis]`` becomes
                  ``n``.

        Examples::

            A.matvec(x)  # x shape (m,)      → (n,)
            A.matvec(X, axis=0)  # X shape (m, k)    → (n, k)
            A.matvec(X, axis=1)  # X shape (k, m)    → (k, n)
            A.matvec(T, axis=2)  # T shape (a, b, m) → (a, b, n)
        """
        n, m = self.shape

        if x.ndim == 1:
            return self.data @ x

        axis = axis % x.ndim
        x_moved = jnp.moveaxis(x, axis, 0)  # (m, *rest)
        rest_shape = x_moved.shape[1:]
        batch = x_moved.size // m
        x2d = x_moved.reshape(m, batch)  # (m, batch)
        y2d = self.data @ x2d  # (n, batch)
        y_moved = y2d.reshape((n,) + rest_shape)
        return jnp.moveaxis(y_moved, 0, axis)

    @jax.jit(static_argnums=(2,))
    def matmat(self, X: Array, axis: int = 0) -> Array:
        """Alias for :meth:`matvec` — multiply ``A`` along ``axis`` of ``X``."""
        return self.matvec(X, axis=axis)

    @jax.jit(static_argnums=(2,))
    def apply(self, x: Array, axis: int = 0) -> Array:
        """Apply ``A`` along ``axis`` of ``x`` (alias for :meth:`matvec`)."""
        return self.matvec(x, axis=axis)

    def lu_factor(self) -> LUFactors:
        """Compute the LU factorisation of this (square) matrix.

        The result is cached on the matrix instance so that repeated calls
        pay the factorisation cost only once.  The cache is stored as
        ``_lu_cache`` via ``object.__setattr__`` so it is invisible to JAX's
        pytree machinery.

        Uses :func:`jax.scipy.linalg.lu_factor` (partial pivoting).

        Returns:
            :class:`LUFactors` containing the packed ``(lu, piv)`` arrays
            and the original shape.

        Raises:
            ValueError: if the matrix is not square.
        """
        cached: LUFactors | None = getattr(self, "_lu_cache", None)
        if cached is not None:
            return cached
        n, m = self.shape
        if n != m:
            raise ValueError(
                f"lu_factor requires a square matrix, got shape {self.shape}"
            )
        lu, piv = jax.scipy.linalg.lu_factor(self.data)
        result = LUFactors(lu=lu, piv=piv, shape=(n, n))
        object.__setattr__(self, "_lu_cache", result)
        return result

    def solve(self, b: Array, axis: int = 0) -> Array:
        """Solve ``A x = b`` via LU factorisation.

        Args:
            b:    Right-hand side array.  ``b.shape[axis]`` must equal ``n``.
            axis: The axis of ``b`` along which the system is solved.  All
                  other axes are treated as independent batch dimensions.
                  The output has the same shape as ``b``.

        Examples::

            A.solve(b)  # b shape (n,)      → (n,)
            A.solve(B, axis=0)  # B shape (n, k)    → (n, k)
            A.solve(B, axis=1)  # B shape (k, n)    → (k, n)
            A.solve(T, axis=2)  # T shape (a, b, n) → (a, b, n)
        """
        return self.lu_factor().solve(b, axis=axis)

    @property
    def T(self) -> Matrix:
        """Return the transpose ``A^T`` as a new :class:`Matrix`."""
        return Matrix(self.data.T)

    def diagonal(self, k: int = 0) -> Array:
        """Return the ``k``-th diagonal as a 1-D array.

        Args:
            k: Diagonal offset (0 = main, positive = upper, negative = lower).
        """
        return jnp.diag(self.data, k)

    def to_dense(self) -> Array:
        """Return the underlying ``(n, m)`` array (identity — already dense)."""
        return self.data

    todense = to_dense

    def get_row(self, i: int | Array) -> Array:
        """Return row ``i`` as a dense 1-D array of length ``m``.

        Args:
            i: Row index (0-based).  May be a traced JAX scalar so the method
               is usable inside :func:`jax.jit` / :func:`jax.vmap`.

        Returns:
            Array of shape ``(m,)``.
        """
        return self.data[i]

    def get_column(self, j: int | Array) -> Array:
        """Return column ``j`` as a dense 1-D array of length ``n``.

        Args:
            j: Column index (0-based).  May be a traced JAX scalar so the
               method is usable inside :func:`jax.jit` / :func:`jax.vmap`.

        Returns:
            Array of shape ``(n,)``.
        """
        return self.data[:, j]

    def scale(self, alpha: float | Array) -> Matrix:
        """Return ``alpha * A`` as a new :class:`Matrix`."""
        return Matrix(self.data * alpha)

    def __mul__(self, other: float | Array) -> Matrix:
        return self.scale(other)

    def __rmul__(self, other: float | Array) -> Matrix:
        return self.scale(other)

    def __neg__(self) -> Matrix:
        return self.scale(-1)

    def __len__(self) -> int:
        return min(self.shape)

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Matrix(self.data + other.data)

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Matrix(self.data - other.data)

    def __getitem__(self, key: tuple[int, int]) -> Array:
        i, j = key
        return self.data[i, j]

    @overload
    def __matmul__(self, other: Array) -> Array: ...
    @overload
    def __matmul__(self, other: Matrix) -> Matrix: ...
    @overload
    def __matmul__(self, other: JAXFunction) -> Array: ...
    def __matmul__(self, other: Array | Matrix | JAXFunction) -> Array | Matrix:
        """Support ``A @ x`` (vector/array) and ``A @ B`` (Matrix)."""
        from jaxfun.galerkin import JAXFunction as _JAXFunction

        if isinstance(other, Matrix):
            n, m = self.shape
            if m != other.shape[0]:
                raise ValueError(
                    f"Shape mismatch for matrix product: ({n}, {m}) @ {other.shape}"
                )
            return Matrix(self.data @ other.data)
        elif isinstance(other, _JAXFunction):
            return self @ other.array
        return self.apply(other)

    def __rmatmul__(self, other: Array) -> Array:
        """Support ``x @ A`` (row-vector or 2-D array on the left)."""
        if other.ndim == 1:
            return self.data.T @ other
        return other @ self.data

    def astype(self, dtype: jnp.dtype) -> Matrix:
        """Return a copy with data cast to ``dtype``."""
        return Matrix(self.data.astype(dtype))

    def __repr__(self) -> str:
        n, m = self.shape
        return f"Matrix(shape=({n}, {m}), dtype={self.dtype})"


class LUFactors:
    """Result of :meth:`Matrix.lu_factor`.

    Holds the packed LU decomposition as returned by
    :func:`jax.scipy.linalg.lu_factor`.

    Example::

        A = Matrix(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
        lu = A.lu_factor()
        x = lu.solve(jnp.array([1.0, 2.0]))
    """

    def __init__(
        self,
        lu: Array,
        piv: Array,
        shape: tuple[int, int],
    ):
        self.lu = lu  # packed (n, n) LU output from jax.scipy.linalg.lu_factor
        self.piv = piv  # (n,) int32 pivot indices
        self.shape = shape

    def solve(self, b: Array, axis: int = 0) -> Array:
        """Solve ``A x = b`` using the stored LU factorisation.

        Args:
            b:    Right-hand side array.  ``b.shape[axis]`` must equal ``n``.
            axis: The axis of ``b`` along which the system is solved.  All
                  other axes are treated as independent batch dimensions.
                  The output has the same shape as ``b``.

        Examples::

            lu.solve(b)  # b shape (n,)      → (n,)
            lu.solve(B, axis=0)  # B shape (n, k)    → (n, k)
            lu.solve(B, axis=1)  # B shape (k, n)    → (k, n)
            lu.solve(T, axis=2)  # T shape (a, b, n) → (a, b, n)
        """
        n = self.shape[0]

        if b.ndim == 1:
            return jax.scipy.linalg.lu_solve((self.lu, self.piv), b)

        axis = axis % b.ndim
        b_moved = jnp.moveaxis(b, axis, 0)  # (n, *rest)
        rest_shape = b_moved.shape[1:]
        batch = b_moved.size // n
        b2d = b_moved.reshape(n, batch)  # (n, batch)
        x2d = jax.scipy.linalg.lu_solve((self.lu, self.piv), b2d)  # (n, batch)
        x_moved = x2d.reshape((n,) + rest_shape)
        return jnp.moveaxis(x_moved, 0, axis)

    def __repr__(self) -> str:
        n, m = self.shape
        return f"LUFactors(shape=({n}, {m}))"
