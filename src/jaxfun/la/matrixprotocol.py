from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

import jax
import jax.numpy as jnp

Array = jax.Array

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction


@runtime_checkable
class MatrixProtocol(Protocol):
    """Structural interface shared by all matrix types in ``jaxfun.la``.

    Any class that implements all of these methods and properties satisfies the
    protocol without needing to inherit from it.  Use ``@runtime_checkable``
    to allow ``isinstance(A, MatrixProtocol)`` checks at runtime.

    Example::

        from jaxfun.la import DiaMatrix, Matrix, MatrixProtocol
        import jax.numpy as jnp


        def apply_operator(A: MatrixProtocol, x: Array) -> Array:
            return A.matvec(x)


        A_dense = Matrix(jnp.eye(4))
        A_sparse = DiaMatrix.from_dense(jnp.eye(4))

        assert isinstance(A_dense, MatrixProtocol)
        assert isinstance(A_sparse, MatrixProtocol)
    """

    @property
    def shape(self) -> tuple[int, int]:
        """``(n, m)`` shape of the matrix."""
        ...

    @property
    def ndim(self) -> int:
        """Always 2."""
        ...

    @property
    def dtype(self) -> jnp.dtype:
        """Element dtype."""
        ...

    def matvec(self, x: Array, axis: int = 0) -> Array:
        """Multiply ``A`` along ``axis`` of ``x``.

        ``x.shape[axis]`` must equal ``m``; the output has the same shape as
        ``x`` except ``shape[axis]`` becomes ``n``.
        """
        ...

    def matmat(self, X: Array, axis: int = 0) -> Array:
        """Alias for :meth:`matvec` treating ``X`` as a batch of column vectors."""
        ...

    def apply(self, x: Array, axis: int = 0) -> Array:
        """Apply ``A`` along ``axis`` of ``x`` (alias for :meth:`matvec`)."""
        ...

    def solve(self, b: Array, axis: int = 0) -> Array:
        """Solve ``A x = b`` along ``axis`` of ``b``."""
        ...

    def lu_solve(self, b: Array, axis: int = 0) -> Array:
        """Solve ``A x = b`` using LU factors along ``axis`` of ``b``."""
        ...

    def lu_factor(self) -> LUProtocol:
        """Return LU factors and pivot indices."""
        ...

    @property
    def T(self) -> MatrixProtocol:
        """Transpose ``A^T``."""
        ...

    def diagonal(self, k: int = 0) -> Array:
        """Return the ``k``-th diagonal as a 1-D array."""
        ...

    def to_dense(self) -> Array:
        """Return or compute the equivalent dense ``(n, m)`` array."""
        ...

    def todense(self) -> Array:
        """Return or compute the equivalent dense ``(n, m)`` array."""
        ...

    def get_row(self, i: int | Array) -> Array:
        """Return row ``i`` as a dense 1-D array of length ``m``."""
        ...

    def get_column(self, j: int | Array) -> Array:
        """Return column ``j`` as a dense 1-D array of length ``n``."""
        ...

    def scale(self, alpha: float | Array) -> MatrixProtocol:
        """Return ``alpha * A``."""
        ...

    def astype(self, dtype: jnp.dtype) -> MatrixProtocol:
        """Return a copy cast to ``dtype``."""
        ...

    @property
    def size(self) -> int:
        """Return total number of nonzero elements"""
        ...

    def __getitem__(self, key: tuple[int, int]) -> Array: ...
    def __mul__(self, other: float | Array) -> MatrixProtocol: ...
    def __rmul__(self, other: float | Array) -> MatrixProtocol: ...
    def __neg__(self) -> MatrixProtocol: ...
    def __len__(self) -> int: ...
    def __add__(self, other: MatrixProtocol) -> MatrixProtocol: ...
    def __sub__(self, other: MatrixProtocol) -> MatrixProtocol: ...
    @overload
    def __matmul__(self, other: Array) -> Array: ...
    @overload
    def __matmul__(self, other: MatrixProtocol) -> MatrixProtocol: ...
    @overload
    def __matmul__(self, other: JAXFunction) -> Array: ...
    def __rmatmul__(self, other: Array) -> Array: ...


@runtime_checkable
class LUProtocol(Protocol):
    """Structural interface for LU factorization and solving linear systems."""

    def solve(self, b: Array, axis: int = 0) -> Array:
        """Solve linear system using LU factors."""
        ...

    def get_pivots(self) -> Array | None:
        """Return pivot indices or None if no pivoting was done."""
        ...

    def __repr__(self) -> str:
        """String representation of the LU factorization."""
        ...
