from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from flax import nnx

Array = jax.Array

if TYPE_CHECKING:
    from jaxfun.la import DiaMatrix, Matrix


class DiaMatrixSolveMethod(StrEnum):
    AUTO = "auto"
    BANDED = "banded"
    RCM = "rcm"
    DENSE = "dense"


class SolverNotApplicable(Exception):
    """Raised when a solver strategy cannot be applied to the given matrix structure.

    Used by :func:`~jaxfun.galerkin.tensorproductspace.tpmats_lu_factor` and
    :func:`~jaxfun.galerkin.tensorproductspace.tpmats_wavenumber_factor` to
    signal that the factor-matrix structure is incompatible with the requested
    solver.  Caught by :meth:`~jaxfun.galerkin.TPMatrices.lu_factor` and
    :meth:`~jaxfun.galerkin.TPMatrices.solve` when selecting a fallback.
    """


class _CacheBox[T]:
    """Thin wrapper that provides identity-based equality and hashing.

    Flax NNX captures all instance ``__dict__`` entries as pytree aux_data
    (metadata).  Metadata is compared by equality on every JIT cache lookup.
    Storing a :class:`~jaxfun.la.DiaMatrix` or a :class:`LUFactors`
    containing JAX arrays directly would trigger array equality checks and
    crash.  Wrapping the cached value in ``_CacheBox`` makes the comparison
    use ``is`` (identity), so the same cached object always compares equal to
    itself.
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


class IndexedArray(nnx.Pytree):
    def __init__(self, i: int, data: Array):
        self.data = data
        self.i = i


class BaseMatrix(ABC, nnx.Pytree):
    """Nominal base class for matrix-like operators in ``jaxfun.la``.

    The base keeps unsupported behavior conservative: named operations raise
    ``NotImplementedError`` while binary operators return ``NotImplemented`` so
    Python can try reflected dispatch.
    """

    is_zero = False

    if TYPE_CHECKING:
        data: Array
        shape: tuple[int, int]

    @property
    def ndim(self) -> int:
        """Always 2 for matrix-like operators."""
        return 2

    def matvec(self, *_args: Any, **_kwargs: Any) -> Array:
        """Multiply ``A`` along ``axis`` of ``x``.

        ``x.shape[axis]`` must equal ``m``; the output has the same shape as
        ``x`` except ``shape[axis]`` becomes ``n``.
        """
        raise NotImplementedError

    def _check_same_shape(self, other, /) -> None:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

    def _check_square(self, operation: str) -> tuple[int, int]:
        n, m = self.shape
        if n != m:
            raise ValueError(
                f"{operation} requires a square matrix, got shape {self.shape}"
            )
        return n, m

    def _check_matmul_shape(self, other, /) -> tuple[int, int]:
        n, m = self.shape
        if m != other.shape[0]:
            raise ValueError(
                f"Shape mismatch for matrix product: ({n}, {m}) @ {other.shape}"
            )
        return n, m

    def _check_same_data_shape(self, other, /, *, label: str = "Data") -> None:
        if self.data.shape != other.data.shape:
            raise ValueError(
                f"{label} shape mismatch: {self.data.shape} vs {other.data.shape}"
            )

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        """Element dtype."""

    @abstractmethod
    def solve(self, *_args: Any, **_kwargs: Any) -> Array:
        """Solve ``A x = b`` along ``axis`` of ``b``."""

    def lu_solve(self, *_args: Any, **_kwargs: Any) -> Array:
        """Solve ``A x = b`` using LU factors along ``axis`` of ``b``."""
        raise NotImplementedError

    def lu_factor(self) -> Any:
        """Return LU factors and pivot indices."""
        raise NotImplementedError

    @property
    def T(self) -> BaseMatrix:
        """Transpose ``A^T``."""
        raise NotImplementedError

    def diagonal(self, k: int = 0) -> Array:
        """Return the ``k``-th diagonal as a 1-D array."""
        raise NotImplementedError

    @property
    def is_diagonal(self) -> bool:
        """Whether this matrix is purely main-diagonal."""
        return self.diagonal_or_none() is not None

    def diagonal_or_none(self) -> Array | None:
        """Return the main diagonal only when this matrix is purely diagonal."""
        return None

    @abstractmethod
    def todense(self) -> Array:
        """Return or compute the equivalent dense ``(n, m)`` array."""

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        """Return a sparse representation of the matrix."""
        raise NotImplementedError

    def to_matrix(self) -> Matrix:
        """Return a Matrix representation of the matrix."""
        raise NotImplementedError

    def get_row(self, i: int | Array) -> Array:
        """Return row ``i`` as a dense 1-D array of length ``m``."""
        raise NotImplementedError

    def get_column(self, j: int | Array) -> Array:
        """Return column ``j`` as a dense 1-D array of length ``n``."""
        raise NotImplementedError

    @abstractmethod
    def scale(self, *_args: Any, **_kwargs: Any) -> BaseMatrix:
        """Return ``alpha * A``."""

    def astype(self, dtype: jnp.dtype) -> Self:
        """Return a copy cast to ``dtype``."""
        raise NotImplementedError

    @property
    def size(self) -> int:
        """Return total number of nonzero elements"""
        raise NotImplementedError

    def _as_array(self, u: Any) -> Any:
        from jaxfun.galerkin import JAXFunction

        return u.get_array() if isinstance(u, JAXFunction) else u

    def __call__(self, u: Any) -> Any:
        raise NotImplementedError

    def __getitem__(self, key: tuple[int, int], /) -> Array:
        raise NotImplementedError

    def __matmul__(self, other, /):
        return NotImplemented

    def __rmatmul__(self, other, /):
        return NotImplemented

    def __mul__(self, other: complex | Array, /):
        return self.scale(other)

    def __rmul__(self, other: complex | Array, /):
        return self.scale(other)

    def __neg__(self):
        return self.scale(-1)

    def __len__(self) -> int:
        raise NotImplementedError

    def __add__(self, other, /):
        return NotImplemented

    def __radd__(self, other, /):
        return self.__add__(other)

    def __sub__(self, other, /):
        if isinstance(other, BaseMatrix):
            self._check_same_shape(other)
        try:
            other = -other
        except TypeError:
            return NotImplemented
        return self.__add__(other)

    def __rsub__(self, other, /):
        if isinstance(other, BaseMatrix):
            self._check_same_shape(other)
        try:
            self_negative = -self
        except TypeError:
            return NotImplemented
        return self_negative.__add__(other)


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
