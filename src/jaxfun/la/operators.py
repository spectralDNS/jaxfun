from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Self, overload

import jax.numpy as jnp
from jax import Array

from jaxfun.la import DiagonalMatrix
from jaxfun.la.diamatrix import DiaMatrix, diags
from jaxfun.la.matrix import Matrix
from jaxfun.la.matrixprotocol import BaseMatrix

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction


def _as_state_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    return (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)


def _dtype_or_default(dtype) -> jnp.dtype:
    return jnp.dtype(jnp.float32 if dtype is None else dtype)


def _check_same_shape(left, right) -> None:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")


class SpecialMatrix(BaseMatrix):
    """Base class for shape-preserving special matrix operators."""

    is_diagonal = True

    def __init__(
        self, state_shape: int | tuple[int, ...], dtype: jnp.dtype | None = None
    ) -> None:
        self.state_shape = _as_state_shape(state_shape)
        self._dtype = _dtype_or_default(dtype)

    @property
    def shape(self) -> tuple[int, int]:
        n = int(prod(self.state_shape))
        return (n, n)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    @property
    def size(self) -> int:
        return self.shape[0]

    @property
    def data(self) -> Array:
        return self.todense()

    def diagonal(self, k: int = 0) -> Array:
        raise NotImplementedError

    def diagonal_or_none(self) -> Array:
        return self.diagonal().reshape(self.state_shape)

    @property
    def T(self) -> Self:
        return self

    def todense(self) -> Array:
        return jnp.diag(self.diagonal())

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        return diags([self.diagonal()], offsets=(0,))

    def to_matrix(self) -> Matrix:
        return Matrix(self.todense())

    def get_row(self, i: int | Array) -> Array:
        return self.todense()[i]

    def get_column(self, j: int | Array) -> Array:
        return self.todense()[:, j]

    def scale(self, alpha: complex | Array):
        raise NotImplementedError

    def astype(self, dtype: jnp.dtype) -> Self:
        return type(self)(self.state_shape, dtype=dtype)

    def __call__(self, u: Array | JAXFunction) -> Array:
        raise NotImplementedError

    def __matmul__(self, other: Array | JAXFunction) -> Array:
        return self(other)

    def __rmatmul__(self, other: Array | JAXFunction) -> Array:
        return self(other)

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: tuple[int, int], /) -> Array:
        return self.todense()[key]


class IdentityMatrix(SpecialMatrix):
    """Identity operator that preserves coefficient-array shape."""

    is_zero = False

    def diagonal(self, k: int = 0) -> Array:
        if k != 0:
            n = max(self.shape[0] - abs(k), 0)
            return jnp.zeros(n, dtype=self.dtype)
        return jnp.ones(self.shape[0], dtype=self.dtype)

    def matvec(self, x: Array, axis: int = 0) -> Array:
        return x.astype(jnp.result_type(x.dtype, self.dtype), copy=False)

    def solve(self, b: Array, axis: int = 0) -> Array:
        return b

    lu_solve = solve

    def lu_factor(self) -> IdentityMatrix:
        return self

    def get_pivots(self) -> None:
        return None

    def scale(self, alpha: complex | Array) -> DiagonalMatrix:
        return DiagonalMatrix(jnp.ones(self.shape[0])).scale(alpha)

    def __call__(self, u: Array | JAXFunction) -> Array:
        return self._as_array(u)

    @overload
    def __add__(self, other: ZeroMatrix) -> IdentityMatrix: ...
    @overload
    def __add__[T](self, other: T) -> T: ...
    def __add__(self, other):
        if isinstance(other, ZeroMatrix):
            _check_same_shape(self, other)
            return self
        if isinstance(other, IdentityMatrix | DiaMatrix):
            _check_same_shape(self, other)
            return self.scale(1) + other
        if isinstance(other, Matrix):
            _check_same_shape(self, other)
            return other + self
        return NotImplemented

    @overload
    def __sub__(self, other: ZeroMatrix) -> IdentityMatrix: ...
    @overload
    def __sub__[T](self, other: T) -> T: ...
    def __sub__(self, other):
        if isinstance(other, ZeroMatrix):
            _check_same_shape(self, other)
            return self
        if isinstance(other, IdentityMatrix | DiaMatrix):
            _check_same_shape(self, other)
            return self.scale(1) - other
        if isinstance(other, Matrix):
            _check_same_shape(self, other)
            return self.to_matrix() - other
        return NotImplemented

    @overload
    def __rsub__(self, other: ZeroMatrix) -> IdentityMatrix: ...
    @overload
    def __rsub__[T](self, other: T) -> T: ...
    def __rsub__[T](self, other: T) -> Self | T:
        if isinstance(other, ZeroMatrix):
            _check_same_shape(self, other)
            return -self
        if isinstance(other, DiaMatrix | Matrix):
            _check_same_shape(self, other)
            return other - self  # ty:ignore[invalid-return-type]
        return NotImplemented


class ZeroMatrix(SpecialMatrix):
    """Zero operator that preserves coefficient-array shape on application."""

    is_zero = True

    @property
    def size(self) -> int:
        return 0

    def diagonal(self, k: int = 0) -> Array:
        n = self.shape[0] if k == 0 else max(self.shape[0] - abs(k), 0)
        return jnp.zeros(n, dtype=self.dtype)

    def matvec(self, x: Array, axis: int = 0) -> Array:
        return jnp.zeros_like(x)

    def solve(self, b: Array, axis: int = 0) -> Array:
        raise ValueError("Cannot solve a system with the zero operator")

    lu_solve = solve

    def lu_factor(self) -> ZeroMatrix:
        raise ValueError("Cannot factorize the zero operator")

    def get_row(self, i: int | Array) -> Array:
        return jnp.zeros(self.shape[1], dtype=self.dtype)

    def get_column(self, j: int | Array) -> Array:
        return jnp.zeros(self.shape[0], dtype=self.dtype)

    def scale(self, alpha: complex | Array) -> ZeroMatrix:
        result_dtype = jnp.result_type(alpha, self.dtype)
        if result_dtype == self.dtype:
            return self
        return ZeroMatrix(self.state_shape, dtype=result_dtype)

    def __call__(self, u: Array | JAXFunction) -> Array:
        w = self._as_array(u)
        return jnp.zeros_like(w)

    def __add__[T](self, other: T) -> T:
        if hasattr(other, "shape"):
            _check_same_shape(self, other)
        return other

    def __sub__[T](self, other: T) -> T:
        return self.__add__(-other)  # ty:ignore[unsupported-operator]

    def __rsub__[T](self, other: T) -> T:
        return self.__add__(other)

    def __getitem__(self, key: tuple[int, int], /) -> Array:
        return jnp.zeros((), dtype=self.dtype)
