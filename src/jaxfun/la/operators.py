from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Self, overload

import jax.numpy as jnp
from flax import nnx
from jax import Array

from jaxfun.la.diamatrix import DiagonalMatrix, DiaMatrix, diags
from jaxfun.la.matrix import Matrix
from jaxfun.la.matrixprotocol import BaseMatrix

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction


def _as_state_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    return (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)


def _dtype_or_default(dtype) -> jnp.dtype:
    return jnp.dtype(jnp.float32 if dtype is None else dtype)


class GlobalArray(nnx.Pytree):
    def __init__(self, i: int, data: Array):
        self.data = data
        self.i = i


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

    @overload
    def __matmul__(self, other: Array | JAXFunction) -> Array: ...
    @overload
    def __matmul__(self, other: DiaMatrix | Matrix) -> DiaMatrix | Matrix: ...
    def __matmul__(
        self, other: Array | JAXFunction | DiaMatrix | Matrix
    ) -> Array | DiaMatrix | Matrix:
        return self._as_array(other)

    @overload
    def __rmatmul__(self, other: Array | JAXFunction) -> Array: ...
    @overload
    def __rmatmul__(self, other: DiaMatrix | Matrix) -> DiaMatrix | Matrix: ...
    def __rmatmul__(
        self, other: Array | JAXFunction | DiaMatrix | Matrix
    ) -> Array | DiaMatrix | Matrix:
        return self._as_array(other)

    def __add__(self, other):
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
    def __matmul__(self, other: Array | JAXFunction) -> Array: ...
    @overload
    def __matmul__(self, other: DiaMatrix | Matrix) -> DiaMatrix | Matrix: ...
    def __matmul__(
        self, other: Array | JAXFunction | DiaMatrix | Matrix
    ) -> Array | DiaMatrix | Matrix:
        if isinstance(other, DiaMatrix | Matrix):
            self._check_matmul_shape(other)
            return other
        return self._as_array(other)

    @overload
    def __rmatmul__(self, other: Array | JAXFunction) -> Array: ...
    @overload
    def __rmatmul__(self, other: DiaMatrix | Matrix) -> DiaMatrix | Matrix: ...
    def __rmatmul__(
        self, other: Array | JAXFunction | DiaMatrix | Matrix
    ) -> Array | DiaMatrix | Matrix:
        if isinstance(other, DiaMatrix | Matrix):
            if other.shape[1] != self.shape[0]:
                raise ValueError(
                    f"Shape mismatch for matrix product: {other.shape} @ {self.shape}"
                )
            return other
        return self._as_array(other)

    @overload
    def __add__(self, other: ZeroMatrix) -> IdentityMatrix: ...
    @overload
    def __add__(self, other: IdentityMatrix) -> DiagonalMatrix: ...
    @overload
    def __add__[T: DiaMatrix | Matrix](self, other: T) -> T: ...
    def __add__(self, other):
        if isinstance(other, ZeroMatrix):
            self._check_same_shape(other)
            return self
        if isinstance(other, DiaMatrix | Matrix):
            self._check_same_shape(other)
            return self.scale(1) + other
        if isinstance(other, IdentityMatrix):
            self._check_same_shape(other)
            return self.scale(1) + other.scale(1)
        return NotImplemented

    @overload
    def __sub__(self, other: ZeroMatrix) -> IdentityMatrix: ...
    @overload
    def __sub__(self, other: IdentityMatrix) -> ZeroMatrix: ...
    @overload
    def __sub__[T: DiaMatrix | Matrix](self, other: T) -> T: ...
    def __sub__(self, other):
        if isinstance(other, ZeroMatrix):
            self._check_same_shape(other)
            return self
        if isinstance(other, DiaMatrix | Matrix):
            self._check_same_shape(other)
            return self.scale(1) - other
        if isinstance(other, IdentityMatrix):
            self._check_same_shape(other)
            dtype = jnp.result_type(self.dtype, other.dtype)
            return ZeroMatrix(self.state_shape, dtype=dtype)
        return NotImplemented

    @overload
    def __rsub__(self, other: ZeroMatrix) -> DiagonalMatrix: ...
    @overload
    def __rsub__(self, other: IdentityMatrix) -> ZeroMatrix: ...
    @overload
    def __rsub__[T: DiaMatrix | Matrix](self, other: T) -> T: ...
    def __rsub__(self, other):
        if isinstance(other, ZeroMatrix):
            self._check_same_shape(other)
            return -self
        if isinstance(other, IdentityMatrix):
            self._check_same_shape(other)
            dtype = jnp.result_type(self.dtype, other.dtype)
            return ZeroMatrix(self.state_shape, dtype=dtype)
        if isinstance(other, DiaMatrix | Matrix):
            self._check_same_shape(other)
            return other.__sub__(self.scale(1))
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

    @overload
    def __matmul__(self, other: Array | JAXFunction) -> Array: ...
    @overload
    def __matmul__(self, other: DiaMatrix | Matrix) -> Matrix: ...
    def __matmul__(
        self, other: Array | JAXFunction | DiaMatrix | Matrix
    ) -> Array | Matrix:
        if isinstance(other, DiaMatrix | Matrix):
            self._check_matmul_shape(other)
            n = self.shape[0]
            k = other.shape[1]
            dtype = jnp.result_type(self.dtype, other.dtype)
            return Matrix(jnp.zeros((n, k), dtype=dtype))
        w = self._as_array(other)
        return jnp.zeros_like(w)

    @overload
    def __rmatmul__(self, other: Array | JAXFunction) -> Array: ...
    @overload
    def __rmatmul__(self, other: DiaMatrix | Matrix) -> Matrix: ...
    def __rmatmul__(
        self, other: Array | JAXFunction | DiaMatrix | Matrix
    ) -> Array | Matrix:
        if isinstance(other, DiaMatrix | Matrix):
            n, m = self.shape
            if other.shape[1] != n:
                raise ValueError(
                    f"Shape mismatch for matrix product: {other.shape} @ {self.shape}"
                )
            dtype = jnp.result_type(self.dtype, other.dtype)
            return Matrix(jnp.zeros((other.shape[0], m), dtype=dtype))
        w = self._as_array(other)
        return jnp.zeros_like(w)

    def __add__[T](self, other: T) -> T:
        if isinstance(other, BaseMatrix):
            self._check_same_shape(other)
        return other

    def __sub__[T](self, other: T) -> T:
        return self.__add__(-other)  # ty:ignore[unsupported-operator]

    def __rsub__[T](self, other: T) -> T:
        return self.__add__(other)

    def __getitem__(self, key: tuple[int, int], /) -> Array:
        return jnp.zeros((), dtype=self.dtype)


class _DataDelegatingMixin:
    """Mixin that delegates all BaseMatrix operations to ``self.data``."""

    data: DiaMatrix | Matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    def matvec(self, x: Array, axis: int = 0) -> Array:
        return self.data.matvec(x, axis=axis)

    def lu_factor(self, **kwargs):
        return self.data.lu_factor(**kwargs)

    def lu_solve(self, b: Array, axis: int = 0, **kwargs) -> Array:
        return self.data.lu_solve(b, axis=axis, **kwargs)

    def solve(self, b: Array, axis: int = 0, **kwargs) -> Array:
        return self.data.solve(b, axis=axis, **kwargs)

    @property
    def T(self) -> BaseMatrix:
        return self.data.T

    def diagonal(self, k: int = 0) -> Array:
        return self.data.diagonal(k)

    def diagonal_or_none(self) -> Array | None:
        return self.data.diagonal_or_none()

    @property
    def is_diagonal(self) -> bool:
        return self.data.is_diagonal

    def todense(self) -> Array:
        return self.data.todense()

    def tosparse(self, *, tol: int = 100) -> DiaMatrix:
        return self.data.tosparse(tol=tol)

    def to_matrix(self) -> Matrix:
        return self.data.to_matrix()

    def get_row(self, i: int | Array) -> Array:
        return self.data.get_row(i)

    def get_column(self, j: int | Array) -> Array:
        return self.data.get_column(j)

    @property
    def size(self) -> int:
        return self.data.size

    def __call__(self, u: Array | JAXFunction) -> Array:
        return self.data(u)

    def __getitem__(self, key: tuple[int, int], /) -> Array:
        return self.data[key]

    def __matmul__(self, other) -> Array:
        return self.data @ other

    def __rmatmul__(self, other) -> Array:
        return other @ self.data

    def __add__(self, other):
        return self.data + other

    def __sub__(self, other):
        return self.data - other

    def __rsub__(self, other):
        return other - self.data

    def __len__(self) -> int:
        return len(self.data)


class GlobalMatrix(_DataDelegatingMixin, BaseMatrix):
    data: DiaMatrix | Matrix

    def __init__(self, global_indices: tuple[int, int], matrix: DiaMatrix | Matrix):
        self.data = matrix
        self.global_indices = global_indices

    def scale(self, alpha: complex | Array) -> GlobalMatrix:
        return GlobalMatrix(self.global_indices, self.data.scale(alpha))

    def astype(self, dtype: jnp.dtype) -> GlobalMatrix:
        return GlobalMatrix(self.global_indices, self.data.astype(dtype))
