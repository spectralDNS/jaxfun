from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from jaxfun.la.matrixprotocol import BaseMatrix

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction


class TensorMatrix(BaseMatrix):  # noqa: B903
    """Non-separable tensor with dims * 2 indices.

    For test function v_{ij} and trial function u_{kl}, the tensor
    represents  A_{ikjl}.

    Matrix vector product

    .. math::

        v_{ij} = \\sum_{k,l} A_{ikjl} u_{kl}

    Stored when coefficient is non-separable in coordinates so Kron
    factorization is unavailable.

    Attributes:
        data: Dense (or sparse) global matrix.
    """

    is_zero = False
    is_diagonal = False

    def __init__(self, data: Array) -> None:
        self.data = data  # mat is A_ikjl

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.data.shape[0] * self.data.shape[2]),
            int(self.data.shape[1] * self.data.shape[3]),
        )

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @jax.jit(static_argnums=0)
    def _matmul_array(self, w: Array) -> Array:
        return jnp.einsum("ikjl,kl->ij", self.data, w)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply matrix to coefficient array u."""
        w = self._as_array(u)
        return self._matmul_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    @jax.jit(static_argnums=0)
    def _rmatmul_array(self, w: Array) -> Array:
        return jnp.einsum("ij,ikjl->kl", w, self.data)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        w = self._as_array(u)
        return self._rmatmul_array(w)

    def solve(self, rhs: Array, axis: int = 0) -> Array:
        """Solve A x = rhs for x."""
        _ = axis
        AT = self.todense()
        return jnp.linalg.solve(AT, rhs.flatten()).reshape(rhs.shape)

    def todense(self) -> Array:
        """Return the equivalent 2-D global matrix."""
        return jnp.transpose(self.data, (0, 2, 1, 3)).reshape(self.shape)

    def scale(self, alpha: complex | Array) -> TensorMatrix:
        return TensorMatrix(self.data * alpha)

    def __add__(self, other):
        from jaxfun.la import ZeroMatrix

        if isinstance(other, ZeroMatrix):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return self
        if isinstance(other, TensorMatrix):
            if self.data.shape != other.data.shape:
                raise ValueError(
                    f"Tensor shape mismatch: {self.data.shape} vs {other.data.shape}"
                )
            return TensorMatrix(self.data + other.data)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from jaxfun.la import ZeroMatrix

        if isinstance(other, ZeroMatrix):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return self
        if isinstance(other, TensorMatrix):
            if self.data.shape != other.data.shape:
                raise ValueError(
                    f"Tensor shape mismatch: {self.data.shape} vs {other.data.shape}"
                )
            return TensorMatrix(self.data - other.data)
        return NotImplemented

    def __rsub__(self, other):
        from jaxfun.la import ZeroMatrix

        if isinstance(other, ZeroMatrix):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {other.shape} vs {self.shape}")
            return -self
        return NotImplemented
