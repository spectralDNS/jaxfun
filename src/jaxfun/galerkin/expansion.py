"""Orthogonal series with exact arithmetic.

`OrthogonalExpansion` is a function u(x) = sum_k c_k psi_k(x) in a 1D orthogonal
basis (Chebyshev, Legendre, Jacobi, Ultraspherical, Fourier), much like the
series classes of `numpy.polynomial`. Unlike a `JAXFunction`, which is a
symbolic argument of weak forms and keeps the size of its function space, an
expansion grows as needed: the product of two expansions holds the exact
product, and so does the product with the coordinate x.
"""

from __future__ import annotations

import numbers
from typing import Any, Self, cast

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.coordinates import BaseScalar

from .arguments import JAXFunction, _as_number, _is_number
from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace


@jax.tree_util.register_pytree_node_class
class OrthogonalExpansion:
    r"""A series :math:`u(x) = \sum_{k=0}^{N-1} c_k \psi_k(x)` in an orthogonal basis.

    Supports, with exact results:

    * ``u + w``, ``u - w``, ``-u`` and the same with a number, which is added
      as a constant function. Operands of different lengths are padded.
    * ``c * u`` and ``u / c`` for a number c.
    * ``u * w`` of lengths M and N, of length ``M + N - 1`` (``M + N`` for
      Fourier), and ``u**n`` for an integer n >= 0.
    * ``x * u`` for the coordinate x of the basis, of length ``N + 1``.
      Not for Fourier, where x is not a basis function.
    * ``u.deriv(k)``, evaluation ``u(x)`` and conversion to and from a
      JAXFunction.

    The expansion is a JAX pytree with the coefficients as its only leaf, so it
    can pass through ``jax.jit``, ``jax.grad`` and ``jax.vmap``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from jaxfun.galerkin import Chebyshev
        >>> from jaxfun.galerkin.expansion import OrthogonalExpansion
        >>> T = Chebyshev.Chebyshev(3)
        >>> x = T.system.x
        >>> u = OrthogonalExpansion(jnp.array([0.0, 1.0, 0.0]), T)  # u(x) = x
        >>> w = u * u - x * u  # x**2 - x**2
        >>> assert w.N == 5 and jnp.allclose(w.coef, 0.0, atol=1e-6)

    Args:
        coef: Coefficients of the series.
        space: The orthogonal basis. It is resized to the number of
            coefficients if needed.
    """

    coef: Array
    space: OrthogonalSpace

    def __init__(self, coef: Array, space: OrthogonalSpace) -> None:
        if space.orthogonal is not space:
            raise TypeError(
                f"{type(space).__name__} is not an orthogonal space; use "
                "OrthogonalExpansion.from_jaxfunction for a space with boundary "
                "conditions"
            )
        self.coef = coef
        self.space = space if len(coef) == space.N else space.resized(len(coef))

    def tree_flatten(self) -> tuple[tuple[Array], OrthogonalSpace]:
        return (self.coef,), self.space

    @classmethod
    def tree_unflatten(cls, space: OrthogonalSpace, children: tuple[Array]) -> Self:
        obj = object.__new__(cls)
        obj.coef = children[0]
        obj.space = space
        return obj

    @classmethod
    def from_jaxfunction(cls, u: JAXFunction) -> Self:
        """Return the JAXFunction `u` of a 1D space as an expansion in the
        space's orthogonal basis, boundary lifting included."""
        if u.functionspace.dims != 1:
            raise NotImplementedError("Only 1D JAXFunctions can be converted")
        w = u.to_orthogonal()
        return cls(cast(Array, w.array), cast(OrthogonalSpace, w.functionspace))

    @classmethod
    def from_expr(cls, expr: sp.Expr, space: OrthogonalSpace) -> Self:
        """Return the projection of `expr` onto the orthogonal basis `space`."""
        from .inner import project

        return cls(project(expr, space), space)

    @property
    def N(self) -> int:
        """Number of basis functions."""
        return self.space.N

    @property
    def degree(self) -> int:
        """Index of the last basis function."""
        return self.N - 1

    @property
    def dim(self) -> int:
        """Dimension of the space, the number of coefficients."""
        return self.space.dim

    @property
    def num_dofs(self) -> int:
        """Number of coefficients."""
        return self.space.num_dofs

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the quadrature mesh of the space."""
        return self.space.shape

    def resize(self, N: int) -> OrthogonalExpansion:
        """Return the expansion padded with zeros, or truncated, to N basis
        functions."""
        return OrthogonalExpansion(
            self.space.resize_coefficients(self.coef, N), self.space.resized(N)
        )

    def truncate(self, N: int) -> OrthogonalExpansion:
        """Return the expansion truncated to its first N basis functions."""
        if N > self.N:
            raise ValueError(f"Cannot truncate {self.N} basis functions to {N}")
        return self.resize(N)

    def __call__(self, x: float | Array) -> Array:
        """Evaluate the expansion at points x of the physical domain."""
        return self.space.evaluate(x, self.coef)

    def deriv(self, k: int = 1) -> OrthogonalExpansion:
        """Return the k'th derivative as an expansion of the same length."""
        # derivative_coeffs differentiates in the reference coordinate
        s = float(self.space.domain_factor) ** k
        return OrthogonalExpansion(
            s * self.space.derivative_coeffs(self.coef, k), self.space
        )

    def to_jaxfunction(self, V: Any, truncate: bool = False) -> JAXFunction:
        """Return the expansion as a JAXFunction of the 1D space V.

        V must have the same orthogonal basis, of any size. Its boundary
        conditions are imposed by `V.from_orthogonal`, which assumes the
        expansion satisfies them.

        Args:
            V: Target function space.
            truncate: Allow dropping basis functions beyond the size of V.
        """
        O = V.get_orthogonal()
        if not self._same_basis(O):
            raise TypeError(f"{V.name} does not share the basis of {self.space.name}")
        if self.N > O.N and not truncate:
            raise ValueError(
                f"The expansion has {self.N} basis functions and {V.name} only "
                f"{O.N}; pass truncate=True to drop the rest"
            )
        return JAXFunction(V.from_orthogonal(self.resize(O.N).coef), V)

    def mul_x(self) -> OrthogonalExpansion:
        """Return x * u, with one more basis function, from the recurrence
        x Q = A^T Q of the reference coordinate."""
        if not isinstance(self.space, Jacobi):
            raise NotImplementedError(
                f"{type(self.space).__name__} has no recurrence for multiplication by x"
            )
        S = self.space.resized(self.N + 1)
        c = S.resize_coefficients(self.coef, S.N)
        # x = x0 + X / s, with X the reference coordinate
        x0 = float(S.map_true_domain(0.0))
        s = float(S.domain_factor)
        return OrthogonalExpansion(x0 * c + S.A() @ c / s, S)

    def _same_basis(self, space: OrthogonalSpace) -> bool:
        return _family(self.space) == _family(space)

    def _coerce(self, other: Any) -> OrthogonalExpansion | None:
        if isinstance(other, OrthogonalExpansion):
            if not self._same_basis(other.space):
                raise TypeError(
                    f"Expansions in {self.space.name} and {other.space.name} have "
                    "different bases"
                )
            return other
        if isinstance(other, JAXFunction):
            return self._coerce(OrthogonalExpansion.from_jaxfunction(other))
        if _is_number(other):
            # The constant as a series of the same length: c / psi_0 at index 0
            psi0 = self.space.evaluate(
                jnp.array([float(self.space.domain.lower)]),
                jnp.ones(1, dtype=self.coef.dtype),
            )[0]
            coef = jnp.zeros_like(self.coef).at[0].set(_as_number(other) / psi0)
            return OrthogonalExpansion(coef, self.space)
        return None

    def _combine(self, other: OrthogonalExpansion, b: complex) -> OrthogonalExpansion:
        N = max(self.N, other.N)
        S = self.space.resized(N)
        c = S.resize_coefficients(self.coef, N) + b * S.resize_coefficients(
            other.coef, N
        )
        return OrthogonalExpansion(c, S)

    def __add__(self, other: Any) -> OrthogonalExpansion:
        w = self._coerce(other)
        return NotImplemented if w is None else self._combine(w, 1)

    __radd__ = __add__

    def __sub__(self, other: Any) -> OrthogonalExpansion:
        w = self._coerce(other)
        return NotImplemented if w is None else self._combine(w, -1)

    def __rsub__(self, other: Any) -> OrthogonalExpansion:
        w = self._coerce(other)
        return NotImplemented if w is None else w._combine(self, -1)

    def __neg__(self) -> OrthogonalExpansion:
        return OrthogonalExpansion(-self.coef, self.space)

    def __mul__(self, other: Any) -> OrthogonalExpansion:
        if _is_number(other):
            return OrthogonalExpansion(_as_number(other) * self.coef, self.space)
        if isinstance(other, BaseScalar):
            if other != self.space.system.base_scalars()[0]:
                return NotImplemented
            return self.mul_x()
        w = self._coerce(other)
        if w is None:
            return NotImplemented
        return OrthogonalExpansion(
            self.space.product_coeffs(self.coef, w.coef),
            self.space,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> OrthogonalExpansion:
        if not _is_number(other):
            return NotImplemented
        return OrthogonalExpansion(self.coef / _as_number(other), self.space)

    def __pow__(self, n: int) -> OrthogonalExpansion:
        if not isinstance(n, numbers.Integral) or n < 0:
            return NotImplemented
        if n == 0:
            one = self._coerce(1)
            assert one is not None
            return one
        result: OrthogonalExpansion | None = None
        base = self
        while True:
            if n & 1:
                result = base if result is None else result * base
            n >>= 1
            if not n:
                return result  # ty:ignore[invalid-return-type]
            base = base * base

    def __repr__(self) -> str:
        return f"OrthogonalExpansion({self.coef}, {self.space.name})"


def _family(space: OrthogonalSpace) -> tuple[Any, ...]:
    return (type(space), tuple(space.parameters.items()), space.domain, space.system)
