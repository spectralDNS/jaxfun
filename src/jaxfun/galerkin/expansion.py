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
from collections.abc import Sequence
from typing import Any, Self, cast

import jax.numpy as jnp
import sympy as sp
from flax import nnx
from jax import Array

from jaxfun.coordinates import BaseScalar
from jaxfun.utils import ulp

from .arguments import JAXFunction, _as_number, _is_number
from .Fourier import RFourier
from .Jacobi import Jacobi
from .orthogonal import OrthogonalSpace


class OrthogonalExpansion(nnx.Pytree):
    r"""A series :math:`u(x) = \sum_{k=0}^{N-1} c_k \psi_k(x)` in an orthogonal basis.

    Supports, with exact results:

    * ``u + w``, ``u - w``, ``-u`` and the same with a number, which is added
      as a constant function. Operands of different lengths are padded.
    * ``c * u`` and ``u / c`` for a number c.
    * ``u * w`` of lengths M and N, of length ``M + N - 1`` (``M + N`` for
      Fourier), and ``u**n`` for an integer n >= 0.
    * ``p * u`` for a polynomial p in the coordinate x of the basis, for
      example ``x * u`` or ``(x**2 - 1) * u``, of length ``N + deg(p)``.
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
        if isinstance(space, RFourier) and space.n_extra and len(coef) == space.N:
            # The padding rows only make the axis divide among devices
            coef, space = coef[: space.n_real], space.resized(space.n_real)
        self.coef = nnx.data(coef)
        self.space = nnx.static(
            space if len(coef) == space.N else space.resized(len(coef))
        )

    @classmethod
    def from_jaxfunction(cls, u: JAXFunction) -> Self:
        """Return the JAXFunction `u` of a 1D space as an expansion in the
        space's orthogonal basis, boundary lifting included."""
        if u.functionspace.dims != 1:
            raise NotImplementedError("Only 1D JAXFunctions can be converted")
        w = u.to_orthogonal()
        return cls(cast(Array, w.array), cast(OrthogonalSpace, w.functionspace))

    @classmethod
    def from_expr(
        cls,
        expr: sp.Expr,
        space: OrthogonalSpace | type[OrthogonalSpace],
        *,
        tol: float | None = None,
        N_max: int = 1024,
        **kwargs: Any,
    ) -> OrthogonalExpansion:
        """Return the projection of `expr` onto an orthogonal basis.

        If `space` is a basis, the expansion has its size. If it is a basis
        class, such as ``Chebyshev.Chebyshev``, the size is chosen to resolve
        `expr`: degree + 1 for a polynomial in x and a polynomial basis, and
        otherwise the trimmed projection on the first N = 16, 32, ... whose
        last quarter of coefficients is below `tol` relative to the largest.

        Args:
            expr: Expression in the coordinate x.
            space: Orthogonal basis, or the class of one.
            tol: Relative tolerance, used with a basis class. Defaults to the
                roundoff of the projection, see
                `OrthogonalSpace.projection_roundoff`.
            N_max: Largest N to try with a basis class.
            kwargs: Keyword arguments of the basis class besides N, e.g.
                domain or alpha. The coordinate system defaults to that of
                the coordinate in `expr`.

        Raises:
            ValueError: If `expr` is not resolved with N_max basis functions.

        Examples:
            >>> from jaxfun.galerkin import Chebyshev
            >>> from jaxfun.galerkin.expansion import OrthogonalExpansion
            >>> x = Chebyshev.Chebyshev(8).system.x
            >>> OrthogonalExpansion.from_expr(x**2, Chebyshev.Chebyshev).N
            3
        """
        from .inner import project

        if isinstance(space, OrthogonalSpace):
            return cls(project(expr, space), space)
        if "system" not in kwargs:
            systems = [
                s._system for s in expr.free_symbols if isinstance(s, BaseScalar)
            ]
            if systems:
                kwargs["system"] = systems[0]
        if issubclass(space, Jacobi):
            S = space(2, **kwargs)
            coeffs = cls._x_polynomial_of(expr, S)
            if coeffs is not None:
                S = S.resized(len(coeffs))
                return cls(project(expr, S), S)
        n = 16
        while n <= N_max:
            S = space(n, **kwargs)
            u = cls(project(expr, S), S)
            m = u.dim
            u = u.trim(S.projection_roundoff() if tol is None else tol)
            if u.dim <= m - m // 4:
                return u
            n *= 2
        raise ValueError(
            f"{expr} is not resolved by {N_max} basis functions; increase N_max or tol"
        )

    def trim(self, tol: float | None = None) -> OrthogonalExpansion:
        """Return the expansion without the trailing coefficients smaller than
        `tol` times the largest one.

        For Fourier the highest wavenumbers are dropped. Since the size of the
        result depends on the coefficient values, `trim` cannot run inside
        ``jax.jit``.

        Args:
            tol: Relative tolerance. Defaults to ``ulp(1)``.
        """
        if tol is None:
            tol = float(ulp(1))
        scale = float(jnp.abs(self.coef).max())
        return self.resize(self.space.trimmed_dim(self.coef, tol * scale))

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

    def __len__(self) -> int:
        """Dimension of space, the number of coefficients."""
        return self.dim

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
        m = O.n_real if isinstance(O, RFourier) else O.N
        if self.dim > m and not truncate:
            raise ValueError(
                f"The expansion has {self.dim} basis functions and {V.name} only "
                f"{m}; pass truncate=True to drop the rest"
            )
        return JAXFunction(V.from_orthogonal(O.resize_coefficients(self.coef, O.N)), V)

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

    def mul_poly(self, coeffs: Sequence[complex]) -> OrthogonalExpansion:
        """Return p(x) * u for the polynomial p(x) = sum_k coeffs[k] x**k, with
        len(coeffs) - 1 more basis functions."""
        if len(coeffs) == 0:
            raise ValueError("The polynomial needs at least one coefficient")
        # Horner: p u = (...(a_n x + a_{n-1}) x + ...) u
        result = coeffs[-1] * self
        for a in reversed(coeffs[:-1]):
            result = result.mul_x() + a * self
        return result

    def _x_polynomial(self, expr: sp.Expr) -> list[complex] | None:
        return self._x_polynomial_of(expr, self.space)

    @staticmethod
    def _x_polynomial_of(expr: sp.Expr, space: OrthogonalSpace) -> list[complex] | None:
        # Coefficients (lowest degree first) of expr as a polynomial in the
        # coordinate x of the basis, or None if it is not one.
        x = space.system.base_scalars()[0]
        if not expr.free_symbols <= {x}:
            return None
        try:
            poly = sp.Poly(expr, x)
        except sp.PolynomialError:
            return None
        return [_as_number(a) for a in reversed(poly.all_coeffs())]

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
        if isinstance(other, sp.Expr):
            coeffs = self._x_polynomial(other)
            return NotImplemented if coeffs is None else self.mul_poly(coeffs)
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
