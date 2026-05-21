"""Exponential time differencing Runge-Kutta integrators."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from jax.scipy.linalg import expm as _expm

from jaxfun.la import DiagonalMatrix, Matrix
from jaxfun.typing import Array, FunctionSpaceType, Padding

from .base import BaseIntegrator

expm = cast(Callable[[Array], Array], _expm)
type ETDCoefficients = tuple[DiagonalMatrix, ...] | tuple[Matrix, ...]


def _phi1(z: Array) -> Array:
    """Return ``phi_1(z) = (exp(z) - 1) / z`` with a small-argument expansion."""
    small = jnp.abs(z) < 1e-7
    series = 1 + z / 2 + z**2 / 6 + z**3 / 24 + z**4 / 120
    return jnp.where(small, series, jnp.expm1(z) / z)


def _phi2(z: Array) -> Array:
    """Return ``phi_2(z) = (exp(z) - 1 - z) / z^2`` with series stabilization."""
    small = jnp.abs(z) < 1e-6
    series = 0.5 + z / 6 + z**2 / 24 + z**3 / 120 + z**4 / 720
    return jnp.where(small, series, (jnp.expm1(z) - z) / z**2)


def _phi3(z: Array) -> Array:
    """Return ``phi_3(z) = (exp(z) - 1 - z - z^2 / 2) / z^3`` safely."""
    small = jnp.abs(z) < 1e-5
    series = 1 / 6 + z / 24 + z**2 / 120 + z**3 / 720 + z**4 / 5040
    return jnp.where(small, series, (jnp.expm1(z) - z - z**2 / 2) / z**3)


def _etdrk4_nonlinear_weights(
    phi1: Array, phi2: Array, phi3: Array
) -> tuple[Array, Array, Array]:
    """Return ETDRK4 nonlinear combination weights.

    Coeffs for N(u_n), N(a_n), N(b_n), N(c_n) in Kassam & Trefethen.
    """
    f1 = phi1 - 3 * phi2 + 4 * phi3
    f2 = phi2 - 2 * phi3
    f3 = 4 * phi3 - phi2
    return f1, f2, f3


def _etdrk4_diag_coeffs(L: Array, dt: float) -> tuple[DiagonalMatrix, ...]:
    """Return ETDRK4 coefficients for a diagonal linear operator."""
    z = dt * L
    E = jnp.exp(z)
    E2 = jnp.exp(z / 2)
    # Stage coefficient:
    #   L^{-1}(exp(hL/2)-I) = (h/2) * phi1(hL/2)
    # and step uses h*Q, hence Q = 0.5 * phi1(hL/2).
    Q = 0.5 * _phi1(z / 2)
    phi1 = _phi1(z)
    phi2 = _phi2(z)
    phi3 = _phi3(z)
    f1, f2, f3 = _etdrk4_nonlinear_weights(phi1, phi2, phi3)
    return tuple(DiagonalMatrix(c.reshape((-1,))) for c in (E, E2, Q, f1, f2, f3))


def _phi_matrices(z: Array) -> tuple[Array, Array, Array]:
    """Return dense matrix versions of ``phi_1``, ``phi_2``, and ``phi_3``."""
    n = z.shape[0]
    I = jnp.eye(n, dtype=z.dtype)
    expz = expm(z)
    zinv = jnp.linalg.pinv(z)
    phi1 = zinv @ (expz - I)
    phi2 = zinv @ (phi1 - I)
    phi3 = zinv @ (phi2 - 0.5 * I)
    return phi1, phi2, phi3


class ETDRK4(BaseIntegrator):
    """Fourth-order exponential time differencing for semilinear systems."""

    def __init__(
        self,
        V: FunctionSpaceType,
        equation: sp.Expr,
        *,
        initial: sp.Expr | Array,
        time: tuple[float, float] | None = None,
        **params,
    ):
        """Construct an ETDRK4 integrator for a semilinear weak form."""
        super().__init__(V, equation, initial=initial, time=time, **params)
        zero = jnp.zeros(self.functionspace.num_dofs)
        forcing_rhs = (
            self.apply_mass_inverse(jnp.asarray(self.linear_forcing))
            if self.linear_forcing is not None
            else zero
        )
        self._forcing_rhs = nnx.data(forcing_rhs)

    def setup(self, dt: float) -> None:
        """Precompute ETD propagators and nonlinear stage coefficients."""
        is_diag = self._has_diagonal_etd()
        self.is_diag = nnx.static(is_diag)

        coeffs = self._setup_diagonal_etd(dt) if is_diag else self._setup_dense_etd(dt)

        E, E2, Q, f1, f2, f3 = coeffs
        self.E = nnx.data(E)
        self.E2 = nnx.data(E2)
        self.Q = nnx.data(Q)
        self.f1 = nnx.data(f1)
        self.f2 = nnx.data(f2)
        self.f3 = nnx.data(f3)

    def _mass_diagonal_or_identity(self) -> Array | None:
        """Return the mass diagonal, treating a missing mass operator as identity."""
        return self.mass_operator.diagonal_or_none()

    def _has_diagonal_etd(self) -> bool:
        """Return True when the ETD coefficients can stay elementwise."""
        mass_diag = self._mass_diagonal_or_identity()
        linear_diag = self.linear_operator.diagonal_or_none()
        return mass_diag is not None and linear_diag is not None

    def _setup_diagonal_etd(self, dt: float) -> ETDCoefficients:
        """Return ETD coefficients for the diagonal operator path."""
        mass_diag = self._mass_diagonal_or_identity()
        assert mass_diag is not None
        linear_diag = self.linear_operator.diagonal_or_none()
        Ldiag = cast(Array, linear_diag) / mass_diag
        return _etdrk4_diag_coeffs(Ldiag, dt)

    def _setup_dense_etd(self, dt: float) -> ETDCoefficients:
        """Return ETD coefficients for the dense matrix-function path."""
        A = self.linear_operator.todense()
        mass_diag = self._mass_diagonal_or_identity()
        if mass_diag is not None:
            Lmat = A / mass_diag.reshape((-1,))[:, None]
        else:
            M = self.mass_operator.todense()
            Lmat = Matrix(M).solve(A)

        z = dt * Lmat
        E = expm(z)
        E2 = expm(z / 2)
        phi1_half, _, _ = _phi_matrices(z / 2)
        Q = 0.5 * phi1_half
        phi1, phi2, phi3 = _phi_matrices(z)
        f1, f2, f3 = _etdrk4_nonlinear_weights(phi1, phi2, phi3)
        return tuple(map(Matrix, (E, E2, Q, f1, f2, f3)))

    def _N(self, u_hat: Array, N: Padding = None) -> Array:
        """Return the nonlinear-plus-forcing contribution used in ETDRK4 stages."""
        nval = (
            self.nonlinear_rhs(u_hat, N)
            if self.has_nonlinear
            else jnp.zeros_like(u_hat)
        )
        return nval + self._forcing_rhs

    @jax.jit(static_argnums=(0, 3))
    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array:
        """Advance one ETDRK4 step in coefficient space."""
        dtQ = dt * self.Q
        n1 = self._N(u_hat, N)
        a = self.E2 @ u_hat + dtQ @ n1
        n2 = self._N(a, N)
        b = self.E2 @ u_hat + dtQ @ n2
        n3 = self._N(b, N)
        c = self.E2 @ a + dtQ @ (2 * n3 - n1)
        n4 = self._N(c, N)

        return self.E @ u_hat + dt * (
            (self.f1 @ n1) + 2 * (self.f2 @ (n2 + n3)) + (self.f3 @ n4)
        )
