"""Backward Euler integrator for semi-discrete Galerkin systems."""

import jax.numpy as jnp
from flax import nnx

from jaxfun.typing import Array, Padding
from jaxfun.utils.operator_tools import (
    operator_diagonal,
    operator_is_zero,
    solve_operator,
    warm_operator_solve_cache,
)

from .base import BaseIntegrator


class BackwardEuler(BaseIntegrator):
    """First-order implicit Euler for linear terms (IMEX for nonlinear terms)."""

    _system_diag: Array | None = None
    _system_operator = None

    def setup(self, dt: float) -> None:
        """Precompute the implicit system matrix for the given step size."""
        self._system_diag = None
        self._system_operator = None

        if operator_is_zero(self.linear_operator):
            return

        self._system_operator = nnx.data(self.mass_operator - dt * self.linear_operator)
        self._system_diag = nnx.data(operator_diagonal(self._system_operator))
        warm_operator_solve_cache(self._system_operator)

    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array:
        """Advance one backward-Euler step in coefficient space."""
        rhs = self.apply_mass(u_hat)
        if self.linear_forcing is not None:
            rhs = rhs + dt * jnp.asarray(self.linear_forcing)
        if self.has_nonlinear:
            rhs = rhs + dt * self.nonlinear_rhs_scalar_product(u_hat, N)

        if self._system_operator is not None:
            return solve_operator(self._system_operator, rhs)
        return self.apply_mass_inverse(rhs)
