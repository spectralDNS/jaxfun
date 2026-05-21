"""Backward Euler integrator for semi-discrete Galerkin systems."""

import jax.numpy as jnp
from flax import nnx

from jaxfun.la import Matrix
from jaxfun.typing import Array, Padding
from jaxfun.utils.operator_tools import LinearTerm

from .base import BaseIntegrator


class BackwardEuler(BaseIntegrator):
    """First-order implicit Euler for linear terms (IMEX for nonlinear terms)."""

    _system_diag: Array | None = None
    _system_matrix: Matrix | None = None
    _system_term: LinearTerm | None = None

    def setup(self, dt: float) -> None:
        """Precompute the implicit system matrix for the given step size."""
        self._system_diag = None
        self._system_matrix = None
        self._system_term = None

        if self.linear_operator is None:
            return

        mass_diag = self.mass_term.diagonal_or_none()
        linear_diag = self.linear_term.diagonal_or_none()
        if linear_diag is not None:
            if mass_diag is None and self.mass_term.operator is None:
                mass_diag = jnp.ones_like(linear_diag)
            if mass_diag is not None:
                self._system_diag = nnx.data(mass_diag - dt * linear_diag)
                self._system_term = LinearTerm(diagonal=self._system_diag)
                return

        mass_mat = self.mass_term.todense(self._state_size, identity_if_zero=True)
        linear_mat = self.linear_term.todense(self._state_size)
        self._system_matrix = nnx.data(Matrix(mass_mat - dt * linear_mat))
        # Warm the cached factorization outside the timestep loop.
        self._system_matrix.lu_factor()
        self._system_term = LinearTerm(operator=self._system_matrix)

    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array:
        """Advance one backward-Euler step in coefficient space."""
        rhs = self.apply_mass(u_hat)
        if self.linear_forcing is not None:
            rhs = rhs + dt * jnp.asarray(self.linear_forcing)
        if self.has_nonlinear:
            rhs = rhs + dt * self.nonlinear_rhs_scalar_product(u_hat, N)

        if self._system_term is not None:
            return self._system_term.solve(rhs)
        return self.apply_mass_inverse(rhs)
