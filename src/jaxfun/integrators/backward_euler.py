import jax.numpy as jnp
from flax import nnx

from jaxfun.typing import Array

from .base import BaseIntegrator, _operator_to_dense


class BackwardEuler(BaseIntegrator):
    """First-order implicit Euler for linear terms (IMEX for nonlinear terms)."""

    _system_diag: Array | None = nnx.data(None)
    _system_matrix: Array | None = nnx.data(None)

    def setup(self, dt: float) -> None:
        if self.linear_operator is None:
            return

        if self.mass_diag is not None and self.linear_diag is not None:
            self._system_diag = nnx.data(self.mass_diag - dt * self.linear_diag)
            return

        if self.mass_operator is not None:
            mass_mat = _operator_to_dense(self.mass_operator)
        else:
            assert self.mass_diag is not None
            mass_mat = jnp.diag(self.mass_diag.reshape((-1,)))
        linear_mat = _operator_to_dense(self.linear_operator)
        self._system_matrix = nnx.data(mass_mat - dt * linear_mat)

    def step(self, u_hat: Array, dt: float) -> Array:
        rhs = self.apply_mass(u_hat)
        if self.linear_forcing is not None:
            rhs = rhs + dt * jnp.asarray(self.linear_forcing)
        if self.has_nonlinear:
            rhs = rhs + dt * self.apply_mass(self.nonlinear_rhs(u_hat))

        if self._system_diag is not None:
            return rhs / self._system_diag
        if self._system_matrix is not None:
            return jnp.linalg.solve(self._system_matrix, rhs.reshape((-1,))).reshape(
                rhs.shape
            )
        return self.apply_mass_inverse(rhs)
