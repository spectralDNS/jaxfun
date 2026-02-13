from typing import Any

import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.typing import Array

from .base import BaseIntegrator, _operator_to_dense


class BackwardEuler(BaseIntegrator):
    """First-order implicit Euler for linear terms (IMEX for nonlinear terms)."""

    def __init__(
        self,
        V: OrthogonalSpace,
        equation: sp.Expr,
        u0: sp.Expr | Array | None = None,
        *,
        time: tuple[float, float] | None = None,
        initial: sp.Expr | Array | None = None,
        **params: Any,
    ):
        super().__init__(
            V,
            equation,
            u0,
            time=time,
            initial=initial,
            sparse=bool(params.get("sparse", False)),
            sparse_tol=int(params.get("sparse_tol", 1000)),
        )
        self.params = dict(params)
        self._system_diag = nnx.data(None)
        self._system_matrix = nnx.data(None)

    def setup(self, dt: float) -> None:
        self._system_diag = nnx.data(None)
        self._system_matrix = nnx.data(None)
        if self.linear_operator is None:
            return

        if self.mass_diag is not None and self.linear_diag is not None:
            self._system_diag = nnx.data(self.mass_diag - dt * self.linear_diag)
            return

        mass_mat = (
            _operator_to_dense(self.mass_operator)
            if self.mass_operator is not None
            else jnp.diag(self.mass_diag.reshape((-1,)))
        )
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
