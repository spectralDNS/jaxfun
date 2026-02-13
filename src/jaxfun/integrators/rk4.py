from typing import Any

import sympy as sp
from flax import nnx

from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.typing import Array

from .base import BaseIntegrator


class RK4(BaseIntegrator):
    """Regular 4th-order Runge-Kutta integrator."""

    def __init__(
        self,
        V: OrthogonalSpace,
        equation: sp.Expr,
        u0: sp.Expr | Array | None = None,
        *,
        time: tuple[float, float] | None = None,
        initial: sp.Expr | Array | None = None,
        update: Any | None = None,
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
        self.update_fn = update
        self._dt: float | None = None

    @nnx.jit
    def step(self, u_hat: Array, dt: float) -> Array:
        k1 = self.total_rhs(u_hat)
        k2 = self.total_rhs(u_hat + 0.5 * dt * k1)
        k3 = self.total_rhs(u_hat + 0.5 * dt * k2)
        k4 = self.total_rhs(u_hat + dt * k3)
        return u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
