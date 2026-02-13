from typing import Any

import sympy as sp
from flax import nnx

from jaxfun.galerkin import Composite, DirectSum
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.typing import Array

from .base import BaseIntegrator


class RK4(BaseIntegrator):
    """Regular 4th-order Runge-Kutta integrator."""

    def __init__(
        self,
        V: OrthogonalSpace | Composite | DirectSum,
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
            **params,
        )

    @nnx.jit
    def step(self, u_hat: Array, dt: float) -> Array:
        k1 = self.total_rhs(u_hat)
        k2 = self.total_rhs(u_hat + 0.5 * dt * k1)
        k3 = self.total_rhs(u_hat + 0.5 * dt * k2)
        k4 = self.total_rhs(u_hat + dt * k3)
        return u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
