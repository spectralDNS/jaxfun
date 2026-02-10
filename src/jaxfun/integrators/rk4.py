from __future__ import annotations

from typing import Any

import jax

from .base import BaseIntegrator


class RK4(BaseIntegrator):
    """Classical explicit 4th-order Runge–Kutta integrator (skeleton).

    Numerical assembly (mass matrix inversion, RHS evaluation, etc.) is not
    implemented yet; this class currently provides the common interface and
    symbolic preprocessing via `BaseIntegrator`.
    """

    name = "RK4"

    @jax.jit(static_argnums=0)
    def step(self, state: Any, t: float, dt: float) -> Any:
        # Only linear time stepping is implemented for now.
        if self.nonlinear_form != 0:
            raise NotImplementedError("RK4 nonlinear stepping not implemented yet")

        u = state
        k1 = self.linear_rhs(u)
        k2 = self.linear_rhs(u + 0.5 * dt * k1)
        k3 = self.linear_rhs(u + 0.5 * dt * k2)
        k4 = self.linear_rhs(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
