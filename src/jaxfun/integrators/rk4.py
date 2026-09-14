"""Classical explicit Runge-Kutta time integration."""

from jaxfun.typing import Array, ScalarPadding

from .base import BaseIntegrator


class RK4(BaseIntegrator):
    """Regular 4th-order Runge-Kutta integrator."""

    def _step_impl(
        self,
        u_hat: Array,
        dt: float,
        N: ScalarPadding = None,
        t: Array | float = 0.0,
        /,
    ) -> Array:
        """Advance one classical RK4 step in coefficient space."""
        k1 = self.total_rhs(u_hat, N, t)
        k2 = self.total_rhs(u_hat + 0.5 * dt * k1, N, t + 0.5 * dt)
        k3 = self.total_rhs(u_hat + 0.5 * dt * k2, N, t + 0.5 * dt)
        k4 = self.total_rhs(u_hat + dt * k3, N, t + dt)
        return u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
