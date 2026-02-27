from flax import nnx

from jaxfun.typing import Array

from .base import BaseIntegrator


class RK4(BaseIntegrator):
    """Regular 4th-order Runge-Kutta integrator."""

    @nnx.jit
    def step(self, u_hat: Array, dt: float) -> Array:
        k1 = self.total_rhs(u_hat)
        k2 = self.total_rhs(u_hat + 0.5 * dt * k1)
        k3 = self.total_rhs(u_hat + 0.5 * dt * k2)
        k4 = self.total_rhs(u_hat + dt * k3)
        return u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
