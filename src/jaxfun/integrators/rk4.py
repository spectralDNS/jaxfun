from __future__ import annotations

from typing import Any

from .base import BaseIntegrator


class RK4(BaseIntegrator):
    """Classical explicit 4th-order Runge–Kutta integrator (skeleton).

    Numerical assembly (mass matrix inversion, RHS evaluation, etc.) is not
    implemented yet; this class currently provides the common interface and
    symbolic preprocessing via `BaseIntegrator`.
    """

    name = "RK4"

    def _prepare(self) -> None:
        raise NotImplementedError("RK4 assembly not implemented yet")

    def initial_state(self) -> Any:
        raise NotImplementedError("RK4 state initialization not implemented yet")

    def step(self, state: Any, t: float, dt: float) -> Any:
        raise NotImplementedError("RK4 stepping not implemented yet")
