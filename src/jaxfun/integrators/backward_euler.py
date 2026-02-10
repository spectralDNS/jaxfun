from __future__ import annotations

from typing import Any

from .base import BaseIntegrator


class BackwardEuler(BaseIntegrator):
    """Backward Euler integrator (skeleton).

    This will typically require solving a linear (or nonlinear) system each step.
    Currently only the symbolic preprocessing is implemented.
    """

    name = "BackwardEuler"

    def _prepare(self) -> None:
        raise NotImplementedError("BackwardEuler assembly not implemented yet")

    def initial_state(self) -> Any:
        raise NotImplementedError(
            "BackwardEuler state initialization not implemented yet"
        )

    def step(self, state: Any, t: float, dt: float) -> Any:
        raise NotImplementedError("BackwardEuler stepping not implemented yet")
