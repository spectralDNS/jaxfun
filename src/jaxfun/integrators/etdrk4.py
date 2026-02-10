from __future__ import annotations

from typing import Any

from .base import BaseIntegrator


class ETDRK4(BaseIntegrator):
    """Exponential time-differencing RK4 integrator (skeleton).

    Intended target structure after Galerkin discretization:

        M * u_t = L * u + N(u)

    where `L` is linear and `N` nonlinear. This class currently only performs
    symbolic preprocessing (time-term and linear/nonlinear splitting).
    """

    name = "ETDRK4"

    def _prepare(self) -> None:
        raise NotImplementedError("ETDRK4 assembly not implemented yet")

    def initial_state(self) -> Any:
        raise NotImplementedError("ETDRK4 state initialization not implemented yet")

    def step(self, state: Any, t: float, dt: float) -> Any:
        raise NotImplementedError("ETDRK4 stepping not implemented yet")
