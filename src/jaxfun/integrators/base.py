from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import sympy as sp

from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.typing import FunctionSpaceType
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms


@dataclass(frozen=True, slots=True)
class SplitWeakForm:
    """Preprocessed weak form split into time / linear / nonlinear pieces."""

    time_symbol: sp.Symbol
    trial: sp.Expr
    test: sp.Expr | None
    time_terms: sp.Expr
    rhs: sp.Expr
    rhs_linear: sp.Expr
    rhs_nonlinear: sp.Expr
    time_order: int


def _time_derivative_order(derivative: sp.Derivative, time_symbol: sp.Symbol) -> int:
    return sum(1 for v in derivative.variables if v == time_symbol)


class BaseIntegrator[T: FunctionSpaceType](ABC):
    """Base class for time integrators over a Galerkin-discretized PDE.

    This is a skeleton that focuses on *symbolic preprocessing*:

    - identify time-derivative terms using `split_time_derivative_terms`
    - classify RHS into linear/nonlinear parts using `split_linear_nonlinear_terms`
    - record time derivative order (e.g. first-order vs second-order)

    Subclasses are expected to implement numerical assembly and stepping.

    Args:
        space: Function space describing the spatial discretization.
        weak_form: SymPy expression representing the weak form (usually contains
            TestFunction * (PDE residual)).
        time: `(t0, t1)` interval.
        initial: Initial condition in physical space (symbolic expression).
    """

    name: str = "BaseIntegrator"

    def __init__(
        self,
        space: T,
        weak_form: sp.Expr,
        *,
        time: tuple[float, float],
        initial: sp.Expr,
    ) -> None:
        self.space = space
        self.weak_form = sp.sympify(weak_form)
        self.time = time
        self.initial = sp.sympify(initial)

        t = self.space.system.base_time()
        time_terms, rhs = split_time_derivative_terms(self.weak_form, t)
        test, trial = get_basisfunctions(rhs)
        self.trial: TrialFunction = trial
        self.test: TestFunction | None = test

        rhs_linear, rhs_nonlinear = split_linear_nonlinear_terms(rhs, self.trial)
        orders = [
            _time_derivative_order(d, t)
            for d in time_terms.atoms(sp.Derivative)
            if (t in d.variables) and isinstance(d.expr, TrialFunction)
        ]
        time_order = max(orders) if orders else 0

        self.split = SplitWeakForm(
            time_symbol=t,
            trial=self.trial,
            test=self.test,
            time_terms=time_terms,
            rhs=rhs,
            rhs_linear=rhs_linear,
            rhs_nonlinear=rhs_nonlinear,
            time_order=time_order,
        )

        self._prepared = False

    def prepare(self) -> None:
        """Assemble discrete operators (mass matrix, linear operator, etc.).

        Skeleton: subclasses should override `_prepare`.
        """
        if self._prepared:
            return
        self._prepare()
        self._prepared = True

    @abstractmethod
    def _prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: Any, t: float, dt: float) -> Any:
        raise NotImplementedError

    def solve(self, *, dt: float, steps: int) -> Any:
        """Run the integrator for a fixed number of steps.

        Skeleton: requires subclasses to provide a concrete state representation.
        """
        self.prepare()
        state = self.initial_state()
        t = float(self.time[0])
        for _ in range(steps):
            state = self.step(state, t, dt)
            t += dt
        return state

    @abstractmethod
    def initial_state(self) -> Any:
        raise NotImplementedError
