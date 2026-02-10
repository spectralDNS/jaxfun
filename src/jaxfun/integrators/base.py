from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin import (
    TensorProductSpace,
    TestFunction,
    TrialFunction,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import inner, project, project1D
from jaxfun.typing import Array, FunctionSpaceType
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms
from jaxfun.utils.sympy_factoring import get_time_independent


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


class BaseIntegrator(ABC):
    """Base class for time integrators for PDEs discretized in *space*.

    We use the Galerkin machinery only to discretize the spatial operators.
    Time is *not* treated with Galerkin; instead we discretize time using an
    integrator (RK4/ETD/implicit/etc.) on the resulting semi-discrete ODE system.

    This skeleton focuses on symbolic preprocessing:

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
        space: FunctionSpaceType,
        weak_form: sp.Expr,
        *,
        time: tuple[float, float],
        initial: sp.Expr,
        sparse: bool = False,
        sparse_tol: int = 1000,
    ) -> None:
        self.space: FunctionSpaceType = space
        self.weak_form = sp.sympify(weak_form)
        self.time = time
        self.initial = sp.sympify(initial)
        self.sparse = sparse
        self.sparse_tol = sparse_tol

        t = self.space.system.base_time()
        time_terms, rhs = split_time_derivative_terms(self.weak_form, t)

        # After splitting, the RHS should not contain any time derivatives.
        # (It may still contain explicit symbols like forcing f(t) if the user
        # includes them, but no d/dt operators should remain.)
        assert not any(t in d.variables for d in rhs.atoms(sp.Derivative)), (
            "RHS unexpectedly contains time derivatives"
        )

        test, trial = get_basisfunctions(self.weak_form)
        assert isinstance(trial, TrialFunction), "No TrialFunction found in weak form"
        self.trial: TrialFunction = trial
        self.test: TestFunction | None = (
            test if isinstance(test, TestFunction) else None
        )

        # RHS must be time-independent with respect to the trial/test functions.
        # The splitting utilities are responsible for dropping any transient time
        # argument from TrialFunction instances on the RHS.
        assert all(not f.transient for f in rhs.atoms(TrialFunction)), (
            "RHS unexpectedly contains transient TrialFunction instances"
        )

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

        # These are populated by `prepare()`.
        self.mass_matrix: Any | None = None
        self.linear_operator: Any | None = None
        self.linear_forcing: Any | None = None
        self.nonlinear_form: sp.Expr = -self.split.rhs_nonlinear

    def prepare(self) -> None:
        """Assemble discrete operators (mass matrix, linear operator, etc.)."""
        if self._prepared:
            return
        self._prepare()
        self._prepared = True

        # Cached dense versions (used by time steppers that rely on dense solves).
        self._mass_dense: Array | None = None
        self._linear_dense: Array | None = None

    def _prepare(self) -> None:
        """Default preparation: assemble linear forms using Galerkin `inner`.

        This assembles only what can be assembled automatically:

        - Mass matrix from the time-derivative term(s)
        - Linear operator and forcing from the linear RHS

        Nonlinear terms remain symbolic and are intended to be handled later via
        lambdify/evaluation on physical space samples.
        """
        if self.split.time_order != 1:
            raise NotImplementedError(
                "Linear form assembly currently supports only first-order time "
                f"derivatives; got order={self.split.time_order}."
            )

        t = self.split.time_symbol
        time_terms = sp.sympify(self.split.time_terms)
        repl: dict[sp.Expr, sp.Expr] = {}
        for d in time_terms.atoms(sp.Derivative):
            if (t in d.variables) and isinstance(d.expr, TrialFunction):
                u_ind = get_time_independent(d.expr)
                repl[d] = u_ind

        # Assemble the spatial mass matrix M from the time term by replacing
        # d/dt u with u (time discretization happens in the integrator, not here).
        mass_form = time_terms.xreplace(repl)
        self.mass_matrix = inner(
            mass_form,
            sparse=self.sparse,
            sparse_tol=self.sparse_tol,
        )

        rhs_linear = sp.sympify(self.split.rhs_linear)
        if rhs_linear == 0:
            self.linear_operator = None
            self.linear_forcing = None
            return

        linear_form = inner(
            -rhs_linear,
            sparse=self.sparse,
            sparse_tol=self.sparse_tol,
        )

        if isinstance(linear_form, tuple) and len(linear_form) == 2:
            self.linear_operator, self.linear_forcing = linear_form
        else:
            _, trial = get_basisfunctions(rhs_linear)
            if trial is None:
                self.linear_operator = None
                self.linear_forcing = linear_form
            else:
                # inner returns matrix (bilinear only) or vector (linear only)
                self.linear_operator = linear_form
                self.linear_forcing = None

    def _dense_linear_operators(self) -> tuple[Array, Array, Array | None]:
        """Return (M, L, b) as dense JAX arrays (L defaults to zeros)."""
        self.prepare()

        if self.mass_matrix is None:
            raise RuntimeError("Integrator not prepared: missing mass matrix")

        if self._mass_dense is None:
            M = self.mass_matrix
            M = M.todense() if hasattr(M, "todense") else M
            self._mass_dense = jnp.asarray(M)

        if self._linear_dense is None:
            if self.linear_operator is None:
                self._linear_dense = jnp.zeros_like(self._mass_dense)
            else:
                L = self.linear_operator
                L = L.todense() if hasattr(L, "todense") else L
                self._linear_dense = jnp.asarray(L)

        b = self.linear_forcing
        b = None if b is None else jnp.asarray(b)

        return self._mass_dense, self._linear_dense, b

    @jax.jit(static_argnums=0)
    def linear_rhs(self, state: Array) -> Array:
        """Compute u_t = M^{-1}(L u + b) for the assembled linear system."""
        M, L, b = self._dense_linear_operators()
        rhs = L @ state
        if b is not None:
            rhs = rhs + b
        return jnp.linalg.solve(M, rhs)

    @abstractmethod
    def step(self, state: Array, t: float, dt: float) -> Array:
        raise NotImplementedError

    def solve(self, *, dt: float, steps: int) -> Any:
        """Run the integrator for a fixed number of steps.

        Skeleton: requires subclasses to provide a concrete state representation.
        """
        self.prepare()
        state0 = self.initial_state()
        t0 = float(self.time[0])

        def body(
            carry: tuple[Array, float],
            _unused: Any,
        ) -> tuple[tuple[Array, float], None]:
            state, t = carry
            state = self.step(state, t, dt)
            return (state, t + dt), None

        (stateT, _tT), _ = jax.lax.scan(body, (state0, t0), None, length=steps)
        return stateT

    def initial_state(self) -> Array:
        # Coefficient-space initial state via Galerkin projection.
        if isinstance(self.space, TensorProductSpace | VectorTensorProductSpace):
            return project(self.initial, self.space)  # ty:ignore[invalid-argument-type]
        return project1D(self.initial, self.space)
