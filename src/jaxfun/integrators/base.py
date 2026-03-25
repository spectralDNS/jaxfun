"""Base abstractions for time integrators on Galerkin coefficient spaces."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import prod
from typing import cast

import jax
import jax.numpy as jnp
import sympy as sp
import tqdm
from flax import nnx
from sympy.core.function import AppliedUndef

from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import project
from jaxfun.typing import Array, FunctionSpaceType, GalerkinOperatorLike, Padding
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms
from jaxfun.utils.operator_tools import (
    apply_operator,
    assemble_linear_term,
    operator_to_dense,
    solve_operator,
)
from jaxfun.utils.sympy_factoring import time_derivative_as_operator

from .nonlinear import (
    compile_nonlinear_evaluator,
    remove_test_function,
    replace_trial_with_jaxfunction,
)


class BaseIntegrator(ABC, nnx.Module):
    """Base class for time integration of semi-discrete Galerkin systems.

    The input weak form is split into a time-derivative operator, linear
    right-hand-side terms, and nonlinear terms. The latter are compiled into a
    cached physical-space evaluator, while the linear and mass terms are
    assembled once in coefficient space.
    """

    def __init__(
        self,
        V: FunctionSpaceType,
        equation: sp.Expr,
        *,
        initial: sp.Expr | Array,
        time: tuple[float, float] | None = None,
        sparse: bool = False,
        sparse_tol: int = 1000,
    ):
        """Build an integrator from a weak form and an initial condition.

        Args:
            V: Function space used for the semi-discrete state.
            equation: Weak-form expression containing a first-order time
                derivative of a transient TrialFunction.
            initial: Initial condition, either symbolically in physical space or
                directly as coefficients.
            time: Optional default integration interval.
            sparse: Assemble sparse operators when possible.
            sparse_tol: Sparsification tolerance passed to Galerkin assembly.
        """
        if initial is None:
            raise ValueError("Initial condition must be provided via `initial`")

        self.sparse = sparse
        self.sparse_tol = sparse_tol
        self.time = time
        self.initial_condition = initial
        self.functionspace = V
        self._state_shape = self._coefficient_shape(V)
        self._state_size = int(prod(self._state_shape))

        trial, mass_expr, linear_expr, nonlinear_expr = self._extract_equation_terms(
            equation
        )
        self.mass_expr = mass_expr
        self.linear_expr = linear_expr
        self.nonlinear_expr = nonlinear_expr
        self.has_nonlinear = bool(sp.sympify(nonlinear_expr) != 0)

        mass_term = assemble_linear_term(
            self.mass_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        if mass_term.forcing is not None:
            raise ValueError("Time-derivative operator assembly produced forcing")
        self.mass_operator: GalerkinOperatorLike | None = nnx.data(mass_term.operator)
        self.mass_diag: Array | None = nnx.data(mass_term.diagonal)

        linear_term = assemble_linear_term(
            self.linear_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        self.linear_operator: GalerkinOperatorLike | None = nnx.data(
            linear_term.operator
        )
        self.linear_forcing: Array | None = nnx.data(linear_term.forcing)
        self.linear_diag: Array | None = nnx.data(linear_term.diagonal)

        self._nonlinear_jaxfunction: AppliedUndef | None = None
        self._nonlinear_evaluator: Callable[[Array, Padding], Array] | None = None
        if self.has_nonlinear:
            self._setup_nonlinear_evaluator(trial)

    @staticmethod
    def _coefficient_shape(V: FunctionSpaceType) -> tuple[int, ...]:
        """Return the coefficient-array shape for the given space."""
        num_dofs = V.num_dofs
        return num_dofs if isinstance(num_dofs, tuple) else (num_dofs,)

    def _extract_equation_terms(
        self, equation: sp.Expr
    ) -> tuple[TrialFunction, sp.Expr, sp.Expr, sp.Expr]:
        """Split a weak form into mass, linear, and nonlinear components."""
        t = self.functionspace.system.base_time()
        lhs, rhs = split_time_derivative_terms(equation, t)
        if sp.sympify(lhs) == 0:
            raise ValueError(
                "Time integrators require a first-order time derivative "
                "in the weak form"
            )

        lhs_test, lhs_trial = get_basisfunctions(lhs)
        assert isinstance(lhs_test, TestFunction), (
            "Currently only supports one TestFunction in weak form"
        )
        assert isinstance(lhs_trial, TrialFunction), (
            "Currently only supports one TrialFunction in weak form"
        )

        mass_expr = time_derivative_as_operator(lhs, lhs_trial, t)

        basis_expr = rhs if sp.sympify(rhs) != 0 else mass_expr
        test, trial = get_basisfunctions(basis_expr)
        assert isinstance(test, TestFunction), (
            "Currently only supports one TestFunction in weak form"
        )
        assert isinstance(trial, TrialFunction), (
            "Currently only supports one TrialFunction in weak form"
        )

        linear, nonlinear = split_linear_nonlinear_terms(-rhs, trial)
        nonlinear = sp.expand(remove_test_function(nonlinear, test))
        return trial, mass_expr, linear, nonlinear

    def _setup_nonlinear_evaluator(self, trial: TrialFunction) -> None:
        """Compile the nonlinear physical-space evaluator for the trial field."""
        base_jaxfunction = JAXFunction(
            jnp.zeros(self._state_shape), self.functionspace, name=f"{trial.name}_jax"
        )
        jaxfunction = cast(AppliedUndef, base_jaxfunction.doit())
        nonlinear_expr = replace_trial_with_jaxfunction(
            self.nonlinear_expr, trial, jaxfunction
        )
        self.nonlinear_expr = nonlinear_expr
        self._nonlinear_jaxfunction = jaxfunction
        self._nonlinear_evaluator = compile_nonlinear_evaluator(
            nonlinear_expr, self.functionspace, jaxfunction
        )

    def _dense_matrix(
        self,
        operator: GalerkinOperatorLike | None,
        diagonal: Array | None,
    ) -> Array:
        """Return a dense matrix representation of a linear operator."""
        if diagonal is not None:
            return jnp.diag(diagonal.reshape((-1,)))
        if operator is None:
            return jnp.zeros((self._state_size, self._state_size))
        return operator_to_dense(operator)

    def mass_matrix_dense(self) -> Array:
        """Return the assembled mass operator as a dense matrix."""
        return self._dense_matrix(self.mass_operator, self.mass_diag)

    def linear_matrix_dense(self) -> Array:
        """Return the assembled linear operator as a dense matrix."""
        return self._dense_matrix(self.linear_operator, self.linear_diag)

    def initial_coefficients(self, initial: sp.Expr | Array | None = None) -> Array:
        """Return coefficient-space data for an initial condition."""
        init = self.initial_condition if initial is None else initial
        if isinstance(init, sp.Expr):
            return project(init, self.functionspace)
        return jnp.asarray(init).reshape(self.functionspace.num_dofs)

    def resolve_time(
        self,
        dt: float,
        steps: int | None = None,
        trange: tuple[float, float] | None = None,
    ) -> tuple[float, float, int]:
        """Resolve the effective time interval and number of time steps."""
        interval = self.time if trange is None else trange
        if interval is None:
            if steps is None:
                raise ValueError("Either `steps` or `trange`/`time` must be provided")
            return 0.0, float(dt * steps), int(steps)

        t0, t1 = float(interval[0]), float(interval[1])
        if steps is None:
            span = t1 - t0
            steps = int(round(span / dt))
        return t0, t1, int(steps)

    def apply_mass(self, uh: Array) -> Array:
        """Apply the assembled mass operator to a coefficient state."""
        if self.mass_diag is not None:
            return self.mass_diag * uh
        if self.mass_operator is None:
            return uh
        return apply_operator(self.mass_operator, uh)

    def apply_mass_inverse(self, rhs: Array) -> Array:
        """Apply the inverse mass operator to a coefficient-space right-hand side."""
        if self.mass_diag is not None:
            return rhs / self.mass_diag
        if self.mass_operator is None:
            return rhs
        return solve_operator(self.mass_operator, rhs)

    def nonlinear_rhs(self, uh: Array, N: Padding = None) -> Array:
        """Return the nonlinear contribution in coefficient space."""
        if not self.has_nonlinear:
            return jnp.zeros_like(uh)
        assert self._nonlinear_evaluator is not None
        return self.functionspace.forward(self._nonlinear_evaluator(uh, N))

    def nonlinear_rhs_scalar_product(self, uh: Array, N: Padding = None) -> Array:
        """Return the nonlinear contribution in coefficient space.

        Do *not* apply the mass inverse to complete the forward transformation,
        because the mass inverse may be required elsewhere.
        """
        if not self.has_nonlinear:
            return jnp.zeros_like(uh)
        assert self._nonlinear_evaluator is not None
        return self.functionspace.scalar_product(self._nonlinear_evaluator(uh, N))

    def linear_rhs(self, uh: Array) -> Array:
        """Return the linear contribution after applying the inverse mass matrix."""
        rhs = jnp.zeros_like(uh)
        if self.linear_operator is not None:
            if self.linear_diag is not None:
                rhs = rhs + self.linear_diag * uh
            else:
                rhs = rhs + apply_operator(self.linear_operator, uh)
        if self.linear_forcing is not None:
            rhs = rhs + jnp.asarray(self.linear_forcing)
        return self.apply_mass_inverse(rhs)

    @jax.jit(static_argnums=(0, 2))
    def total_rhs(self, uh: Array, N: Padding = None) -> Array:
        """Return the full semi-discrete right-hand side."""
        return self.linear_rhs(uh) + self.nonlinear_rhs(uh, N)

    @abstractmethod
    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array: ...

    def setup(self, dt: float) -> None:
        """Precompute step-size-dependent coefficients before time stepping."""
        ...

    def solve(
        self,
        dt: float,
        steps: int | None = None,
        state0: Array | None = None,
        trange: tuple[float, float] | None = None,
        N: int | tuple[int | None, ...] | None = None,
        progress: bool = True,
        n_batches: int = 100,
        return_batch_snapshots: bool = False,
    ) -> Array:
        """Advance the coefficient state in time.

        Args:
            dt: Time-step size.
            steps: Number of steps to take.
            state0: Optional coefficient-space restart state. If omitted, use the
                projected constructor `initial`.
            trange: Optional `(t0, t1)` override for `self.time`.
            N: Optional physical-space padding passed through to nonlinear
                backward evaluations.
            progress: Show a progress bar when True.
            n_batches: Number of batched integration chunks.
            return_batch_snapshots: When True, return the initial state plus one
                state per completed batch/remainder chunk instead of only the final
                state.
        """
        if n_batches <= 0:
            raise ValueError("n_batches must be a positive integer")

        self.setup(dt)
        _, _, n_steps = self.resolve_time(dt, steps=steps, trange=trange)

        if state0 is None:
            u_hat = self.initial_coefficients()
        else:
            u_hat = jnp.asarray(state0).reshape(self.functionspace.num_dofs)
        if n_steps <= 0:
            if not return_batch_snapshots:
                return u_hat
            return jnp.expand_dims(u_hat, axis=0)

        def inner_n_steps(i: int, u_hat: Array) -> Array:
            u_hat = self.step(u_hat, dt, N)
            return u_hat

        batch_count = min(n_batches, n_steps)

        batch_len = n_steps // batch_count
        remainder = n_steps - batch_count * batch_len
        states: list[Array] = [u_hat]
        diverged = False

        r_batch = range(batch_count)
        iterator = (
            tqdm.tqdm(r_batch, desc="Integrating", unit="step", unit_scale=batch_len)
            if progress
            else r_batch
        )
        for _ in iterator:
            u_hat = jax.lax.fori_loop(0, batch_len, inner_n_steps, u_hat)
            if return_batch_snapshots:
                states.append(u_hat)
            if jnp.isnan(u_hat).any() or jnp.isinf(u_hat).any():
                diverged = True
                break

        if remainder and not diverged:
            u_hat = jax.lax.fori_loop(0, remainder, inner_n_steps, u_hat)
            if return_batch_snapshots:
                states.append(u_hat)

        if return_batch_snapshots:
            return jnp.stack(states)

        return u_hat
