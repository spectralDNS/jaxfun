"""Base abstractions for time integrators on Galerkin coefficient spaces."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import sympy as sp
import tqdm
from flax import nnx
from sympy.core.function import AppliedUndef

from jaxfun.coordinates import get_system
from jaxfun.galerkin import TensorProductSpace, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import project
from jaxfun.la import BaseMatrix, IdentityMatrix, ZeroMatrix
from jaxfun.la.matrixprotocol import SolverNotApplicable
from jaxfun.typing import Array, IntegratorState, ScalarPadding, ScalarSpaceType
from jaxfun.utils import (
    get_time_independent,
    normalize_explicit,
    split_linear_nonlinear_terms,
    split_time_derivative_terms,
)
from jaxfun.utils.operator_tools import assemble_linear_term
from jaxfun.utils.sympy_factoring import time_derivative_as_operator

from .nonlinear import (
    compile_nonlinear_evaluator,
    remove_test_function,
    replace_trial_with_jaxfunction,
)

type SolverOptions = tuple[tuple[str, Any], ...]


def _accepted_solve_options(cls: type) -> frozenset[str]:
    """Return the keyword-only options ``cls.solve`` declares.

    Operators differ in what they can be tuned with -- `TPMatrices.solve` picks
    between factored and Kronecker paths, `DiaMatrix.solve` between banded and
    dense, and `Matrix.solve` has nothing to pick -- so each one is offered only
    the options it names.
    """
    solve = getattr(cls, "solve", None)
    if solve is None:
        return frozenset()
    try:
        params = inspect.signature(solve).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtin/C signatures
        return frozenset()
    return frozenset(
        name
        for name, param in params.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )


def known_solve_options() -> frozenset[str]:
    """Return every solve option any `BaseMatrix` in `jaxfun.la` accepts."""
    names: set[str] = set()
    pending: list[type] = [BaseMatrix]
    seen: set[type] = set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        pending.extend(cls.__subclasses__())
        names |= _accepted_solve_options(cls)
    return frozenset(names)


def _validate_solver_options(options: Mapping[str, Any] | None) -> SolverOptions:
    """Normalize user solver options into a hashable, order-independent tuple.

    A mapping cannot be stored as `nnx.static` state: static fields are compared
    (and hashed) on every jit cache lookup. Sorted pairs also make two integrators
    configured the same way compare equal regardless of how the mapping was
    written.

    Raises:
        ValueError: if an option is one no operator in `jaxfun.la` accepts. The
            options are filtered per operator when used, so an unrecognized name
            would otherwise be silently dropped everywhere.
    """
    if not options:
        return ()
    known = known_solve_options()
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(
            f"Unknown solver option(s): {', '.join(unknown)}. "
            f"Accepted options are: {', '.join(sorted(known))}."
        )
    return tuple(sorted(options.items()))


def _solve(op: BaseMatrix, rhs: Array, options: SolverOptions = ()) -> Array:
    """Solve ``op x = rhs``, passing on the options `op` knows what to do with."""
    if not options:
        return op.solve(rhs)
    accepted = _accepted_solve_options(type(op))
    return op.solve(rhs, **{k: v for k, v in options if k in accepted})


def _warm_operator_solve_cache(
    op: BaseMatrix,
    shape: tuple[int, ...] | None = None,
    options: SolverOptions = (),
) -> None:
    """Warm native solver caches for operators that support factorization.

    Everything a solver decides by inspecting matrix *values* -- which
    factorization applies, a sparsity reordering, the factors themselves -- has
    to happen here, while the matrices are still concrete. Inside the jitted
    step there is no second chance: the arrays are tracers by then.

    Args:
        op: Operator whose solve caches should be populated.
        shape: Coefficient shape of the right-hand sides `op` will be solving.
            When given, an operator with no applicable factored solver is warmed
            by running one throwaway solve, which fills the caches of whichever
            fallback path `solve` settles on.
        options: Solver options, passed on so that the warming solve takes the
            same path the stepping solves will.
    """
    lu_factor = getattr(op, "lu_factor", None)
    if lu_factor is not None:
        try:
            lu_factor()
            return
        except (SolverNotApplicable, ValueError, TypeError, RuntimeError):
            pass
    if shape is None:
        return
    try:
        _solve(op, jnp.zeros(shape), options)
    except (SolverNotApplicable, ValueError, TypeError, RuntimeError):
        return


def _boundary_values(space) -> list[sp.Expr]:
    """Return the boundary values of every BC-carrying factor of ``space``.

    A `Composite`/`DirectSum` holds its own boundary conditions, while a tensor
    product space delegates to its 1D factors.
    """
    bcs = getattr(space, "bcs", None)
    if bcs is not None:
        return [sp.sympify(val) for val in bcs.orderedvals()]
    if isinstance(space, TensorProductSpace):
        return [val for factor in space for val in _boundary_values(factor)]
    return []


class TimeStepper[StateT: IntegratorState](ABC, nnx.Module):
    """Step-loop driver shared by scalar and system integrators.

    The coefficient state is any pytree of arrays: a single array for a scalar
    equation, or one array per field for a system of coupled equations.
    Subclasses provide `step` and the initial state; everything below is
    agnostic to which of the two it is.
    """

    time: tuple[float, float] | None

    @abstractmethod
    def step(self, u_hat: StateT, dt: float, N: ScalarPadding = None) -> StateT: ...

    @abstractmethod
    def initial_coefficients(self, initial=None) -> StateT:
        """Return coefficient-space data for an initial condition.

        A scalar integrator takes one initial condition, a system integrator one
        per field.
        """
        ...

    @abstractmethod
    def _coerce_state(self, state0: StateT) -> StateT:
        """Coerce a restart state into the integrator's coefficient layout."""
        ...

    def setup(self, dt: float) -> None:
        """Precompute step-size-dependent coefficients before time stepping."""
        ...

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

    def solve(
        self,
        dt: float,
        steps: int | None = None,
        state0: StateT | None = None,
        trange: tuple[float, float] | None = None,
        N: ScalarPadding = None,
        progress: bool = True,
        n_batches: int = 100,
        return_batch_snapshots: bool = False,
    ) -> StateT:
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

        Returns:
            The final state, or the stacked snapshots when
            `return_batch_snapshots` is True. For a system of equations both are
            tuples with one entry per field.
        """
        if n_batches <= 0:
            raise ValueError("n_batches must be a positive integer")

        self.setup(dt)
        _, _, n_steps = self.resolve_time(dt, steps=steps, trange=trange)

        if state0 is None:
            u_hat = self.initial_coefficients()
        else:
            u_hat = self._coerce_state(state0)
        if n_steps <= 0:
            if not return_batch_snapshots:
                return u_hat
            return jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), u_hat)

        def inner_n_steps(i: int, u_hat: StateT) -> StateT:
            u_hat = self.step(u_hat, dt, N)
            return u_hat

        batch_count = min(n_batches, n_steps)

        batch_len = n_steps // batch_count
        remainder = n_steps - batch_count * batch_len
        states: list[StateT] = [u_hat]
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
            if any(
                bool(jnp.isnan(leaf).any() or jnp.isinf(leaf).any())
                for leaf in jax.tree.leaves(u_hat)
            ):
                diverged = True
                break

        if remainder and not diverged:
            u_hat = jax.lax.fori_loop(0, remainder, inner_n_steps, u_hat)
            if return_batch_snapshots:
                states.append(u_hat)

        if return_batch_snapshots:
            return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        return u_hat


class BaseIntegrator(TimeStepper[Array]):
    """Base class for time integration of semi-discrete Galerkin systems.

    The input weak form is split into a time-derivative operator, linear
    right-hand-side terms, and nonlinear terms. The latter are compiled into a
    cached physical-space evaluator, while the linear and mass terms are
    assembled once in coefficient space.
    """

    def __init__(
        self,
        equation: sp.Expr,
        *,
        initial: sp.Expr | Array,
        time: tuple[float, float] | None = None,
        sparse: bool = False,
        sparse_tol: int = 1000,
        explicit_trials: Sequence[TrialFunction] = (),
        fields: Sequence[tuple[sp.Expr, AppliedUndef]] = (),
        field_order: tuple[AppliedUndef, ...] | None = None,
        field_index: int = 0,
        solver_options: Mapping[str, Any] | None = None,
    ):
        """Build an integrator from a weak form and an initial condition.

        Args:
            equation: Weak-form expression containing a first-order time
                derivative of a transient TrialFunction.
            initial: Initial condition, either symbolically in physical space or
                directly as coefficients.
            time: Optional default integration interval.
            sparse: Assemble sparse operators when possible.
            sparse_tol: Sparsification tolerance passed to Galerkin assembly.
            explicit_trials: TrialFunctions of *other* equations of a coupled
                system. Terms containing any of them are lagged into the explicit
                nonlinear part instead of being assembled into an operator.
            fields: `(time-independent trial, shared JAXFunction node)` pairs, one
                per field of the system. There must be exactly one node per field,
                reused by every equation, so that a stage update is seen by all of
                them. Pairs rather than a dict because JAX sorts dictionary keys
                when flattening, and SymPy objects are not orderable.
            field_order: Global field order the compiled nonlinear evaluator
                expects its coefficient tuple in.
            field_index: This equation's slot in that global order.
            solver_options: Options forwarded to every coefficient-space linear
                solve the integrator performs -- the mass solves and, for
                implicit methods, the stage solves. Each operator is offered the
                options its own `solve` declares and ignores the rest, so one
                mapping covers a mass matrix and a stage operator that tune
                differently. See `known_solve_options()` for the accepted names;
                `auto_threshold` is the usual one to raise when a sparse
                tensor-product operator falls back to a dense solve.
        """
        if initial is None:
            raise ValueError("Initial condition must be provided via `initial`")

        self.sparse = sparse
        self.sparse_tol = sparse_tol
        self.time = time
        self.initial_condition = initial
        self._explicit_trials = nnx.static(tuple(explicit_trials))
        self._fields = nnx.static(tuple(fields))
        self._field_order = nnx.static(field_order)
        self._field_index = nnx.static(field_index)
        self._solver_options: SolverOptions = nnx.static(
            _validate_solver_options(solver_options)
        )

        test, trial, mass_expr, linear_expr, nonlinear_expr = (
            self._extract_equation_terms(equation)
        )
        self.trialspace = cast(ScalarSpaceType, trial.functionspace)
        self.testspace = cast(ScalarSpaceType, test.functionspace)
        self._state_shape = self._coefficient_shape(self.trialspace)
        self.mass_expr = mass_expr
        self.linear_expr = linear_expr
        self.nonlinear_expr = nonlinear_expr
        self.has_nonlinear = bool(sp.sympify(nonlinear_expr) != 0)

        mass_operator, mass_forcing = assemble_linear_term(
            self.mass_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        if mass_forcing is not None:
            # A space with inhomogeneous boundary conditions splits the solution
            # into a free part and a fixed boundary lifting B, so the mass term
            # assembles as `M @ u_hat + inner(v, B)`. The whole term sits under
            # d/dt, and B is constant in time, so the lifting contributes
            # nothing to d/dt(M @ u_hat + inner(v, B)) = M @ du_hat/dt and is
            # dropped here. Time-dependent boundary data would instead need
            # d/dt inner(v, B) on the right-hand side; see the guard below.
            self._check_static_boundary_data()
        if mass_operator is None:
            mass_operator = IdentityMatrix(self._state_shape)
        self.mass_operator: BaseMatrix = nnx.data(mass_operator)
        self.mass_diag: Array | None = nnx.data(self.mass_operator.diagonal_or_none())
        if self.mass_diag is None:
            _warm_operator_solve_cache(
                self.mass_operator, self._state_shape, self._solver_options
            )

        linear_operator, linear_forcing = assemble_linear_term(
            self.linear_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        if linear_operator is None:
            linear_operator = ZeroMatrix(self._state_shape)
        self.linear_operator: BaseMatrix = nnx.data(linear_operator)
        self.linear_forcing: Array | None = nnx.data(linear_forcing)
        self.linear_diag: Array | None = nnx.data(
            self.linear_operator.diagonal_or_none()
        )

        self._nonlinear_jaxfunction: AppliedUndef | None = None
        self._nonlinear_evaluator: (
            Callable[[IntegratorState, ScalarPadding], Array] | None
        ) = None
        if self.has_nonlinear:
            self._setup_nonlinear_evaluator(trial)

    def _check_static_boundary_data(self) -> None:
        """Reject boundary data that varies in time.

        The boundary lifting enters the mass term under the time derivative, and
        the integrators assume it differentiates away. That only holds for
        boundary values that are constant in time. They may vary in space: in a
        tensor-product space each 1D factor carries its own boundary values,
        which are allowed to depend on the remaining coordinates.
        """
        values = _boundary_values(self.trialspace)
        if not values:
            raise ValueError(
                "Time-derivative operator assembly produced forcing, but the "
                "trial space has no boundary conditions to explain it"
            )
        t = self.trialspace.system.base_time()
        transient = [val for val in values if val.has(t)]
        if transient:
            raise NotImplementedError(
                "Time-dependent boundary conditions are not supported by the "
                f"integrators, got {transient}"
            )

    @staticmethod
    def _coefficient_shape(V: ScalarSpaceType) -> tuple[int, ...]:
        """Return the coefficient-array shape for the given space."""
        num_dofs = V.num_dofs
        return num_dofs if isinstance(num_dofs, tuple) else (num_dofs,)

    def _physical_shape(self, N: ScalarPadding) -> int | tuple[int | None, ...]:
        if N is None:
            N = self.testspace.shape
            if self.testspace.dims == 1:
                N = N[0]
        return N

    def _extract_equation_terms(
        self, equation: sp.Expr
    ) -> tuple[TestFunction, TrialFunction, sp.Expr, sp.Expr, sp.Expr]:
        """Split a weak form into mass, linear, and nonlinear components."""
        system = get_system(equation)
        t = system.base_time()
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

        # The time-derivative term identifies this equation's own field. Every
        # other field of a coupled system is foreign and must be lagged.
        explicit = normalize_explicit(self._explicit_trials)
        mass_expr = time_derivative_as_operator(lhs, lhs_trial, t, explicit=explicit)
        trial = get_time_independent(lhs_trial)

        basis_expr = rhs if sp.sympify(rhs) != 0 else mass_expr
        rhs_test, rhs_trials = get_basisfunctions(basis_expr)
        # `get_basisfunctions` returns sets as soon as *either* kind occurs more
        # than once, so a lone test function arrives wrapped when the equation
        # holds several fields.
        tests = rhs_test if isinstance(rhs_test, set) else {rhs_test}
        if len(tests) != 1:
            raise ValueError(
                "Currently only supports one TestFunction in weak form, got "
                f"{len(tests)}"
            )
        test = tests.pop()
        assert isinstance(test, TestFunction), (
            "Currently only supports one TestFunction in weak form"
        )
        found = rhs_trials if isinstance(rhs_trials, set) else {rhs_trials}
        unexpected = {
            u
            for u in found
            if u is not None and u != trial and not any(u == e for e in explicit)
        }
        if unexpected:
            names = ", ".join(sorted(str(u) for u in unexpected))
            raise ValueError(
                f"Unexpected TrialFunction(s) in weak form: {names}. Fields of "
                "other equations must be declared through `explicit_trials`."
            )

        linear, nonlinear = split_linear_nonlinear_terms(-rhs, trial, explicit=explicit)
        nonlinear = sp.expand(remove_test_function(nonlinear, test))
        return test, trial, mass_expr, linear, nonlinear

    def _node_for(self, trial: TrialFunction) -> AppliedUndef:
        """Return the shared JAXFunction node representing ``trial``."""
        for field, node in self._fields:
            if field == trial:
                return node
        V = cast(ScalarSpaceType, trial.functionspace)
        return cast(
            AppliedUndef,
            JAXFunction(
                jnp.zeros(self._coefficient_shape(V)), V, name=f"{trial.name}_jax"
            ).doit(),
        )

    def _setup_nonlinear_evaluator(self, trial: TrialFunction) -> None:
        """Compile the nonlinear physical-space evaluator for the trial field."""
        jaxfunction = self._node_for(trial)
        nonlinear_expr = replace_trial_with_jaxfunction(
            self.nonlinear_expr, trial, jaxfunction
        )
        for item in normalize_explicit(self._explicit_trials):
            foreign = cast(TrialFunction, item)
            nonlinear_expr = replace_trial_with_jaxfunction(
                nonlinear_expr, foreign, self._node_for(foreign)
            )
        if nonlinear_expr.atoms(TrialFunction):
            raise ValueError(
                "Nonlinear term still contains TrialFunctions after substitution: "
                f"{nonlinear_expr}. This usually means a coupled field was not "
                "declared through `explicit_trials`."
            )
        self.nonlinear_expr = nonlinear_expr
        self._nonlinear_jaxfunction = jaxfunction
        self._nonlinear_evaluator = compile_nonlinear_evaluator(
            nonlinear_expr,
            self.trialspace,
            self._field_order if self._field_order is not None else jaxfunction,
        )

    def _dense_matrix(self, operator: BaseMatrix) -> Array:
        """Return a dense matrix representation of a linear operator."""
        return operator.todense()

    def mass_matrix_dense(self) -> Array:
        """Return the assembled mass operator as a dense matrix."""
        return self._dense_matrix(self.mass_operator)

    def linear_matrix_dense(self) -> Array:
        """Return the assembled linear operator as a dense matrix."""
        return self._dense_matrix(self.linear_operator)

    def initial_coefficients(self, initial: sp.Expr | Array | None = None) -> Array:
        """Return coefficient-space data for an initial condition."""
        init = self.initial_condition if initial is None else initial
        if isinstance(init, sp.Expr):
            return project(init, self.trialspace)
        return jnp.asarray(init).reshape(self.trialspace.num_dofs)

    def _coerce_state(self, state0: IntegratorState) -> Array:
        """Coerce a restart state into this equation's coefficient layout."""
        return jnp.asarray(state0).reshape(self.trialspace.num_dofs)

    def apply_mass(self, uh: Array) -> Array:
        """Apply the assembled mass operator to a coefficient state."""
        return self.mass_operator @ uh

    def apply_mass_inverse(self, rhs: Array) -> Array:
        """Apply the inverse mass operator to a coefficient-space right-hand side."""
        return _solve(self.mass_operator, rhs, self._solver_options)

    def _no_nonlinear(self, uh: IntegratorState) -> Array:
        """Return a zero nonlinear contribution shaped like the test space."""
        own = uh[self._field_index] if isinstance(uh, tuple) else uh
        return jnp.zeros(self._coefficient_shape(self.testspace), dtype=own.dtype)

    def nonlinear_rhs(self, uh: IntegratorState, N: ScalarPadding = None) -> Array:
        """Return the nonlinear contribution in coefficient space.

        For a system of coupled equations `uh` is the tuple of all fields'
        coefficients, in the global field order.
        """
        if not self.has_nonlinear:
            return self._no_nonlinear(uh)
        assert self._nonlinear_evaluator is not None
        M = self._physical_shape(N)
        return self.testspace.forward(self._nonlinear_evaluator(uh, M))

    def nonlinear_rhs_scalar_product(
        self, uh: IntegratorState, N: ScalarPadding = None
    ) -> Array:
        """Return the nonlinear contribution in coefficient space.

        Do *not* apply the mass inverse to complete the forward transformation,
        because the mass inverse may be required elsewhere.
        """
        if not self.has_nonlinear:
            return self._no_nonlinear(uh)
        assert self._nonlinear_evaluator is not None
        M = self._physical_shape(N)
        return self.testspace.scalar_product(self._nonlinear_evaluator(uh, M))

    def linear_rhs(self, uh: Array) -> Array:
        """Return the linear contribution after applying the inverse mass matrix."""
        rhs = self.linear_operator @ uh
        if self.linear_forcing is not None:
            rhs = rhs + jnp.asarray(self.linear_forcing)
        return self.apply_mass_inverse(rhs)

    @jax.jit(static_argnums=(0, 2))
    def total_rhs(self, uh: Array, N: ScalarPadding = None) -> Array:
        """Return the full semi-discrete right-hand side."""
        return self.linear_rhs(uh) + self.nonlinear_rhs(uh, N)
