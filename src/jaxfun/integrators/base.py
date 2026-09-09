"""Base abstractions for time integrators on Galerkin coefficient spaces."""

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
from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import BoundaryForcing, project
from jaxfun.la import BaseMatrix, IdentityMatrix, ZeroMatrix
from jaxfun.sharding import pin_state, replicate, replicate_scalar
from jaxfun.typing import Array, IntegratorState, ScalarPadding, ScalarSpaceType
from jaxfun.utils import (
    get_time_independent,
    normalize_explicit,
    split_linear_couplings,
    split_linear_nonlinear_terms,
    split_time_derivative_terms,
)
from jaxfun.utils.operator_tools import (
    assemble_boundary_term,
    assemble_linear_term,
)
from jaxfun.utils.sympy_factoring import time_derivative_as_operator

from ._utils import (
    CoupledOperator,
    SolverOptions,
    apply_field_couplings,
    assemble_field_couplings,
    boundary_values,
    coefficient_shape,
    node_for,
    physical_shape,
    solve_with_options,
    split_couplings,
    validate_solver_options,
    warm_operator_solve_cache,
)
from .nonlinear import compile_field_evaluator, remove_test_function


class TimeStepper[StateT: IntegratorState](ABC, nnx.Module):
    """Step-loop driver shared by scalar and system integrators.

    The coefficient state is any pytree of arrays: a single array for a scalar
    equation, or one array per field for a system of coupled equations.
    Subclasses provide `step` and the initial state; everything below is
    agnostic to which of the two it is.
    """

    time: tuple[float, float] | None

    @abstractmethod
    def _step_impl(
        self,
        u_hat: StateT,
        dt: float,
        N: ScalarPadding = None,
        t: Array | float = 0.0,
        /,
    ) -> StateT:
        """Advance the state one step -- the traceable body of a step.

        Deliberately *not* jitted, so that it can be traced from inside an
        enclosing computation. `_advance` is the only jit boundary: `step` asks
        it for one step and `solve` for a batch, so the two cannot drift into
        compiling different things, which they previously did.

        Positional-only, because `_advance` is the only caller and the state
        is not the same thing in every subclass -- one array for a scalar
        equation, one per field for a system. Binding the parameter names would
        make every subclass answer to `u_hat`.

        `t` is the time at the start of the step. It defaults to zero so that a
        stepper with no time-dependent data -- which is every stepper whose
        boundary values and sources are steady -- behaves exactly as before, and
        so that calling a step directly stays a two-argument affair.
        """
        ...

    def step(
        self,
        u_hat: StateT,
        dt: float,
        N: ScalarPadding = None,
        t: float = 0.0,
    ) -> StateT:
        """Advance the state one step, as one compiled computation.

        A batch of one, so that a single step and a whole `solve` go through
        exactly the same machinery. They used to have separate jit entry points,
        and on more than one device the combination was fatal: whichever
        compiled first left an entry the other could not execute.

        `dt` is static, so a sweep over step sizes recompiles once per value.
        See `_advance` for why it has to be.
        """
        return _advance(self, u_hat, dt, 1, N, _as_start_time(t))

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
        """Precompute step-size-dependent coefficients before time stepping.

        Idempotent for a given `dt`, and that is load-bearing, not just thrifty.
        `_setup_impl` *replaces* operators, and the integrator is a static jit
        argument keyed by identity -- so a rebuild behind an already-compiled
        `_advance` leaves the cache holding an executable built around the old
        arrays. It hits, because the key is the same object. On one device the
        old arrays are embedded constants of equal value and nothing is visibly
        wrong; on more than one they are hoisted into the executable's argument
        list, the caller supplies the new ones, and PJRT aborts the process over
        the count. Not rebuilding is the fix for both.

        Re-setting up at a *different* `dt` is still allowed and still rebuilds;
        that is a real change of operator, and `dt` is part of `_advance`'s jit
        key, so it compiles afresh.
        """
        dt = float(dt)
        if getattr(self, "_setup_dt", None) == dt:
            return
        self._setup_impl(dt)
        self._setup_dt = dt

    def _setup_impl(self, dt: float) -> None:
        """Build whatever depends on the step size. Nothing, by default."""
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

    def end_time(
        self,
        dt: float,
        steps: int | None = None,
        trange: tuple[float, float] | None = None,
    ) -> float:
        """Return the time a `solve` with these arguments would finish at.

        Use it to reconstruct a solution whose boundary data moved::

            V.backward(u_hat, t=integrator.end_time(dt, steps))
        """
        # A query rather than something `solve` records, because `solve` must
        # store nothing on the stepper: binding or rebinding an attribute
        # between two `_advance` calls changes the module the jit cache was
        # keyed on, and the mismatch surfaces as a PJRT abort over the argument
        # count rather than as anything to do with the attribute. `setup`
        # describes the same hazard from the other direction.
        t0, _, n_steps = self.resolve_time(dt, steps=steps, trange=trange)
        return t0 + n_steps * dt

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
        t_start, _, n_steps = self.resolve_time(dt, steps=steps, trange=trange)

        if state0 is None:
            u_hat = self.initial_coefficients()
        else:
            u_hat = self._coerce_state(state0)
        if n_steps <= 0:
            if not return_batch_snapshots:
                return u_hat
            return jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), u_hat)

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
        taken = 0
        for _ in iterator:
            u_hat = _advance(
                self, u_hat, dt, batch_len, N, _as_start_time(t_start + taken * dt)
            )
            taken += batch_len
            if return_batch_snapshots:
                states.append(u_hat)
            if any(
                bool(jnp.isnan(leaf).any() or jnp.isinf(leaf).any())
                for leaf in jax.tree.leaves(u_hat)
            ):
                diverged = True
                break

        if remainder and not diverged:
            u_hat = _advance(
                self, u_hat, dt, remainder, N, _as_start_time(t_start + taken * dt)
            )
            taken += remainder
            if return_batch_snapshots:
                states.append(u_hat)

        if return_batch_snapshots:
            return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        return u_hat


@jax.jit(static_argnums=(2, 4))
def _advance[StateT: IntegratorState](
    stepper: TimeStepper[StateT],
    u_hat: StateT,
    dt: float,
    n_steps: int,
    N: ScalarPadding = None,
    t0: Array | None = None,
) -> StateT:
    """Advance `n_steps` steps as a single compiled computation.

    `stepper` is *traced*: it flattens as a pytree and every array it holds
    arrives as a jit argument carrying its own sharding. Static -- hashed by
    identity, the arrays folded in as constants -- measures a few percent
    faster, and that is what this used to do. It cannot do it any more, because
    a constant is exactly what a per-device operator may not be.

    JAX refuses to close over an array spanning devices the process cannot
    address, so under a static stepper every operator had to be replicated, and
    the per-wavenumber factors were replicated on every device at full size.
    Passing them instead is what the error message asks for and what makes
    per-device factorisation reachable: the sharded factor arrays cross the
    boundary as arguments, and `shard_map` hands each device its own block.

    Note that only *arrays* gain this freedom. Anything used as a Python value
    -- an index, a shape, a flag -- has to stay under `nnx.static`, or it
    arrives as a tracer where a concrete value is needed. `_coupling_slots` is
    split out from `_couplings` for precisely that reason; see `split_couplings`.

    `n_steps` is *traced*, so this compiles exactly once per `(dt, N)` and the
    loop is a `while_loop` over a dynamic bound. Static would let XLA see the
    trip count, but it also gives the jit cache more than one entry, and on a
    multi-device mesh a *hit* on such an entry aborts the process inside PJRT
    ("Execution supplied 1 arguments but compiled program expected 10") -- a
    dispatch collision, not anything about this loop. One entry cannot collide.
    Reproduce with `n_steps` static and JAX 0.11: call at 1, at 10, then at 10
    again; the third call dies.

    Call it positionally. `static_argnums` has no effect on an argument passed
    by keyword, so `_advance(..., N=N)` makes `N` traced where `_advance(..., N)`
    makes it static -- two different signatures for the same call, and on more
    than one device the second one to run dies in PJRT over the argument count.

    `dt` is static for a subtler reason. Left traced it is a rank-0 parameter,
    and GSPMD propagates a sharding onto it from the sharded arrays it is
    multiplied into: `P("k")` on a scalar, which JAX then cannot apply
    (`IndexError` out of `_to_xla_hlo_sharding`). Static, it is a compile-time
    constant no propagation can reach, and `dt * a_ij` folds into the Butcher
    coefficients as a bonus. A `solve` uses one step size throughout, so nothing
    recompiles that would not have anyway.
    """

    def body(i: int, u: StateT) -> StateT:
        return pin_state(stepper._step_impl(u, dt, N, _step_time(t0, i, dt)))

    return jax.lax.fori_loop(0, n_steps, body, pin_state(u_hat))


def _as_start_time(t0: float) -> Array:
    """Return a batch start time in the shape `_advance` wants it.

    Shape `(1,)`, not a scalar; see `_step_time` for why that matters.
    """
    return jnp.asarray([t0], dtype=jnp.result_type(float))


def _step_time(t0: Array | None, i: int, dt: float) -> Array | float:
    """Return the time at the start of step `i` of a batch beginning at `t0`.

    `t0` arrives as a shape-`(1,)` array rather than a scalar, and that is
    deliberate. A rank-0 float parameter is exactly what `dt` had to stop being:
    GSPMD propagates a sharding backwards onto the parameters a sharded result
    was computed from, and `P("k")` on a rank-0 parameter is the `IndexError`
    out of `_to_xla_hlo_sharding` described above and in `pin_state`. A rank-1
    array carries `P(None)` instead. It is then constrained to a replicated
    sharding here, so the placement is stated rather than inferred -- the same
    discipline `pin_state` applies to the state.

    `dt` is a compile-time constant, so `t0 + i*dt` costs nothing at runtime.
    Note it is computed in the loop index rather than accumulated, so a long
    batch does not drift; `solve` likewise recomputes `t0` per batch from the
    step count.
    """
    if t0 is None:
        return 0.0
    return replicate_scalar(t0)[0] + i * dt


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
            validate_solver_options(solver_options)
        )

        test, trial, mass_expr, linear_expr, nonlinear_expr, coupling_exprs = (
            self._extract_equation_terms(equation)
        )
        self.trialspace = cast(ScalarSpaceType, trial.functionspace)
        self.testspace = cast(ScalarSpaceType, test.functionspace)
        self._state_shape = coefficient_shape(self.trialspace)
        self.mass_expr = mass_expr
        self.linear_expr = linear_expr
        self.nonlinear_expr = nonlinear_expr
        self.has_nonlinear = bool(sp.sympify(nonlinear_expr) != 0)

        mass_operator, mass_forcing = assemble_linear_term(
            self.mass_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        self._transient_boundary = nnx.static(self._has_transient_boundary())
        if mass_forcing is not None and not self._transient_boundary:
            # A space with inhomogeneous boundary conditions splits the solution
            # into a free part and a fixed boundary lifting B, so the mass term
            # assembles as `M @ u_hat + inner(v, B)`. The whole term sits under
            # d/dt, and B is constant in time, so the lifting contributes
            # nothing to d/dt(M @ u_hat + inner(v, B)) = M @ du_hat/dt and is
            # dropped here. When B *does* vary in time, `d/dt inner(v, B)` is
            # carried on the right-hand side instead; see `forcing_at`.
            self._check_mass_forcing_is_a_lifting()
        if mass_operator is None:
            mass_operator = IdentityMatrix(self._state_shape)
        self.mass_operator: BaseMatrix = nnx.data(mass_operator)
        self.mass_diag: Array | None = nnx.data(self.mass_operator.diagonal_or_none())
        if self.mass_diag is None:
            warm_operator_solve_cache(
                self.mass_operator, self._state_shape, self._solver_options
            )

        linear_operator, linear_forcing = assemble_linear_term(
            self.linear_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        if linear_operator is None:
            linear_operator = ZeroMatrix(self._state_shape)
        self.linear_operator: BaseMatrix = nnx.data(linear_operator)

        # Boundary blocks, kept uncontracted so they can be re-evaluated as the
        # lifting moves. `inner` has already folded the boundary contribution
        # into `linear_forcing` at the lifting the space was built with, so that
        # frozen copy is subtracted back out and `linear_forcing` is left
        # holding the genuine source terms alone. Both sides go through the same
        # contraction, so what is removed is exactly what was added.
        self._mass_boundary: BoundaryForcing | None = nnx.data(None)
        self._linear_boundary: BoundaryForcing | None = nnx.data(None)
        if self._transient_boundary:
            if not self.supports_transient_boundary:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support time-dependent "
                    "boundary data. Use IMEXRungeKutta or BackwardEuler."
                )
            self._mass_boundary = nnx.data(assemble_boundary_term(self.mass_expr))
            self._linear_boundary = nnx.data(assemble_boundary_term(self.linear_expr))
            if self._linear_boundary is not None and linear_forcing is not None:
                linear_forcing = linear_forcing - self._linear_boundary()

        # Replicated, not sharded: this is stored on the integrator and so is
        # reached through `_advance`'s closure. See `replicate`.
        self.linear_forcing: Array | None = nnx.data(replicate(linear_forcing))
        self.linear_diag: Array | None = nnx.data(
            self.linear_operator.diagonal_or_none()
        )

        _coupling_slots, _coupling_ops = split_couplings(
            assemble_field_couplings(
                coupling_exprs,
                self._node_for,
                self._field_order,
                sparse=self.sparse,
                sparse_tol=self.sparse_tol,
            )
        )
        self._coupling_slots: tuple[int, ...] = nnx.static(_coupling_slots)
        self._couplings: tuple[CoupledOperator, ...] = nnx.data(_coupling_ops)

        self._nonlinear_jaxfunction: AppliedUndef | None = None
        self._nonlinear_evaluator: (
            Callable[[IntegratorState, ScalarPadding], Array] | None
        ) = None
        if self.has_nonlinear:
            self._setup_nonlinear_evaluator(trial)

    supports_transient_boundary: bool = True
    """Whether this stepper can carry boundary data that varies in time."""

    def _has_transient_boundary(self) -> bool:
        """Return True if any of the trial space's boundary values varies in time."""
        t = self.trialspace.system.base_time()
        return any(val.has(t) for val in boundary_values(self.trialspace))

    def _check_mass_forcing_is_a_lifting(self) -> None:
        """Check that a forcing under the time derivative is boundary data.

        The mass term is dropped on the steady path, and the only thing it is
        allowed to hold is `inner(v, B)` for a lifting `B` that differentiates
        away. Anything else would be silently discarded. Boundary values may
        still vary in space: in a tensor-product space each 1D factor carries
        its own, which may depend on the remaining coordinates.

        Only reached when the boundary data is steady -- `_has_transient_boundary`
        has already sent the other case to `forcing_at`, which keeps the term.
        """
        if not boundary_values(self.trialspace):
            raise ValueError(
                "Time-derivative operator assembly produced forcing, but the "
                "trial space has no boundary conditions to explain it"
            )

    def _extract_equation_terms(
        self, equation: sp.Expr
    ) -> tuple[
        TestFunction, TrialFunction, sp.Expr, sp.Expr, sp.Expr, dict[Any, sp.Expr]
    ]:
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
        # Terms linear in one foreign field assemble as operators instead of
        # being evaluated pointwise; the rest stay with the nonlinear evaluator.
        couplings, nonlinear = split_linear_couplings(nonlinear, explicit)
        nonlinear = sp.expand(remove_test_function(nonlinear, test))
        return test, trial, mass_expr, linear, nonlinear, couplings

    def _node_for(self, trial: TrialFunction) -> AppliedUndef:
        """Return the shared JAXFunction node representing ``trial``."""
        return node_for(self._fields, trial)

    def _setup_nonlinear_evaluator(self, trial: TrialFunction) -> None:
        """Compile the nonlinear physical-space evaluator for the trial field."""
        (
            self.nonlinear_expr,
            self._nonlinear_jaxfunction,
            self._nonlinear_evaluator,
        ) = compile_field_evaluator(
            self.nonlinear_expr,
            trial,
            self._explicit_trials,
            self._node_for,
            self.trialspace,
            self._field_order,
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

    @property
    def start_time(self) -> float:
        """Return the time the integration starts at."""
        return 0.0 if self.time is None else float(self.time[0])

    def initial_coefficients(self, initial: sp.Expr | Array | None = None) -> Array:
        """Return coefficient-space data for an initial condition."""
        # The state holds the *homogeneous* coefficients, so projecting takes
        # the boundary lifting back out -- and the one to take out is the
        # lifting at the start time, not at zero. For steady data the two
        # coincide and nothing changes.
        init = self.initial_condition if initial is None else initial
        if isinstance(init, sp.Expr):
            t = self.start_time if self._transient_boundary else None
            return project(init, self.trialspace, t)
        return jnp.asarray(init).reshape(self.trialspace.num_dofs)

    def _coerce_state(self, state0: IntegratorState) -> Array:
        """Coerce a restart state into this equation's coefficient layout."""
        return jnp.asarray(state0).reshape(self.trialspace.num_dofs)

    def apply_mass(self, uh: Array) -> Array:
        """Apply the assembled mass operator to a coefficient state."""
        return self.mass_operator @ uh

    def apply_mass_inverse(self, rhs: Array) -> Array:
        """Apply the inverse mass operator to a coefficient-space right-hand side."""
        return solve_with_options(self.mass_operator, rhs, self._solver_options)

    def _no_nonlinear(self, uh: IntegratorState) -> Array:
        """Return a zero nonlinear contribution shaped like the test space."""
        own = uh[self._field_index] if isinstance(uh, tuple) else uh
        return jnp.zeros(coefficient_shape(self.testspace), dtype=own.dtype)

    def nonlinear_rhs(self, uh: IntegratorState, N: ScalarPadding = None) -> Array:
        """Return the nonlinear contribution in coefficient space.

        For a system of coupled equations `uh` is the tuple of all fields'
        coefficients, in the global field order.
        """
        if self._couplings:
            # `forward` divides by the *test space* mass, while the coupling
            # operators produce a scalar product, and this equation's own mass
            # operator need not be either. Only the explicit scalar-product path
            # is used for systems, so rather than guess a normalization here:
            raise NotImplementedError(
                "nonlinear_rhs does not carry assembled field couplings; use "
                "nonlinear_rhs_scalar_product, which the system integrators do."
            )
        if not self.has_nonlinear:
            return self._no_nonlinear(uh)
        assert self._nonlinear_evaluator is not None
        M = physical_shape(self.testspace, N)
        return self.testspace.forward(self._nonlinear_evaluator(uh, M))

    def nonlinear_rhs_scalar_product(
        self, uh: IntegratorState, N: ScalarPadding = None
    ) -> Array:
        """Return the nonlinear contribution in coefficient space.

        Covers both explicit paths: terms evaluated pointwise in physical space,
        and terms linear in a single foreign field, which are applied as
        assembled operators instead.

        Do *not* apply the mass inverse to complete the forward transformation,
        because the mass inverse may be required elsewhere.
        """
        total = apply_field_couplings(self._coupling_slots, self._couplings, uh)
        if self.has_nonlinear:
            assert self._nonlinear_evaluator is not None
            M = physical_shape(self.testspace, N)
            pointwise = self.testspace.scalar_product(self._nonlinear_evaluator(uh, M))
            total = pointwise if total is None else total + pointwise
        return self._no_nonlinear(uh) if total is None else total

    def forcing_at(self, t: Array | float = 0.0) -> Array | None:
        """Return the right-hand-side forcing at time `t`.

        For steady boundary data this is `linear_forcing`, unchanged.
        """
        # When the lifting varies, two terms join the source: `a(v, B(t))`, the
        # stiffness acting on the lifting, which `inner` would have collapsed at
        # a single time; and `-<v, dB/dt>`, what the lifting contributes under
        # the time derivative. The second is dropped for steady data because it
        # is then exactly zero -- and still is here, the rate coming from
        # differentiating the boundary values symbolically.
        if self._linear_boundary is None:
            return self.linear_forcing
        total = self._linear_boundary(t)
        if self._mass_boundary is not None:
            total = total - self._mass_boundary.rate(t)
        if self.linear_forcing is not None:
            total = total + jnp.asarray(self.linear_forcing)
        return total

    def linear_rhs(self, uh: Array, t: Array | float = 0.0) -> Array:
        """Return the linear contribution after applying the inverse mass matrix."""
        rhs = self.linear_operator @ uh
        forcing = self.forcing_at(t)
        if forcing is not None:
            rhs = rhs + jnp.asarray(forcing)
        return self.apply_mass_inverse(rhs)

    @jax.jit(static_argnums=(0, 2))
    def total_rhs(
        self, uh: Array, N: ScalarPadding = None, t: Array | float = 0.0
    ) -> Array:
        """Return the full semi-discrete right-hand side."""
        return self.linear_rhs(uh, t) + self.nonlinear_rhs(uh, N)
