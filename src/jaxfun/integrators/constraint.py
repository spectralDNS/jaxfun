"""Algebraic constraint equations of a coupled system.

Not every equation of a system evolves in time. A streamfunction-vorticity
formulation, for instance, transports the vorticity but recovers the
streamfunction from a Poisson equation that holds at every instant. Such an
equation carries no time derivative and no state of its own: its field is a
function of the fields that *are* being stepped, recomputed from them at every
Runge-Kutta stage rather than integrated alongside them.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax.numpy as jnp
import sympy as sp
from flax import nnx
from sympy.core.function import AppliedUndef

from jaxfun.coordinates import get_system
from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import project
from jaxfun.la import BaseMatrix
from jaxfun.sharding import replicate
from jaxfun.typing import Array, IntegratorState, ScalarPadding, ScalarSpaceType
from jaxfun.utils import (
    normalize_explicit,
    split_linear_couplings,
    split_time_derivative_terms,
)
from jaxfun.utils.operator_tools import assemble_linear_term
from jaxfun.utils.sympy_factoring import split_linear_nonlinear_terms

from ._utils import (
    FieldCoupling,
    SolverOptions,
    apply_field_couplings,
    assemble_field_couplings,
    coefficient_shape,
    node_for,
    physical_shape,
    solve_with_options,
    validate_solver_options,
    warm_operator_solve_cache,
)
from .nonlinear import compile_field_evaluator, remove_test_function


class ConstraintSolver(nnx.Module):
    """Solve an algebraic (non-transient) weak form for its own field.

    The weak form is split the way `BaseIntegrator` splits an evolution
    equation, minus the mass term. The part linear in the constraint's own field
    assembles into one square operator, so the constraint is solved for its own
    field alone, over its own test space -- never as a block system in several
    fields at once. Everything touching a foreign field is explicit, by one of
    two routes: a term linear in a single foreign field is assembled as an
    operator between the two spaces and applied as a matrix-vector product,
    while the rest are evaluated pointwise in physical space through the shared
    JAXFunction nodes. The first route matters -- for a Poisson-type constraint
    it is the whole coupling, and it saves a padded transform and a padded
    scalar product at every stage.

    Sign convention: the weak form is a residual that vanishes, so with the
    assembled terms it reads `operator @ u + forcing + <nonlinear, v> = 0` and
    `solve_field` inverts it. Note this splits the *un-negated* right-hand side,
    unlike `BaseIntegrator`, whose equation is `M du/dt = -rhs`.
    """

    def __init__(
        self,
        equation: sp.Expr,
        *,
        own_trial: TrialFunction,
        initial: sp.Expr | Array | None = None,
        explicit_trials: Sequence[TrialFunction] = (),
        fields: Sequence[tuple[sp.Expr, AppliedUndef]] = (),
        field_order: tuple[AppliedUndef, ...] | None = None,
        field_index: int = 0,
        sparse: bool = False,
        sparse_tol: int = 1000,
        solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Assemble a constraint equation.

        Args:
            equation: Weak form without a time derivative.
            own_trial: The field this constraint determines, time-independent.
            initial: Optional initial value for the field. None means it is
                derived from the transient fields instead.
            explicit_trials: Fields of the other equations of the system.
            fields: `(time-independent trial, shared JAXFunction node)` pairs, as
                in `BaseIntegrator`.
            field_order: Global field order the compiled evaluator expects its
                coefficient tuple in.
            field_index: This constraint's slot in that global order.
            sparse: Assemble sparse operators when possible.
            sparse_tol: Sparsification tolerance passed to Galerkin assembly.
            solver_options: Options forwarded to the constraint solve.
        """
        self.sparse = sparse
        self.sparse_tol = sparse_tol
        self.initial_condition = initial
        self._explicit_trials = nnx.static(tuple(explicit_trials))
        self._fields = nnx.static(tuple(fields))
        self._field_order = nnx.static(field_order)
        self._field_index = nnx.static(field_index)
        self._solver_options: SolverOptions = nnx.static(
            validate_solver_options(solver_options)
        )

        test, linear_expr, nonlinear_expr, coupling_exprs = (
            self._extract_equation_terms(equation, own_trial)
        )
        self.trialspace = cast(ScalarSpaceType, own_trial.functionspace)
        self.testspace = cast(ScalarSpaceType, test.functionspace)
        self._state_shape = coefficient_shape(self.trialspace)
        self.linear_expr = linear_expr
        self.nonlinear_expr = nonlinear_expr
        self.has_nonlinear = bool(sp.sympify(nonlinear_expr) != 0)

        operator, forcing = assemble_linear_term(
            linear_expr, sparse=self.sparse, sparse_tol=self.sparse_tol
        )
        if operator is None:
            raise ValueError(
                f"Constraint equation for {own_trial.name} assembled no operator "
                f"for {own_trial.name} itself. A constraint must be solvable for "
                "its own field, so the field has to appear linearly in it."
            )
        self.operator: BaseMatrix = nnx.data(operator)
        # Replicated for the same reason `BaseIntegrator.linear_forcing` is:
        # assembly places a right-hand side on the global spectral sharding, and
        # a constraint solver is reached through `_advance`'s closure because the
        # system stepper is a static argument there. JAX refuses to close over an
        # array spanning devices this process cannot address, so an inhomogeneous
        # constraint on a divisible multidimensional space would fail to compile
        # under multi-process SPMD. See `replicate`.
        self.forcing: Array | None = nnx.data(replicate(forcing))

        self._couplings: tuple[FieldCoupling, ...] = nnx.data(
            assemble_field_couplings(
                coupling_exprs,
                self._node_for,
                self._field_order,
                sparse=self.sparse,
                sparse_tol=self.sparse_tol,
            )
        )

        self._nonlinear_evaluator: (
            Callable[[IntegratorState, ScalarPadding], Array] | None
        ) = None
        if self.has_nonlinear:
            if field_order is None:
                # `solve_field` takes the whole system's coefficient tuple, so
                # without a global order there is nothing to index it by -- and a
                # constraint couples to foreign fields by its very nature, so
                # this is not a case worth quietly defaulting.
                raise ValueError(
                    f"Constraint equation for {own_trial.name} couples to other "
                    "fields, so `field_order` is required to know which entry of "
                    "the state tuple each field is."
                )
            self.nonlinear_expr, _, self._nonlinear_evaluator = compile_field_evaluator(
                nonlinear_expr,
                own_trial,
                self._explicit_trials,
                self._node_for,
                self.trialspace,
                self._field_order,
            )

    def _extract_equation_terms(
        self, equation: sp.Expr, own_trial: TrialFunction
    ) -> tuple[TestFunction, sp.Expr, sp.Expr, dict[Any, sp.Expr]]:
        """Split a constraint weak form into linear, coupling and nonlinear parts."""
        t = get_system(equation).base_time()
        # Splitting on the time derivative also drops the time argument from the
        # transient foreign fields, which is what lets `explicit` below match
        # them -- their nodes in the equation are the time-dependent ones.
        lhs, rhs = split_time_derivative_terms(equation, t)
        if sp.sympify(lhs) != 0:
            raise ValueError(
                "A constraint equation must not contain a time derivative, got "
                f"{lhs}. Equations that evolve in time are integrated instead."
            )

        test, trials = get_basisfunctions(rhs)
        tests = test if isinstance(test, set) else {test}
        if len(tests) != 1:
            raise ValueError(
                "Currently only supports one TestFunction in weak form, got "
                f"{len(tests)}"
            )
        test = tests.pop()
        assert isinstance(test, TestFunction), (
            "Currently only supports one TestFunction in weak form"
        )

        explicit = normalize_explicit(self._explicit_trials)
        found = trials if isinstance(trials, set) else {trials}
        unexpected = {
            u
            for u in found
            if u is not None and u != own_trial and not any(u == e for e in explicit)
        }
        if unexpected:
            names = ", ".join(sorted(str(u) for u in unexpected))
            raise ValueError(
                f"Unexpected TrialFunction(s) in weak form: {names}. Fields of "
                "other equations must be declared through `explicit_trials`."
            )

        linear, nonlinear = split_linear_nonlinear_terms(
            rhs, own_trial, explicit=explicit
        )
        # A coupling linear in one foreign field -- the usual case, and the whole
        # of it for a Poisson-type constraint -- becomes an operator rather than
        # a pointwise evaluation on the padded mesh.
        couplings, nonlinear = split_linear_couplings(nonlinear, explicit)
        nonlinear = sp.expand(remove_test_function(nonlinear, test))
        return test, linear, nonlinear, couplings

    def _node_for(self, trial: TrialFunction) -> AppliedUndef:
        """Return the shared JAXFunction node representing ``trial``."""
        return node_for(self._fields, trial)

    def initial_coefficients(
        self, initial: sp.Expr | Array | None = None
    ) -> Array | None:
        """Return coefficient-space data for the field's initial value.

        Returns None when neither the caller nor the constructor supplied one,
        which is the caller's signal to derive the field from the constraint.
        """
        init = self.initial_condition if initial is None else initial
        if init is None:
            return None
        if isinstance(init, sp.Expr):
            return project(init, self.trialspace)
        return jnp.asarray(init).reshape(self.trialspace.num_dofs)

    def setup(self) -> None:
        """Factorize the constraint operator before time stepping starts.

        Nothing here depends on the step size, but it has to run outside the
        jitted step for the same reason `BaseIntegrator.setup` does: the solver
        picks its path by inspecting matrix values, which are tracers by then.
        """
        warm_operator_solve_cache(
            self.operator, self._state_shape, self._solver_options
        )

    def solve_field(self, states: tuple[Array, ...], N: ScalarPadding = None) -> Array:
        """Return this constraint's field, given every field's coefficients.

        `states` is the whole system's coefficient tuple in global field order;
        only the foreign fields the constraint couples to are read, and this
        constraint's own slot is ignored (it is what gets computed here).
        """
        # Everything in the residual that does not involve the own field, in
        # scalar-product form; the solve then inverts `operator @ u = -total`.
        total = apply_field_couplings(self._couplings, states)
        if self._nonlinear_evaluator is not None:
            pointwise = self.testspace.scalar_product(
                self._nonlinear_evaluator(states, physical_shape(self.testspace, N))
            )
            total = pointwise if total is None else total + pointwise
        if self.forcing is not None:
            forcing = jnp.asarray(self.forcing)
            total = forcing if total is None else total + forcing
        if total is None:
            # Homogeneous: `operator @ u = 0`, so the field vanishes identically.
            return jnp.zeros(self._state_shape)
        return solve_with_options(self.operator, -total, self._solver_options)
