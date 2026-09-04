"""Time integration of systems of equations coupled through nonlinear terms.

Each equation of the system is advanced by its own scalar integrator, so the
implicit (linear) operators stay decoupled and are assembled and factorized once
per equation. The equations see each other only explicitly, from the state of
*all* fields at the current stage: terms linear in a single foreign field are
assembled as operators between the two spaces and applied as matrix-vector
products, and everything else is evaluated pointwise in physical space. Every
field is represented by a single shared JAXFunction node, so updating a stage
makes it visible to all equations at once.
"""

import inspect
from abc import ABC
from collections.abc import Sequence
from typing import Any, cast, get_args, get_origin

import jax.numpy as jnp
import sympy as sp
from flax import nnx
from sympy.core.function import AppliedUndef

from jaxfun.galerkin import TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.typing import Array, IntegratorState, ScalarPadding, ScalarSpaceType
from jaxfun.utils import get_time_independent

from ._utils import (
    apply_field_couplings,
    coefficient_shape,
    constrained_trial,
    mesh_axes,
    scale_real,
    transient_trial,
)
from .base import BaseIntegrator, TimeStepper
from .constraint import ConstraintSolver
from .nonlinear import compile_coupled_nonlinear_evaluator


class FieldRegistry:
    """One shared symbolic JAXFunction node per field of a coupled system.

    There must be exactly one node per field: SymPy compares and hashes applied
    undefined functions by name, so two independently created nodes sharing a
    name would be indistinguishable inside an expression while carrying
    independent coefficient arrays -- one field would silently evaluate to the
    other's values.
    """

    def __init__(self, trials: Sequence[TrialFunction]) -> None:
        """Create the shared node for each field, keyed by its trial function."""
        names = [u.name for u in trials]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Coupled fields must have unique names, got {names}. Rename the "
                "TrialFunctions so each field of the system is distinguishable."
            )
        self.trials = tuple(trials)
        self.independent = tuple(get_time_independent(u) for u in trials)
        self.nodes = tuple(
            cast(
                AppliedUndef,
                JAXFunction(
                    jnp.zeros(
                        coefficient_shape(cast(ScalarSpaceType, u.functionspace))
                    ),
                    u.functionspace,
                    name=f"{u.name}_jax",
                ).doit(),
            )
            for u in trials
        )
        # Pairs rather than a dict: JAX sorts dictionary keys when flattening a
        # pytree, and SymPy objects are not orderable.
        self.pairs: tuple[tuple[sp.Expr, AppliedUndef], ...] = tuple(
            zip(self.independent, self.nodes, strict=True)
        )

    def __len__(self) -> int:
        return len(self.trials)


class SystemIntegrator[IntegratorT: BaseIntegrator](
    TimeStepper[tuple[Array, ...]], ABC
):
    """Base class for integrating systems coupled through nonlinear terms.

    Holds one scalar sub-integrator of type `integrator_type` per *evolution*
    equation, plus the shared field registry that lets their nonlinear terms see
    each other. `IntegratorT` is that sub-integrator type, so subclasses keep
    access to the methods specific to it. Subclasses implement `step`.

    An equation without a time derivative is a *constraint*: it is solved for
    its own field at every stage rather than integrated, and is held in
    `constraints` instead. Its field is still part of the state tuple -- carried
    along in equation declaration order, so `step` returns one array per
    equation either way -- but it is recomputed from the transient fields rather
    than stepped, and so can never drift out of sync with them.

    `integrators` and `constraints` are each indexed in their own order, and
    `transient_slots` / `constraint_slots` map those local indices to the global
    field order that the state tuple and the shared nodes use.
    """

    integrator_type: type[IntegratorT]
    integrators: tuple[IntegratorT, ...]
    constraints: tuple[ConstraintSolver, ...]

    def __init_subclass__(cls, **kwargs) -> None:
        """Take the sub-integrator class from the `SystemIntegrator[...]` argument.

        Subclassing `SystemIntegrator[IMEXRungeKutta]` states the sub-integrator
        type for the type checker; constructing one needs the same class at
        runtime. Reading it back off the base means it is written once instead of
        being declared and then repeated as an attribute that could disagree
        with it. Subclasses that stay generic (a type variable rather than a
        class as the argument) simply inherit nothing here.
        """
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is not SystemIntegrator:
                continue
            (argument,) = get_args(base)
            if isinstance(argument, type):
                cls.integrator_type = argument
            return

    def __init__(
        self,
        equations: Sequence[sp.Expr],
        *,
        initial: Sequence[sp.Expr | Array | None],
        time: tuple[float, float] | None = None,
        **params,
    ):
        """Build one sub-integrator or constraint solver per equation.

        Args:
            equations: One weak form per equation. An equation with a
                first-order time derivative of its own transient TrialFunction
                is integrated; one without is a constraint, solved at every
                stage for the field no other equation evolves.
            initial: One entry per equation, in the same order. A constraint
                equation may be given None, and its field is then derived from
                the transient ones; passing a value instead states it directly,
                which is worth doing when the field is what the problem is
                naturally posed in terms of.
            time: Optional default integration interval.
            **params: Forwarded unchanged to every sub-integrator.
        """
        equations = tuple(equations)
        initial = tuple(initial)
        if len(equations) != len(initial):
            raise ValueError(
                f"Got {len(equations)} equations but {len(initial)} initial "
                "conditions; there must be exactly one of each per field."
            )
        if len(equations) == 0:
            raise ValueError("A system needs at least one equation")

        self.time = time
        transient = tuple(transient_trial(eq) for eq in equations)
        if all(u is None for u in transient):
            raise ValueError(
                "A system needs at least one equation with a time derivative; "
                "every equation given is a constraint, so nothing evolves."
            )
        evolved = tuple(u for u in transient if u is not None)
        trials = tuple(
            u if u is not None else constrained_trial(eq, evolved)
            for eq, u in zip(equations, transient, strict=True)
        )

        self.transient_slots = nnx.static(
            tuple(k for k, u in enumerate(transient) if u is not None)
        )
        self.constraint_slots = nnx.static(
            tuple(k for k, u in enumerate(transient) if u is None)
        )
        for k in self.transient_slots:
            if initial[k] is None:
                raise ValueError(
                    f"Equation {k} has a time derivative, so its field "
                    f"{trials[k].name} is integrated and needs an initial "
                    "condition. Only a constraint equation may be given None."
                )

        self.fields = nnx.static(FieldRegistry(trials))
        registry: FieldRegistry = self.fields

        def coupling(k: int) -> dict[str, Any]:
            """Hooks tying equation `k` to every other field of the system."""
            return {
                "explicit_trials": tuple(u for j, u in enumerate(trials) if j != k),
                "fields": registry.pairs,
                "field_order": registry.nodes,
                "field_index": k,
            }

        # A constraint is not stepped, so the stepping-specific parameters of the
        # sub-integrators (a Butcher tableau, say) mean nothing to it. It is
        # offered only the assembly and solver parameters it declares. Inspect
        # `__init__` rather than the class: `nnx.Module` wraps construction, so
        # the class signature is a bare `(*args, **kwargs)` that would match
        # nothing and silently drop every parameter.
        accepted = frozenset(inspect.signature(ConstraintSolver.__init__).parameters)
        constraint_params = {k: v for k, v in params.items() if k in accepted}

        self.integrators = nnx.data(
            tuple(
                self.integrator_type(
                    equations[k],
                    # Not None at a transient slot; checked above.
                    initial=cast("sp.Expr | Array", initial[k]),
                    time=time,
                    **coupling(k),
                    **params,
                )
                for k in self.transient_slots
            )
        )
        self.constraints = nnx.data(
            tuple(
                ConstraintSolver(
                    equations[k],
                    own_trial=trials[k],
                    initial=initial[k],
                    **coupling(k),
                    **constraint_params,
                )
                for k in self.constraint_slots
            )
        )
        self._validate_common_mesh()
        self._setup_coupled_nonlinear_evaluator()

    def _proportional_groups(
        self, indices: Sequence[int]
    ) -> tuple[tuple[int, tuple[tuple[int, complex], ...]], ...]:
        """Group equations whose nonlinear terms differ only by a constant factor.

        Mass-action kinetics makes this the normal case rather than a special
        one: a reaction consumed by one species and produced by another enters
        the two equations with the same functional form and different
        stoichiometric coefficients (Schnakenberg's `+u^2 v` and `-u^2 v`,
        Gray-Scott's `-u v^2` and `+u v^2`).

        Returns `(representative, ((equation, coefficient), ...))` per group.
        Grouping requires the *same* test space object, since the projection is
        what gets shared.
        """
        groups: list[tuple[int, list[tuple[int, complex]]]] = []
        for k in indices:
            expr = self.integrators[k].nonlinear_expr
            space = self.integrators[k].testspace
            for rep, members in groups:
                if self.integrators[rep].testspace is not space:
                    continue
                try:
                    ratio = sp.simplify(expr / self.integrators[rep].nonlinear_expr)
                except (TypeError, ValueError, ZeroDivisionError):  # pragma: no cover
                    continue
                if ratio.is_number:
                    members.append((k, complex(ratio)))
                    break
            else:
                groups.append((k, [(k, 1.0 + 0j)]))
        return tuple((rep, tuple(members)) for rep, members in groups)

    def _setup_coupled_nonlinear_evaluator(self) -> None:
        """Compile the equations' nonlinear terms so they share their evaluation.

        Each sub-integrator keeps its own standalone evaluator; the system steps
        through this one, which shares two things across equations:

        - the physical-space evaluation, so each field is transformed once per
          stage rather than once per equation;
        - the projection back onto the test space, for equations whose nonlinear
          terms are proportional (see `_proportional_groups`). This is the one
          that matters: the transforms above are common subexpressions that XLA
          already merges on its own, whereas terms differing by a constant reach
          the transform with different values and cannot be merged.

        Constraints are deliberately left out and keep their own evaluators.
        They could not share this pass even in principle: a constraint has to be
        solved *before* the evolution equations are evaluated, because those
        read its field at the stage it was just solved for.
        """
        nonlinear = tuple(k for k, g in enumerate(self.integrators) if g.has_nonlinear)
        self._coupled_nonlinear_evaluator = None
        self._nonlinear_groups = nnx.static(())
        if not nonlinear:
            return
        groups = self._proportional_groups(nonlinear)
        self._nonlinear_groups = nnx.static(groups)
        self._coupled_nonlinear_evaluator = compile_coupled_nonlinear_evaluator(
            tuple(self.integrators[rep].nonlinear_expr for rep, _ in groups),
            # Every field shares this mesh (checked by `_validate_common_mesh`),
            # so any of the spaces can evaluate the purely spatial factors.
            self.integrators[nonlinear[0]].trialspace,
            self.fields.nodes,
        )

    def nonlinear_scalar_products(
        self, states: tuple[Array, ...], N: ScalarPadding = None
    ) -> tuple[Array, ...]:
        """Return each equation's nonlinear term in coefficient space.

        Includes both explicit paths: assembled couplings to other fields, and
        terms evaluated pointwise. As in
        `BaseIntegrator.nonlinear_rhs_scalar_product`, the mass inverse is
        deliberately not applied.
        """
        results: list[Array] = []
        for g in self.integrators:
            coupling = apply_field_couplings(g._coupling_slots, g._couplings, states)
            results.append(g._no_nonlinear(states) if coupling is None else coupling)
        if self._coupled_nonlinear_evaluator is None:
            return tuple(results)
        values = self._coupled_nonlinear_evaluator(states, N)
        for (rep, members), value in zip(self._nonlinear_groups, values, strict=True):
            projected = self.integrators[rep].testspace.scalar_product(value)
            for k, coefficient in members:
                # Added, not assigned: an equation may have both an assembled
                # coupling and a term that has to be evaluated pointwise.
                results[k] = results[k] + (
                    projected
                    if coefficient == 1.0
                    else scale_real(projected, coefficient)
                )
        return tuple(results)

    @property
    def num_fields(self) -> int:
        """Return the number of coupled fields."""
        return len(self.integrators) + len(self.constraints)

    def _solvers(self) -> tuple[tuple[int, Any], ...]:
        """Return every equation's solver, paired with its global field slot."""
        return tuple(zip(self.transient_slots, self.integrators, strict=True)) + tuple(
            zip(self.constraint_slots, self.constraints, strict=True)
        )

    def common_padding(self, N: ScalarPadding = None) -> ScalarPadding:
        """Resolve one physical-space shape shared by every field.

        The nonlinear terms are evaluated pointwise, so all fields must be
        transformed onto the same mesh. Without an explicit `N` this is the
        per-axis maximum over the fields, which lets equations use spaces of
        different size: the smaller ones are simply evaluated on a finer mesh,
        and `scalar_product` truncates each back to its own coefficients.
        """
        if N is not None:
            return N
        shapes = [g.testspace.shape for _, g in self._solvers()]
        shapes += [g.trialspace.shape for _, g in self._solvers()]
        common = tuple(max(axis) for axis in zip(*shapes, strict=True))
        return common[0] if len(common) == 1 else common

    def _validate_common_mesh(self) -> None:
        """Require every field to live on one common quadrature mesh.

        Different bases, boundary conditions and resolutions are all fine -- but
        a pointwise product of fields sampled at different physical points is
        meaningless, so mixed quadrature families are rejected here rather than
        silently producing wrong answers.
        """
        N = self.common_padding()
        spaces: list[tuple[str, ScalarSpaceType]] = []
        for k, g in sorted(self._solvers()):
            spaces.append((f"equation {k} trial space", g.trialspace))
            spaces.append((f"equation {k} test space", g.testspace))

        _, reference = spaces[0]
        for label, V in spaces[1:]:
            # Equality, not identity: separately built spaces get separate
            # CoordSys objects, but equal ones share base scalars that SymPy
            # treats as the same symbols -- which is all the nonlinear terms need.
            if V.system != reference.system:
                raise ValueError(
                    f"{label} uses a different coordinate system than equation 0. "
                    "All fields of a system must share one, because the nonlinear "
                    "terms mix their base scalars. Pass the same `system=` to each "
                    "space (e.g. `TensorProduct(..., system=V.system)`)."
                )
            if V.dims != reference.dims:
                raise ValueError(
                    f"{label} has {V.dims} dimensions, expected {reference.dims}."
                )
            for axis, (mesh, ref_mesh) in enumerate(
                zip(mesh_axes(V, N), mesh_axes(reference, N), strict=True)
            ):
                if not bool(jnp.allclose(mesh, ref_mesh)):
                    raise NotImplementedError(
                        f"{label} does not share the quadrature mesh of equation 0 "
                        f"along axis {axis}. Nonlinear terms are evaluated "
                        "pointwise, so all fields must be sampled at the same "
                        "physical points; mixing quadrature families (or domains) "
                        "would require interpolation between meshes."
                    )

    def resolve_constraints(
        self,
        states: tuple[Array, ...],
        N: ScalarPadding = None,
        slots: tuple[int, ...] | None = None,
    ) -> tuple[Array, ...]:
        """Return `states` with every constrained field solved from the others.

        Constraints are resolved in equation declaration order, so one may read
        the field of a constraint declared before it. `slots` restricts the work
        to those global field slots, which initialization uses to leave a field
        the caller supplied alone.
        """
        out = list(states)
        for slot, c in zip(self.constraint_slots, self.constraints, strict=True):
            if slots is None or slot in slots:
                out[slot] = c.solve_field(tuple(out), N)
        return tuple(out)

    def _check_state_length(self, states: Sequence[Any], what: str) -> None:
        if len(states) != self.num_fields:
            raise ValueError(
                f"A system {what} must have one entry per equation "
                f"({self.num_fields}), got {len(states)}."
            )

    def initial_coefficients(
        self, initial: Sequence[sp.Expr | Array | None] | None = None
    ) -> tuple[Array, ...]:
        """Return coefficient-space data for the initial condition of each field.

        A constrained field given no initial condition is solved for once the
        transient fields are in place. One given a value keeps it: the two agree
        whenever the pair is consistent, and stating the field directly is often
        how the problem is posed -- a cavity started from rest is `psi = 0` with
        a moving lid, whose vorticity is then whatever the constraint says.
        """
        out: list[Array] = [jnp.zeros(())] * self.num_fields
        if initial is not None:
            initial = tuple(initial)
            self._check_state_length(initial, "initial condition")
        for slot, g in zip(self.transient_slots, self.integrators, strict=True):
            out[slot] = g.initial_coefficients(
                None if initial is None else initial[slot]
            )
        derive: list[int] = []
        for slot, c in zip(self.constraint_slots, self.constraints, strict=True):
            given = c.initial_coefficients(None if initial is None else initial[slot])
            if given is None:
                derive.append(slot)
                # A placeholder, so the tuple the constraints read is complete.
                out[slot] = jnp.zeros(c._state_shape)
            else:
                out[slot] = given
        return self.resolve_constraints(tuple(out), slots=tuple(derive))

    def _coerce_state(self, state0: IntegratorState) -> tuple[Array, ...]:
        """Coerce a restart state into one coefficient array per field.

        The constrained entries are recomputed from the transient ones rather
        than taken as given, so a restart cannot resume from a state where the
        two disagree.
        """
        if not isinstance(state0, tuple):
            raise TypeError(
                "A system restart state must be a tuple with one array per field, "
                f"got {type(state0).__name__}."
            )
        self._check_state_length(state0, "restart state")
        out = list(state0)
        for slot, g in zip(self.transient_slots, self.integrators, strict=True):
            out[slot] = g._coerce_state(state0[slot])
        for slot, c in zip(self.constraint_slots, self.constraints, strict=True):
            out[slot] = jnp.zeros(c._state_shape)
        return self.resolve_constraints(tuple(out))

    def _setup_impl(self, dt: float) -> None:
        """Precompute step-size-dependent data for every equation."""
        for g in self.integrators:
            g.setup(dt)
        for c in self.constraints:
            c.setup()
