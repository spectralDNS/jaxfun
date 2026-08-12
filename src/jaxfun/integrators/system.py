"""Time integration of systems of equations coupled through nonlinear terms.

Each equation of the system is advanced by its own scalar integrator, so the
implicit (linear) operators stay decoupled and are assembled and factorized once
per equation. The equations see each other only through their nonlinear terms,
which are evaluated explicitly from the state of *all* fields at the current
stage. Every field is represented by a single shared JAXFunction node, so
updating a stage makes it visible to all equations at once.
"""

from abc import ABC
from collections.abc import Sequence
from typing import cast, get_args, get_origin

import jax.numpy as jnp
import sympy as sp
from flax import nnx
from sympy.core.function import AppliedUndef

from jaxfun.coordinates import get_system
from jaxfun.galerkin import TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.typing import Array, IntegratorState, ScalarPadding, ScalarSpaceType
from jaxfun.utils import get_time_independent, split_time_derivative_terms

from .base import BaseIntegrator, TimeStepper
from .nonlinear import compile_coupled_nonlinear_evaluator


def _own_trial(equation: sp.Expr) -> TrialFunction:
    """Return the transient field an equation is the evolution equation for.

    The time-derivative term identifies it unambiguously; every other field
    appearing in the equation belongs to a different equation of the system.
    """
    t = get_system(equation).base_time()
    lhs, _ = split_time_derivative_terms(equation, t)
    if sp.sympify(lhs) == 0:
        raise ValueError(
            "Time integrators require a first-order time derivative in the weak form"
        )
    _, trial = get_basisfunctions(lhs)
    if not isinstance(trial, TrialFunction):
        raise ValueError(
            "The time-derivative term must contain exactly one TrialFunction. "
            "Equations coupled through their time derivatives are not supported."
        )
    return trial


def _coefficient_shape(V: ScalarSpaceType) -> tuple[int, ...]:
    """Return the coefficient-array shape for the given space."""
    num_dofs = V.num_dofs
    return num_dofs if isinstance(num_dofs, tuple) else (num_dofs,)


def _scale(array: Array, coefficient: complex) -> Array:
    """Multiply by a constant, keeping a real coefficient real.

    A Python `complex` would otherwise promote a real coefficient array to
    complex and silently double its memory traffic.
    """
    if coefficient.imag == 0.0:
        return coefficient.real * array
    return coefficient * array


def _mesh_axes(V: ScalarSpaceType, N: ScalarPadding) -> tuple[Array, ...]:
    """Return the per-axis quadrature mesh of ``V`` evaluated with padding ``N``."""
    if V.dims == 1:
        return (cast(Array, V.mesh(N=cast(int, N))),)  # ty: ignore[invalid-argument-type]
    mesh = V.mesh(N=N, broadcast=False)  # ty: ignore[invalid-argument-type,unknown-argument]
    return cast(tuple[Array, ...], mesh)


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
                        _coefficient_shape(cast(ScalarSpaceType, u.functionspace))
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

    Holds one scalar sub-integrator of type `integrator_type` per equation, plus
    the shared field registry that lets their nonlinear terms see each other.
    `IntegratorT` is that sub-integrator type, so subclasses keep access to the
    methods specific to it. Subclasses implement `step`.
    """

    integrator_type: type[IntegratorT]
    integrators: tuple[IntegratorT, ...]

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
        initial: Sequence[sp.Expr | Array],
        time: tuple[float, float] | None = None,
        **params,
    ):
        """Build one sub-integrator per equation over a shared set of fields.

        Args:
            equations: One weak form per equation, each with a first-order time
                derivative of its own transient TrialFunction.
            initial: One initial condition per equation, in the same order.
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
        trials = tuple(_own_trial(eq) for eq in equations)
        self.fields = nnx.static(FieldRegistry(trials))
        registry: FieldRegistry = self.fields

        self.integrators = nnx.data(
            tuple(
                self.integrator_type(
                    eq,
                    initial=init_k,
                    time=time,
                    explicit_trials=tuple(u for j, u in enumerate(trials) if j != k),
                    fields=registry.pairs,
                    field_order=registry.nodes,
                    field_index=k,
                    **params,
                )
                for k, (eq, init_k) in enumerate(zip(equations, initial, strict=True))
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

        As in `BaseIntegrator.nonlinear_rhs_scalar_product`, the mass inverse is
        deliberately not applied.
        """
        results: list[Array] = [g._no_nonlinear(states) for g in self.integrators]
        if self._coupled_nonlinear_evaluator is None:
            return tuple(results)
        values = self._coupled_nonlinear_evaluator(states, N)
        for (rep, members), value in zip(self._nonlinear_groups, values, strict=True):
            projected = self.integrators[rep].testspace.scalar_product(value)
            for k, coefficient in members:
                results[k] = (
                    projected if coefficient == 1.0 else _scale(projected, coefficient)
                )
        return tuple(results)

    @property
    def num_fields(self) -> int:
        """Return the number of coupled fields."""
        return len(self.integrators)

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
        shapes = [g.testspace.shape for g in self.integrators]
        shapes += [g.trialspace.shape for g in self.integrators]
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
        for k, g in enumerate(self.integrators):
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
                zip(_mesh_axes(V, N), _mesh_axes(reference, N), strict=True)
            ):
                if not bool(jnp.allclose(mesh, ref_mesh)):
                    raise NotImplementedError(
                        f"{label} does not share the quadrature mesh of equation 0 "
                        f"along axis {axis}. Nonlinear terms are evaluated "
                        "pointwise, so all fields must be sampled at the same "
                        "physical points; mixing quadrature families (or domains) "
                        "would require interpolation between meshes."
                    )

    def initial_coefficients(
        self, initial: Sequence[sp.Expr | Array] | None = None
    ) -> tuple[Array, ...]:
        """Return coefficient-space data for the initial condition of each field."""
        if initial is None:
            return tuple(g.initial_coefficients() for g in self.integrators)
        return tuple(
            g.initial_coefficients(init_k)
            for g, init_k in zip(self.integrators, initial, strict=True)
        )

    def _coerce_state(self, state0: IntegratorState) -> tuple[Array, ...]:
        """Coerce a restart state into one coefficient array per field."""
        if not isinstance(state0, tuple):
            raise TypeError(
                "A system restart state must be a tuple with one array per field, "
                f"got {type(state0).__name__}."
            )
        return tuple(
            g._coerce_state(s) for g, s in zip(self.integrators, state0, strict=True)
        )

    def setup(self, dt: float) -> None:
        """Precompute step-size-dependent data for every equation."""
        for g in self.integrators:
            g.setup(dt)
