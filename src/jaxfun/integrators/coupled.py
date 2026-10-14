"""Monolithic IMEX Runge-Kutta integration of systems over mixed spaces.

`SystemIMEXRungeKutta` advances each equation of a system on its own and sees
the others only explicitly. That cannot hold a constraint such as continuity,
whose pressure is a Lagrange multiplier with no equation of its own. Here the
whole weak form, over the test and trial functions of one mixed
(`CartesianProduct`) space, is assembled into block operators,

    M dX/dt = L X + N(X) + f,

and every implicit stage is one block solve for all fields at once. A field
without a time derivative, such as the pressure, is algebraic: its columns of
`M` are empty, and the stage solves enforce its equation at every stage.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.coordinates import get_system
from jaxfun.galerkin import (
    CartesianProductSpace,
    CartesianTensorProductSpace,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import project
from jaxfun.la import BaseMatrix, BlockArray, BlockMatrix, GlobalMatrix, TPMatrix
from jaxfun.la.blockwavenumber import block_lu_factor
from jaxfun.sharding import replicate
from jaxfun.typing import Array, RankTag, ScalarPadding
from jaxfun.utils import get_time_independent, split_time_derivative_terms
from jaxfun.utils.operator_tools import assemble_block_term
from jaxfun.utils.sympy_factoring import (
    drop_time_argument,
    split_linear_couplings,
    time_derivative_as_operator,
)

from ._utils import (
    apply_blocks,
    boundary_values,
    component_offset,
    mesh_axes,
    validate_solver_options,
)
from .base import TimeStepper
from .imex_rk import IMEXStages
from .nonlinear import compile_block_nonlinear_evaluator
from .tableau import IMEXTableau

type Pin = tuple[int, tuple[int, ...]]
type BlockTerm = TPMatrix | GlobalMatrix
type MixedSpace = CartesianTensorProductSpace | CartesianProductSpace

# Solver options that the global sparse block solve understands.
_GLOBAL_SOLVE_OPTIONS = frozenset({"method", "auto_threshold"})


class CoupledIMEXRungeKutta(IMEXStages, TimeStepper[tuple[Array, ...]]):
    """Diagonally-implicit IMEX Runge-Kutta for a coupled system over a mixed space.

    Every term linear in a single field is implicit, wherever it couples
    (pressure gradient, continuity, buoyancy alike), and everything else is
    explicit. The state holds one coefficient array per flattened component
    of the mixed trial space, so a vector field contributes one per component.

    Example, Rayleigh-Benard convection::

        W = CartesianProduct(V, Q, VT)  # V = CartesianProduct(VD, VD, rank=1)
        u, p, T = TrialFunction(W, name="upT", transient=True)
        v, q, s = TestFunction(W, name="vqs")
        equations = (
            Dot(u.diff(t), v)
            + Dot(Dot(Grad(u), u), v)
            - nu * Dot(Div(Grad(u)), v)
            - T * Dot(j, v)
            + Dot(Grad(p), v),
            Div(u) * q,
            (T.diff(t) + Div(u * T) - kappa * Div(Grad(T))) * s,
        )
        rb = CoupledIMEXRungeKutta(
            equations, tableau=ARS443, initial=(0, None, T0), sparse=True
        )
        ux, uy, p_hat, T_hat = rb.solve(dt, steps)

    A system with algebraic fields needs a globally stiffly accurate tableau
    whose first stage is explicit (`ARS443`, `ARS222`, `IMEX_EULER`). The
    accepted solution is then the last stage, solved with the constraints in
    force, and the initial values of the algebraic fields never enter.

    An algebraic field such as the pressure is not integrated in time. Every
    stage determines it afresh, as the Lagrange multiplier of its constraint,
    and the value returned is the last stage's. Where that value balances
    explicit terms, which the stage takes from earlier stage times, it is only
    a first-order approximation of the pressure at the time of the returned
    velocity. The pressure is a function of the velocity alone, though, and
    `consistent_algebraic` solves for it from a state, to the velocity's order.
    """

    def __init__(
        self,
        equations: sp.Expr | Sequence[sp.Expr],
        *,
        tableau: IMEXTableau,
        initial: Sequence[Any],
        time: tuple[float, float] | None = None,
        pin: Mapping[TrialFunction, Any] | None = None,
        sparse: bool = False,
        sparse_tol: int = 1000,
        solver_options: Mapping[str, Any] | None = None,
    ):
        """Assemble a coupled system.

        Args:
            equations: The weak form, as one expression or several (one per
                test function, say) that are summed. Its test and trial
                functions are components of mixed spaces, e.g.
                ``u, p = TrialFunction(W, transient=True)`` and
                ``v, q = TestFunction(W)``.
            tableau: IMEX Butcher tableau.
            initial: One initial condition per top-level field of the mixed
                trial space: an expression (a vector expression for a vector
                field), coefficients, or None for zero. An expression is the
                whole field, boundary values included; coefficients, like the
                state, exclude the boundary lifting. An algebraic field is
                usually given None.
            time: Optional default integration interval.
            pin: Maps a scalar field to a coefficient index, or a sequence of
                them, held at zero. A mode no equation determines -- the
                constant pressure mode of a channel, say -- is detected and
                pinned automatically when its row and column are both empty, so
                this is only needed for a nullspace that is not structural.
            sparse: Assemble sparse operators when possible.
            sparse_tol: Sparsification tolerance passed to Galerkin assembly.
            solver_options: ``method`` picks how each Fourier wavenumber's
                block system is factorized, ``"dense"``, ``"rcm"`` (banded) or
                ``"auto"``, the default; see `block_lu_factor`. Without Fourier
                structure, ``method`` and ``auto_threshold`` go to the global
                sparse solve.
        """
        terms: Sequence[sp.Expr] = (
            (equations,) if isinstance(equations, sp.Expr) else tuple(equations)
        )
        form = sp.expand(sp.Add(*terms))
        t = get_system(form).base_time()

        self.time = time
        self.tableau = nnx.static(tableau)
        # Coefficient arrays are data, and the rest -- expressions, zeros, None
        # -- static, so the two are held apart; `initial_coefficients` merges them.
        given = tuple(_is_array_initial(v) for v in initial)
        self._initial_static = nnx.static(
            tuple(None if a else v for v, a in zip(initial, given, strict=True))
        )
        self._initial_arrays = nnx.data(
            tuple(v if a else None for v, a in zip(initial, given, strict=True))
        )
        options = validate_solver_options(solver_options)
        self._solve_options = nnx.static(
            tuple((k, v) for k, v in options if k in _GLOBAL_SOLVE_OPTIONS)
        )

        fields = _fields(drop_time_argument(form, t))
        trial_space = cast(MixedSpace, fields[0].functionspace.leaf)
        self.fields = nnx.static(fields)
        self.trial_space = nnx.static(trial_space)

        lhs, rhs = split_time_derivative_terms(form, t)
        mass = _mass_expression(lhs, t)
        # The weak form reads `M dX/dt + rest = 0`, so the right-hand side is
        # `-rest`: its terms linear in one field assemble into `L` and `f`, and
        # what is left is the explicit `N(X)`.
        couplings, remaining = split_linear_couplings(-rhs, fields)
        explicit = [
            term for term in sp.Add.make_args(remaining) if term.atoms(TrialFunction)
        ]
        linear = sp.Add(*couplings.values()) + (remaining - sp.Add(*explicit))
        nonlinear = sp.Add(*explicit)
        _check_steady(linear, trial_space, t)

        mass_terms, mass_forcing, mass_spaces = _assemble_blocks(
            mass, sparse, sparse_tol
        )
        linear_terms, forcing, linear_spaces = _assemble_blocks(
            linear, sparse, sparse_tol
        )
        spaces = mass_spaces or linear_spaces
        assert spaces is not None  # the mass term always assembles an operator
        test_space = spaces[0]
        if mass_forcing is not None and not any(
            boundary_values(V) for V in trial_space.flatten()
        ):
            raise ValueError(
                "Time-derivative assembly produced forcing, but no field has "
                "boundary conditions to explain it"
            )
        # A steady lifting `B` sits under the time derivative as `inner(v, B)`,
        # which differentiates away, so the mass forcing is dropped.
        self.test_space = nnx.static(test_space)
        self._test_dofs = nnx.static(_dofs(test_space))
        self._trial_dofs = nnx.static(_dofs(trial_space))

        self._mass_slots = nnx.static(tuple(_slot(m) for m in mass_terms))
        self._mass_terms: tuple[BlockTerm, ...] = nnx.data(tuple(mass_terms))
        self._linear_slots = nnx.static(tuple(_slot(m) for m in linear_terms))
        self._linear_terms: tuple[BlockTerm, ...] = nnx.data(tuple(linear_terms))
        self._forcing: tuple[Array, ...] | None = nnx.data(
            None if forcing is None else tuple(replicate(a) for a in forcing.array)
        )
        self._dtype = nnx.static(
            jnp.result_type(float, *(_operator(m).dtype for m in mass_terms))
        )
        self._pins = nnx.static(_resolve_pins(pin or {}, fields))

        mass_columns = {j for _, j in self._mass_slots}
        self.algebraic = nnx.static(
            tuple(c for c in range(len(self._trial_dofs)) if c not in mass_columns)
        )
        self._mass_solver = nnx.data(None)
        self._algebraic_solver = nnx.static(_Lazy())
        if self.algebraic:
            _check_tableau_for_constraints(tableau)
        else:
            self._mass_solver = nnx.data(
                block_lu_factor(
                    mass_terms, test_space, trial_space, **dict(self._solve_options)
                )
            )

        self._nonlinear_components: tuple[int, ...] = nnx.static(())
        self._nonlinear_evaluator: (
            Callable[[tuple[Array, ...], ScalarPadding], tuple[Array, ...]] | None
        ) = None
        if sp.sympify(nonlinear) != 0:
            components, used, evaluator = compile_block_nonlinear_evaluator(
                nonlinear, fields, test_space, trial_space.flatten()[0]
            )
            self._nonlinear_components = nnx.static(components)
            self._nonlinear_evaluator = evaluator
            _validate_common_mesh(
                [test_space.flatten()[c] for c in components]
                + [trial_space.flatten()[c] for c in used],
                self.common_padding(),
            )

    def common_padding(self, N: ScalarPadding = None) -> ScalarPadding:
        """Resolve the physical-space shape the nonlinear terms are evaluated on.

        Without an explicit `N` this is the per-axis maximum over every
        component's space, so components of different size share one mesh.
        """
        if N is not None:
            return N
        spaces = [*self.trial_space.flatten(), *self.test_space.flatten()]
        shapes = [tuple(V.shape) for V in spaces]
        common = tuple(max(axis) for axis in zip(*shapes, strict=True))
        return common[0] if len(common) == 1 else common

    def apply_mass(self, uh: Sequence[Array]) -> tuple[Array, ...]:
        """Apply the block mass operator; algebraic rows are zero."""
        ops = tuple(_operator(m) for m in self._mass_terms)
        return apply_blocks(self._mass_slots, ops, uh, self._test_dofs)

    def apply_mass_inverse(self, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve the block mass operator, which exists without algebraic fields."""
        if self._mass_solver is None:
            raise RuntimeError(
                "The mass operator of a system with algebraic fields is singular"
            )
        return self._mass_solver.solve(rhs)

    def apply_linear(self, uh: Sequence[Array]) -> tuple[Array, ...]:
        """Apply the block linear operator."""
        ops = tuple(_operator(m) for m in self._linear_terms)
        return apply_blocks(self._linear_slots, ops, uh, self._test_dofs)

    def stage_solve(self, a_ii: float, rhs: Sequence[Array]) -> tuple[Array, ...]:
        """Solve `(M - dt a_ii L) X = rhs` for every field at once."""
        return self._stage_solvers[self._stage_keys.index(a_ii)].solve(rhs)

    def nonlinear_rhs_scalar_product(
        self,
        uh: Sequence[Array],
        N: ScalarPadding = None,
        t: Array | float | None = None,
    ) -> tuple[Array, ...]:
        """Return the explicit terms, one array per test component."""
        out: list[Array | None] = [None] * len(self._test_dofs)
        if self._nonlinear_evaluator is not None:
            values = self._nonlinear_evaluator(tuple(uh), self.common_padding(N))
            spaces = self.test_space.flatten()
            for c, value in zip(self._nonlinear_components, values, strict=True):
                out[c] = spaces[c].scalar_product(value)
        dtype = jnp.result_type(*uh)
        return tuple(
            jnp.zeros(shape, dtype=dtype) if o is None else o
            for o, shape in zip(out, self._test_dofs, strict=True)
        )

    def consistent_algebraic(
        self, state: Sequence[Array], N: ScalarPadding = None
    ) -> tuple[Array, ...]:
        """Return ``state`` with its algebraic fields solved for from the others.

        The algebraic fields a step returns, such as the pressure, are its last
        stage's Lagrange multipliers, and only first-order approximations at the
        step's time. These are the multipliers that make the time derivative of
        the transient fields satisfy the differentiated constraints, at the
        state's own time, and so as accurate as the transient fields are.

        The operator is factorized on the first call and kept; the compiled
        steps are unaffected. Call it outside `jax.jit`.

        Args:
            state: One coefficient array per component, as `solve` returns.
            N: Padding for the nonlinear terms, as passed to `solve`.

        Returns:
            ``state``, its algebraic components replaced.
        """
        state = tuple(state)
        if not self.algebraic:
            return state
        constraints = self._constraint_rows()
        if self._algebraic_solver.value is None:
            # The mass terms and every linear term that meets a constraint row
            # or an algebraic column: what the time derivatives and the
            # multipliers solve together, with the same pins as a stage.
            coupling = [
                m.scale(-1.0)
                for m, (i, j) in zip(
                    self._linear_terms, self._linear_slots, strict=True
                )
                if i in constraints or j in self.algebraic
            ]
            self._algebraic_solver.value = block_lu_factor(
                list(self._mass_terms) + coupling,
                self.test_space,
                self.trial_space,
                self._pins,
                **dict(self._solve_options),
            )
        transient = tuple(
            jnp.zeros_like(a) if c in self.algebraic else a for c, a in enumerate(state)
        )
        linear = self.apply_linear(transient)
        explicit = self.nonlinear_rhs_scalar_product(state, N)
        forcing = self._forcing
        rhs = tuple(
            jnp.zeros_like(linear[i])
            if i in constraints
            else linear[i] + explicit[i] + (0 if forcing is None else forcing[i])
            for i in range(len(linear))
        )
        x = self._algebraic_solver.value.solve(rhs)
        return tuple(x[c] if c in self.algebraic else a for c, a in enumerate(state))

    def _constraint_rows(self) -> tuple[int, ...]:
        """Return the test components without a time derivative: the constraints."""
        mass_rows = {i for i, _ in self._mass_slots}
        return tuple(i for i in range(len(self._test_dofs)) if i not in mass_rows)

    def _forcing_caches(
        self, t: Array | float, dt: float
    ) -> tuple[tuple[Array, ...] | None, None]:
        return self._forcing, None

    def _explicit_first_stage(
        self,
        u_hat: tuple[Array, ...],
        m_u: tuple[Array, ...],
        dt: float,
        forcing: Any,
        forcing_stage: Any,
    ) -> tuple[Array, ...]:
        return u_hat

    def _setup_impl(self, dt: float) -> None:
        """Factorize one stage operator `M - dt a_ii L` per diagonal coefficient."""
        tableau: IMEXTableau = self.tableau
        keys = tuple(a for a in tableau.distinct_diagonal_coeffs if a != 0.0)
        solvers = tuple(
            block_lu_factor(
                list(self._mass_terms)
                + [m.scale(-dt * a_ii) for m in self._linear_terms],
                self.test_space,
                self.trial_space,
                self._pins,
                **dict(self._solve_options),
            )
            for a_ii in keys
        )
        self._stage_keys = nnx.static(keys)
        self._stage_solvers = nnx.data(solvers)

    def initial_coefficients(
        self, initial: Sequence[Any] | None = None, t: float | None = None
    ) -> tuple[Array, ...]:
        """Return the coefficients of an initial condition, one array per component.

        `initial` has one entry per top-level field, as in the constructor.
        """
        init = (
            tuple(
                s if a is None else a
                for s, a in zip(self._initial_static, self._initial_arrays, strict=True)
            )
            if initial is None
            else tuple(initial)
        )
        if len(init) != len(self.fields):
            raise ValueError(
                f"Got {len(init)} initial conditions for {len(self.fields)} fields."
            )
        out: list[Array] = []
        for field, value in zip(self.fields, init, strict=True):
            V = field.functionspace
            dofs = (
                _dofs(V)
                if V.rank == RankTag.VECTOR
                else (_shape(cast(int | tuple[int, ...], V.num_dofs)),)
            )
            if value is None or (
                isinstance(value, sp.Basic | int | float) and value == 0
            ):
                out.extend(jnp.zeros(nd) for nd in dofs)
            elif isinstance(value, sp.Basic):
                projected = project(value, V)
                out.extend(projected if isinstance(projected, tuple) else (projected,))
            else:
                arrays = value if isinstance(value, tuple | list) else (value,)
                out.extend(
                    jnp.asarray(a).reshape(nd)
                    for a, nd in zip(arrays, dofs, strict=True)
                )
        return self._coerce_state(tuple(out), t)

    def _coerce_state(
        self, state0: Sequence[Array], t: float | None = None
    ) -> tuple[Array, ...]:
        """Coerce a restart state into one array per flattened component."""
        state0 = tuple(state0)
        if len(state0) != len(self._trial_dofs):
            raise ValueError(
                f"A coupled state has one array per component "
                f"({len(self._trial_dofs)}), got {len(state0)}."
            )
        dtype = jnp.result_type(self._dtype, *state0)
        return tuple(
            jnp.asarray(a).reshape(nd).astype(dtype)
            for a, nd in zip(state0, self._trial_dofs, strict=True)
        )


def _shape(num_dofs: int | tuple[int, ...]) -> tuple[int, ...]:
    return num_dofs if isinstance(num_dofs, tuple) else (num_dofs,)


def _dofs(space: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(_shape(nd) for nd in space.num_dofs)


def _slot(term: BlockTerm) -> tuple[int, int]:
    i, j = term.global_indices
    return int(i), int(j)


def _operator(term: BlockTerm) -> BaseMatrix:
    """Return the matrix a block term applies to its trial component."""
    return term.data if isinstance(term, GlobalMatrix) else term


def _fields(form: sp.Expr) -> tuple[TrialFunction, ...]:
    """Return the top-level trial functions of the form, in mixed-space order."""
    atoms = form.atoms(TrialFunction)
    leaves = {id(u.functionspace.leaf): u.functionspace.leaf for u in atoms}
    if len(leaves) != 1 or None in leaves.values():
        raise ValueError(
            "Every TrialFunction of a coupled system must be a component of one "
            "mixed space, e.g. `u, p = TrialFunction(CartesianProduct(V, Q))`."
        )
    leaf = cast(MixedSpace, next(iter(leaves.values())))

    fields = tuple(sorted(atoms, key=lambda u: component_offset(u.functionspace)))
    if len(fields) != len(leaf.basespaces):
        raise ValueError(
            f"The weak form holds {len(fields)} of the {len(leaf.basespaces)} "
            "fields of its mixed space; every field needs an equation."
        )
    return fields


def _mass_expression(lhs: sp.Expr, t: sp.Symbol) -> sp.Expr:
    """Turn every time-derivative term into the operator it applies."""
    if sp.sympify(lhs) == 0:
        raise ValueError(
            "Time integrators require a first-order time derivative in the weak form"
        )
    mass = sp.S.Zero
    for term in sp.Add.make_args(lhs):
        _, trial = get_basisfunctions(term)
        if not isinstance(trial, TrialFunction):
            raise ValueError(
                f"Each time-derivative term must hold exactly one field, got {term}"
            )
        mass += time_derivative_as_operator(term, trial, t)
    return mass


def _check_steady(linear: sp.Expr, trial_space: Any, t: sp.Symbol) -> None:
    """Refuse time-dependent sources and boundary data, not yet supported here."""
    if linear.has(t) or any(
        val.has(t) for V in trial_space.flatten() for val in boundary_values(V)
    ):
        raise NotImplementedError(
            "CoupledIMEXRungeKutta does not yet support time-dependent sources "
            "or boundary data."
        )


def _assemble_blocks(
    expr: sp.Expr, sparse: bool, sparse_tol: int
) -> tuple[list[BlockTerm], BlockArray | None, tuple[Any, Any] | None]:
    """Assemble a weak form over mixed spaces, one test function at a time.

    `inner` takes one test function per call, so the terms are grouped by it and
    the resulting block terms and forcing summed. Returns the terms, the forcing
    and the `(test, trial)` spaces.
    """
    groups: dict[TestFunction, list[sp.Expr]] = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        if term == 0:
            continue
        test, _ = get_basisfunctions(term)
        if not isinstance(test, TestFunction):
            raise ValueError(f"Each term must hold exactly one test function: {term}")
        groups.setdefault(test, []).append(term)

    terms: list[BlockTerm] = []
    forcing: BlockArray | None = None
    spaces: tuple[Any, Any] | None = None
    for group in groups.values():
        operator, f = assemble_block_term(
            sp.Add(*group), sparse=sparse, sparse_tol=sparse_tol
        )
        if operator is not None:
            if not isinstance(operator, BlockMatrix):
                raise ValueError(
                    "A coupled system needs test and trial functions of mixed "
                    "(CartesianProduct) spaces."
                )
            terms.extend(cast(list[BlockTerm], list(operator.mats)))
            spaces = (operator.test_space, operator.trial_space)
        if f is not None:
            forcing = f if forcing is None else forcing + f
    return terms, forcing, spaces


def _resolve_pins(
    pin: Mapping[TrialFunction, Any], fields: Sequence[TrialFunction]
) -> tuple[Pin, ...]:
    """Turn `{field: index or indices}` into `(flattened component, index)` pairs."""
    pins: list[Pin] = []
    for key, value in pin.items():
        field = get_time_independent(key)
        if not any(f == field for f in fields):
            raise ValueError(f"Cannot pin {key}: it is not a field of the system.")
        if field.functionspace.rank == RankTag.VECTOR:
            raise ValueError(f"Cannot pin the vector field {key}; pin a scalar field.")
        indices = [value] if isinstance(value[0], int) else list(value)
        c = component_offset(field.functionspace)
        pins.extend((c, tuple(int(i) for i in index)) for index in indices)
    return tuple(pins)


def _check_tableau_for_constraints(tableau: IMEXTableau) -> None:
    """Require a tableau that keeps algebraic fields exact without `M^-1`."""
    A = tableau.implicit.A
    s = tableau.stages
    ok = (
        tableau.is_stiffly_accurate
        and all(A[i][i] != 0.0 for i in range(1, s))
        and (A[0][0] != 0.0 or all(A[i][0] == 0.0 for i in range(s)))
    )
    if not ok:
        raise ValueError(
            "A system with algebraic fields (no time derivative) needs a "
            "globally stiffly accurate tableau whose implicit stages after the "
            "first all have a nonzero diagonal and an empty first column, such "
            "as ARS443, ARS222 or IMEX_EULER."
        )


class _Lazy:
    """A slot filled on first use, outside any trace.

    Held as static metadata and compared by its own identity, not its content's,
    so filling it leaves the stepper's pytree structure -- and with it every
    compiled step -- unchanged.
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: Any = None


def _is_array_initial(value: Any) -> bool:
    """Whether an initial value is given as coefficients rather than symbolically."""
    return value is not None and not isinstance(value, sp.Basic | int | float)


def _validate_common_mesh(spaces: Sequence[Any], N: ScalarPadding) -> None:
    """Require the components the nonlinear terms mix to share one mesh."""
    reference = mesh_axes(spaces[0], N)
    for V in spaces[1:]:
        for axis, (mesh, ref) in enumerate(
            zip(mesh_axes(V, N), reference, strict=True)
        ):
            if mesh.shape != ref.shape or not bool(jnp.allclose(mesh, ref)):
                raise NotImplementedError(
                    f"{V.name} does not share the quadrature mesh of {spaces[0].name} "
                    f"along axis {axis}. Nonlinear terms are evaluated pointwise, so "
                    "the components they mix must be sampled at the same points."
                )
