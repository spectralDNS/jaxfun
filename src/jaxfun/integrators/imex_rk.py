"""Diagonally-implicit IMEX Runge-Kutta time integration."""

from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.la import BaseMatrix
from jaxfun.typing import Array, ScalarPadding

from ._utils import solve_with_options, warm_operator_solve_cache
from .base import BaseIntegrator
from .system import SystemIntegrator
from .tableau import IMEXTableau


class IMEXRungeKutta(BaseIntegrator):
    """Diagonally-implicit IMEX Runge-Kutta integrator for semilinear systems.

    The stiff linear part of the weak form is solved implicitly (one linear
    solve per stage with nonzero diagonal Butcher coefficient), while the
    nonlinear part is evaluated explicitly, combined stage by stage according
    to `tableau`.
    """

    def __init__(
        self,
        equation: sp.Expr,
        *,
        tableau: IMEXTableau,
        initial: sp.Expr | Array,
        time: tuple[float, float] | None = None,
        **params,
    ):
        """Construct an IMEX Runge-Kutta integrator for a semilinear weak form."""
        super().__init__(equation, initial=initial, time=time, **params)
        self.tableau = nnx.static(tableau)

    def _setup_impl(self, dt: float) -> None:
        """Precompute one implicit system operator per distinct diagonal coefficient."""
        tableau: IMEXTableau = self.tableau
        ops: list[BaseMatrix] = []
        for a_ii in tableau.distinct_diagonal_coeffs:
            op = self.mass_operator - dt * a_ii * self.linear_operator
            warm_operator_solve_cache(op, self._state_shape, self._solver_options)
            ops.append(op)
        self._stage_operators: tuple[BaseMatrix, ...] = nnx.data(tuple(ops))

    def _stage_operator(self, a_ii: float) -> BaseMatrix:
        """Return the cached implicit system operator for diagonal coeff `a_ii`."""
        idx = self.tableau.distinct_diagonal_coeffs.index(a_ii)
        return self._stage_operators[idx]

    def stage(
        self,
        i: int,
        m_u: Array,
        dt: float,
        nonlinear_stage: list[Array | None],
        linear_stage: list[Array | None],
        forcing: Array | None,
    ) -> Array:
        """Compute stage `i` from the caches already populated for `j < i`.

        `nonlinear_stage`/`linear_stage` are read-only here (entries for
        `j < i` must already be populated by the caller); this method does
        not append to them -- that bookkeeping is `step`-specific (it depends
        on `step`'s choice of final-combination path) and stays there.
        """
        tableau: IMEXTableau = self.tableau
        a_e, a_i = tableau.explicit.A, tableau.implicit.A
        c_i = tableau.implicit.c

        a_ii = a_i[i][i]
        rhs = m_u
        for j in range(i):
            if a_e[i][j] != 0.0:
                rhs = rhs + dt * a_e[i][j] * cast(Array, nonlinear_stage[j])
            if a_i[i][j] != 0.0:
                rhs = rhs + dt * a_i[i][j] * cast(Array, linear_stage[j])
        if forcing is not None and c_i[i] != 0.0:
            rhs = rhs + dt * c_i[i] * forcing

        if a_ii == 0.0:
            return self.apply_mass_inverse(rhs)
        return solve_with_options(self._stage_operator(a_ii), rhs, self._solver_options)

    def _step_impl(self, u_hat: Array, dt: float, N: ScalarPadding = None) -> Array:
        """Advance one IMEX Runge-Kutta step in coefficient space.

        Three final-combination paths, selected by `tableau`'s (static)
        stiff-accuracy properties:

        - Globally stiffly accurate (`is_stiffly_accurate`): the last stage
          already equals the accepted solution; return it directly.
        - Implicit-only stiffly accurate (`implicit_is_stiffly_accurate`
          without the former): the ``b_i``-weighted combination folds
          algebraically into the last stage, leaving only a
          ``(b_e - explicit.A[-1])``-weighted correction over the cached
          nonlinear stage values (no linear-operator terms needed at all).
        - Otherwise: the general weighted combination over both ``b_e`` and
          ``b_i``.

        The last stage's nonlinear/linear evaluations are skipped entirely
        when the chosen path doesn't need them (never needed for the fully
        stiffly-accurate path; the linear evaluation is additionally never
        needed for the implicit-only path).
        """
        tableau: IMEXTableau = self.tableau
        a_e = tableau.explicit.A
        b_e, b_i = tableau.explicit.b, tableau.implicit.b

        full_gsa = tableau.is_stiffly_accurate
        implicit_only_sa = (not full_gsa) and tableau.implicit_is_stiffly_accurate
        last = tableau.stages - 1

        m_u = self.apply_mass(u_hat)
        forcing = (
            jnp.asarray(self.linear_forcing)
            if self.linear_forcing is not None
            else None
        )

        stages: list[Array] = []
        nonlinear_stage: list[Array | None] = []
        linear_stage: list[Array | None] = []

        for i in range(tableau.stages):
            stage = self.stage(i, m_u, dt, nonlinear_stage, linear_stage, forcing)
            stages.append(stage)
            is_last = i == last
            skip_nonlinear = is_last and full_gsa
            skip_linear = is_last and (full_gsa or implicit_only_sa)
            nonlinear_stage.append(
                None if skip_nonlinear else self.nonlinear_rhs_scalar_product(stage, N)
            )
            linear_stage.append(None if skip_linear else (self.linear_operator @ stage))

        if full_gsa:
            return stages[-1]

        if implicit_only_sa:
            rhs = self.apply_mass(stages[-1])
            for j in range(tableau.stages):
                weight = b_e[j] - a_e[-1][j]
                if weight != 0.0:
                    rhs = rhs + dt * weight * cast(Array, nonlinear_stage[j])
            return self.apply_mass_inverse(rhs)

        rhs = m_u
        for j in range(tableau.stages):
            if b_e[j] != 0.0:
                rhs = rhs + dt * b_e[j] * cast(Array, nonlinear_stage[j])
            if b_i[j] != 0.0:
                rhs = rhs + dt * b_i[j] * cast(Array, linear_stage[j])
        if forcing is not None:
            rhs = rhs + dt * forcing
        return self.apply_mass_inverse(rhs)


class SystemIMEXRungeKutta(SystemIntegrator[IMEXRungeKutta]):
    """IMEX Runge-Kutta integration of equations coupled through nonlinear terms.

    Every evolution equation is advanced by its own `IMEXRungeKutta`, so each
    keeps a separate implicit operator and needs one linear solve per stage. The
    equations are marched through the stages together: within a stage all fields
    are solved for first (their implicit systems are mutually independent), and
    only then are the nonlinear terms evaluated, so each sees every field at the
    same stage.

    Constraint equations are solved in between those two: after the transient
    fields of a stage are known and before anything is evaluated from them. A
    constrained field is therefore never lagged -- at every stage it is the
    exact solution of its equation for the transient fields at that same stage.
    """

    def __init__(
        self,
        equations: Sequence[sp.Expr],
        *,
        tableau: IMEXTableau,
        initial: Sequence[sp.Expr | Array | None],
        time: tuple[float, float] | None = None,
        **params,
    ):
        """Construct an IMEX Runge-Kutta integrator for a coupled system."""
        super().__init__(
            equations, initial=initial, time=time, tableau=tableau, **params
        )
        self.tableau = nnx.static(tableau)

    def _step_impl(
        self, u_hats: tuple[Array, ...], dt: float, N: ScalarPadding = None
    ) -> tuple[Array, ...]:
        """Advance every field one IMEX Runge-Kutta step in coefficient space.

        Mirrors `IMEXRungeKutta.step`, including its three final-combination
        paths, but applies each of them per evolved field. The stage loop is
        what couples the equations; see the comments inside it.

        The state carries one array per equation, constrained fields included,
        in declaration order. Only the transient entries are combined into the
        accepted solution; the constrained ones are re-solved from it at the
        end, which is the only correct choice for the two paths whose accepted
        solution is not simply the last stage.
        """
        tableau: IMEXTableau = self.tableau
        a_e = tableau.explicit.A
        b_e, b_i = tableau.explicit.b, tableau.implicit.b

        full_gsa = tableau.is_stiffly_accurate
        implicit_only_sa = (not full_gsa) and tableau.implicit_is_stiffly_accurate
        last = tableau.stages - 1

        integrators = self.integrators
        slots = self.transient_slots
        # One shared padding for every field: the nonlinear terms are evaluated
        # pointwise, so all fields must land on the same physical mesh.
        M = self.common_padding(N)

        m_u = tuple(
            g.apply_mass(u_hats[slot])
            for g, slot in zip(integrators, slots, strict=True)
        )
        forcing = tuple(
            jnp.asarray(g.linear_forcing) if g.linear_forcing is not None else None
            for g in integrators
        )

        stages: list[list[Array]] = [[] for _ in integrators]
        nonlinear_stage: list[list[Array | None]] = [[] for _ in integrators]
        linear_stage: list[list[Array | None]] = [[] for _ in integrators]
        state_i: tuple[Array, ...] = u_hats

        for i in range(tableau.stages):
            # The implicit solves are mutually independent, so every transient
            # field's stage `i` is computed first, then the constrained fields
            # are solved from them. The tuple must be complete before any
            # nonlinear evaluation below, otherwise an equation would see a
            # mixture of stages `i` and `i-1`.
            stage_i = list(u_hats)
            for k, (g, slot) in enumerate(zip(integrators, slots, strict=True)):
                stage_i[slot] = g.stage(
                    i, m_u[k], dt, nonlinear_stage[k], linear_stage[k], forcing[k]
                )
            state_i = self.resolve_constraints(tuple(stage_i), M)
            is_last = i == last
            skip_nonlinear = is_last and full_gsa
            skip_linear = is_last and (full_gsa or implicit_only_sa)
            # Evaluated for the whole system at once: this is what pushes all
            # stage-`i` coefficients into the shared JAXFunction nodes, and it
            # transforms each field to physical space once for all equations.
            reaction = (
                None if skip_nonlinear else self.nonlinear_scalar_products(state_i, M)
            )
            for k, (g, slot) in enumerate(zip(integrators, slots, strict=True)):
                stages[k].append(state_i[slot])
                nonlinear_stage[k].append(None if reaction is None else reaction[k])
                linear_stage[k].append(
                    None if skip_linear else (g.linear_operator @ state_i[slot])
                )

        def accepted(evolved: Sequence[Array]) -> tuple[Array, ...]:
            """Scatter the evolved fields back, then re-solve the constraints.

            Needed for the two paths whose accepted solution is a weighted
            combination rather than the last stage: the constrained fields have
            to follow the combination, not the stage they were last solved at.
            """
            out = list(u_hats)
            for slot, value in zip(slots, evolved, strict=True):
                out[slot] = value
            return self.resolve_constraints(tuple(out), M)

        if full_gsa:
            # The last stage *is* the accepted solution, so its constrained
            # fields were already solved from exactly these transient values.
            # Re-resolving them would recompute the same answer.
            return state_i

        if implicit_only_sa:
            evolved: list[Array] = []
            for k, g in enumerate(integrators):
                rhs = g.apply_mass(stages[k][-1])
                for j in range(tableau.stages):
                    weight = b_e[j] - a_e[-1][j]
                    if weight != 0.0:
                        rhs = rhs + dt * weight * cast(Array, nonlinear_stage[k][j])
                evolved.append(g.apply_mass_inverse(rhs))
            return accepted(evolved)

        evolved = []
        for k, g in enumerate(integrators):
            rhs = m_u[k]
            for j in range(tableau.stages):
                if b_e[j] != 0.0:
                    rhs = rhs + dt * b_e[j] * cast(Array, nonlinear_stage[k][j])
                if b_i[j] != 0.0:
                    rhs = rhs + dt * b_i[j] * cast(Array, linear_stage[k][j])
            forcing_k = forcing[k]
            if forcing_k is not None:
                rhs = rhs + dt * forcing_k
            evolved.append(g.apply_mass_inverse(rhs))
        return accepted(evolved)
