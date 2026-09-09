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

    def _forcing_caches(
        self, t: Array | float, dt: float
    ) -> tuple[Array | None, list[Array] | None]:
        """Return `(forcing, forcing_stage)` for a step starting at `t`.

        Steady data fills the first and moving data the second; `stage` takes
        whichever it is given.
        """
        # The stage values are computed up front because the forcing does not
        # depend on the state, so none of them has to wait for a solve.
        if not self._transient_boundary:
            forcing = self.linear_forcing
            return (jnp.asarray(forcing) if forcing is not None else None), None
        c_i = self.tableau.implicit.c
        return None, [
            jnp.asarray(self.forcing_at(t + c_i[j] * dt))
            for j in range(self.tableau.stages)
        ]

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
        forcing_stage: list[Array] | None = None,
    ) -> Array:
        """Compute stage `i` from the caches already populated for `j < i`.

        `nonlinear_stage`/`linear_stage` are read-only here (entries for
        `j < i` must already be populated by the caller); this method does
        not append to them -- that bookkeeping is `step`-specific (it depends
        on `step`'s choice of final-combination path) and stays there.

        Pass `forcing_stage` -- the forcing evaluated at every stage time -- when
        it varies in time, and `forcing` when it does not.
        """
        # The forcing rides with the *implicit* tableau, because `A u + f(t)` is
        # the affine part of the implicit operator: that is where it came from,
        # and a stiffly-accurate implicit treatment is what avoids order
        # reduction in the boundary layer a moving wall drives.
        #
        # For a constant forcing the row-sum condition `sum_j a_i[i][j] ==
        # c_i[i]` collapses the accumulation to a single scaling. Both branches
        # are then correct but not bit-for-bit equal -- they sum a different
        # number of terms -- so the shortcut is kept rather than unified, and a
        # steady problem gets exactly the arithmetic it got before.
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
        if forcing_stage is not None:
            # `a_i` is lower triangular including the diagonal, and the forcing
            # does not depend on the state, so stage `i`'s own term is known.
            for j in range(i + 1):
                if a_i[i][j] != 0.0:
                    rhs = rhs + dt * a_i[i][j] * forcing_stage[j]
        elif forcing is not None and c_i[i] != 0.0:
            rhs = rhs + dt * c_i[i] * forcing

        if a_ii == 0.0:
            return self.apply_mass_inverse(rhs)
        return solve_with_options(self._stage_operator(a_ii), rhs, self._solver_options)

    def _step_impl(
        self,
        u_hat: Array,
        dt: float,
        N: ScalarPadding = None,
        t: Array | float = 0.0,
        /,
    ) -> Array:
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
        forcing, forcing_stage = self._forcing_caches(t, dt)

        stages: list[Array] = []
        nonlinear_stage: list[Array | None] = []
        linear_stage: list[Array | None] = []

        for i in range(tableau.stages):
            stage = self.stage(
                i, m_u, dt, nonlinear_stage, linear_stage, forcing, forcing_stage
            )
            stages.append(stage)
            is_last = i == last
            skip_nonlinear = is_last and full_gsa
            skip_linear = is_last and (full_gsa or implicit_only_sa)
            nonlinear_stage.append(
                None if skip_nonlinear else self.nonlinear_rhs_scalar_product(stage, N)
            )
            linear_stage.append(None if skip_linear else (self.linear_operator @ stage))

        # Both stiffly-accurate paths need no forcing term of their own: `b_i`
        # equals `a_i[-1]` there, so the `b_i`-weighted forcing is already
        # inside the last stage. That is why the general path below is the only
        # one that carries it, for a moving forcing exactly as for a fixed one.
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
        if forcing_stage is not None:
            for j in range(tableau.stages):
                if b_i[j] != 0.0:
                    rhs = rhs + dt * b_i[j] * forcing_stage[j]
        elif forcing is not None:
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
        self,
        u_hats: tuple[Array, ...],
        dt: float,
        N: ScalarPadding = None,
        t: Array | float = 0.0,
        /,
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
        caches = tuple(g._forcing_caches(t, dt) for g in integrators)
        forcing = tuple(c[0] for c in caches)
        forcing_stage = tuple(c[1] for c in caches)

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
                    i,
                    m_u[k],
                    dt,
                    nonlinear_stage[k],
                    linear_stage[k],
                    forcing[k],
                    forcing_stage[k],
                )
            # Constraints are algebraic, so they carry no `dB/dt` term -- but
            # their own boundary lifting still has to be read at this stage's
            # time, not the step's.
            state_i = self.resolve_constraints(
                tuple(stage_i), M, t=t + tableau.implicit.c[i] * dt
            )
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
            return self.resolve_constraints(tuple(out), M, t=t + dt)

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
            stage_forcing = forcing_stage[k]
            if stage_forcing is not None:
                for j in range(tableau.stages):
                    if b_i[j] != 0.0:
                        rhs = rhs + dt * b_i[j] * stage_forcing[j]
            elif forcing[k] is not None:
                rhs = rhs + dt * cast(Array, forcing[k])
            evolved.append(g.apply_mass_inverse(rhs))
        return accepted(evolved)
