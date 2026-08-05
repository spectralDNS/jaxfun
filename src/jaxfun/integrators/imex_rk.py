"""Diagonally-implicit IMEX Runge-Kutta time integration."""

from typing import cast

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.la import BaseMatrix
from jaxfun.typing import Array, Padding, ScalarSpaceType

from .base import BaseIntegrator, _warm_operator_solve_cache
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
        V: ScalarSpaceType,
        equation: sp.Expr,
        *,
        tableau: IMEXTableau,
        initial: sp.Expr | Array,
        time: tuple[float, float] | None = None,
        **params,
    ):
        """Construct an IMEX Runge-Kutta integrator for a semilinear weak form."""
        super().__init__(V, equation, initial=initial, time=time, **params)
        self.tableau = nnx.static(tableau)

    def setup(self, dt: float) -> None:
        """Precompute one implicit system operator per distinct diagonal coefficient."""
        tableau: IMEXTableau = self.tableau
        ops: list[BaseMatrix] = []
        for a_ii in tableau.distinct_diagonal_coeffs:
            op = self.mass_operator - dt * a_ii * self.linear_operator
            _warm_operator_solve_cache(op)
            ops.append(op)
        self._stage_operators: tuple[BaseMatrix, ...] = nnx.data(tuple(ops))

    def _stage_operator(self, a_ii: float) -> BaseMatrix:
        """Return the cached implicit system operator for diagonal coeff `a_ii`."""
        idx = self.tableau.distinct_diagonal_coeffs.index(a_ii)
        return self._stage_operators[idx]

    @jax.jit(static_argnums=(0, 3))
    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array:
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
        a_e, a_i = tableau.explicit.A, tableau.implicit.A
        b_e, b_i, c_i = tableau.explicit.b, tableau.implicit.b, tableau.implicit.c

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
                stage = self.apply_mass_inverse(rhs)
            else:
                stage = self._stage_operator(a_ii).solve(rhs)

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
