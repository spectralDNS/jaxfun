from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxfun.typing import Array

from .base import BaseIntegrator


class BackwardEuler(BaseIntegrator):
    """Backward Euler integrator for first-order semi-discrete PDE systems.

    The weak form is assumed to define a first-order in time system of the form
    (after spatial Galerkin discretization):

        M * du/dt = L * u + b

    where `M` is the mass matrix (from the time-derivative term), and `L` / `b`
    come from the linear RHS terms.

    The Backward Euler update is:

        (M - dt*L) * u_{n+1} = M*u_n + dt*b
    """

    name = "BackwardEuler"

    @jax.jit(static_argnums=0)
    def step(self, state: Array, t: float, dt: float) -> Array:
        M, L, b = self._dense_linear_operators()

        rhs = M @ state
        if b is not None:
            rhs = rhs + dt * b

        A = M - dt * L
        return jnp.linalg.solve(A, rhs)
