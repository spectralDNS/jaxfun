from typing import Any, cast

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import sympy as sp
import tqdm
from flax import nnx

from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.typing import Array

from .base import BaseIntegrator, _operator_to_dense


def _phi1(z: Array) -> Array:
    small = jnp.abs(z) < 1e-7
    series = 1 + z / 2 + z**2 / 6 + z**3 / 24 + z**4 / 120
    return jnp.where(small, series, jnp.expm1(z) / z)


def _phi2(z: Array) -> Array:
    small = jnp.abs(z) < 1e-6
    series = 0.5 + z / 6 + z**2 / 24 + z**3 / 120 + z**4 / 720
    return jnp.where(small, series, (jnp.expm1(z) - z) / z**2)


def _phi3(z: Array) -> Array:
    small = jnp.abs(z) < 1e-5
    series = 1 / 6 + z / 24 + z**2 / 120 + z**3 / 720 + z**4 / 5040
    return jnp.where(small, series, (jnp.expm1(z) - z - z**2 / 2) / z**3)


def _etdrk4_diag_coeffs(
    L: Array, dt: float
) -> tuple[Array, Array, Array, Array, Array, Array]:
    z = dt * L
    E = jnp.exp(z)
    E2 = jnp.exp(z / 2)
    Q = _phi1(z / 2)
    phi1 = _phi1(z)
    phi2 = _phi2(z)
    phi3 = _phi3(z)
    f1 = phi1 - 3 * phi2 + 4 * phi3
    f2 = phi2 - 2 * phi3
    f3 = phi3
    return E, E2, Q, f1, f2, f3


def _phi_matrices(z: Array) -> tuple[Array, Array, Array]:
    n = z.shape[0]
    I = jnp.eye(n, dtype=z.dtype)
    expz = jsp_linalg.expm(z)
    zinv = jnp.linalg.pinv(z)
    phi1 = zinv @ (expz - I)
    phi2 = zinv @ (phi1 - I)
    phi3 = zinv @ (phi2 - 0.5 * I)
    return phi1, phi2, phi3


def _apply_etd_operator(op: Array, u: Array, is_diag: bool) -> Array:
    return op * u if is_diag else op @ u


class ETDRK4(BaseIntegrator):
    """Fourth-order exponential time differencing for semilinear systems."""

    def __init__(
        self,
        V: OrthogonalSpace,
        equation: sp.Expr,
        u0: sp.Expr | Array | None = None,
        *,
        time: tuple[float, float] | None = None,
        initial: sp.Expr | Array | None = None,
        **params: Any,
    ):
        super().__init__(
            V,
            equation,
            u0,
            time=time,
            initial=initial,
            sparse=bool(params.get("sparse", False)),
            sparse_tol=int(params.get("sparse_tol", 1000)),
        )
        self.params = dict(params)
        self._dt: float | None = None
        # self._coeffs = nnx.data({})

        zero = jnp.zeros(self.functionspace.num_dofs)
        forcing_rhs = (
            self.apply_mass_inverse(jnp.asarray(self.linear_forcing))
            if self.linear_forcing is not None
            else zero
        )
        self._forcing_rhs = nnx.data(forcing_rhs)

    def setup(self, dt: float) -> None:
        self.params["dt"] = dt
        self._dt = dt

        self.is_diag = nnx.static(
            self.mass_diag is not None
            and (self.linear_operator is None or self.linear_diag is not None)
        )

        if self.is_diag:
            Ldiag = (
                jnp.zeros_like(self.mass_diag)
                if self.linear_operator is None
                else self.linear_diag / self.mass_diag
            )
            E, E2, Q, f1, f2, f3 = _etdrk4_diag_coeffs(Ldiag, dt)
        else:
            zero = jnp.zeros(self.functionspace.num_dofs)
            if self.linear_operator is None:
                Lmat = jnp.zeros((zero.size, zero.size), dtype=zero.dtype)
            else:
                A = _operator_to_dense(self.linear_operator)
                if self.mass_diag is not None:
                    m = self.mass_diag.reshape((-1,))
                    Lmat = A / m[:, None]
                elif self.mass_operator is not None:
                    M = _operator_to_dense(self.mass_operator)
                    Lmat = cast(Array, jnp.linalg.solve(M, A))
                else:
                    Lmat = A

            z = dt * Lmat
            E = cast(Array, jsp_linalg.expm(z))
            E2 = cast(Array, jsp_linalg.expm(z / 2))
            Q, _, _ = _phi_matrices(z / 2)
            phi1, phi2, phi3 = _phi_matrices(z)
            f1 = phi1 - 3 * phi2 + 4 * phi3
            f2 = phi2 - 2 * phi3
            f3 = phi3

        self.E = nnx.data(E)
        self.E2 = nnx.data(E2)
        self.Q = nnx.data(Q)
        self.f1 = nnx.data(f1)
        self.f2 = nnx.data(f2)
        self.f3 = nnx.data(f3)

    def _N(self, u_hat: Array) -> Array:
        nval = (
            self.nonlinear_rhs(u_hat) if self.has_nonlinear else jnp.zeros_like(u_hat)
        )
        return nval + self._forcing_rhs

    def step(self, u_hat: Array, dt: float) -> Array:
        shape = u_hat.shape

        def to_state(x: Array) -> Array:
            return x if self.is_diag else x.reshape((-1,))

        def from_state(x: Array) -> Array:
            return x if self.is_diag else x.reshape(shape)

        def apply(op: Array, u: Array) -> Array:
            return _apply_etd_operator(op, u, self.is_diag)

        u0 = to_state(u_hat)
        n1 = to_state(self._N(from_state(u0)))
        a = apply(self.E2, u0) + dt * apply(self.Q, n1)
        n2 = to_state(self._N(from_state(a)))
        b = apply(self.E2, u0) + dt * apply(self.Q, n2)
        n3 = to_state(self._N(from_state(b)))
        c = apply(self.E2, a) + dt * apply(self.Q, 2 * n3 - n1)
        n4 = to_state(self._N(from_state(c)))

        un = apply(self.E, u0) + dt * (
            apply(self.f1, n1) + 2 * apply(self.f2, n2 + n3) + apply(self.f3, n4)
        )
        return from_state(un)

    def solve_with_frames(
        self,
        dt: float,
        steps: int | None = None,
        u_hat: Array | None = None,
        trange: tuple[float, float] | None = None,
        snapshot_stride: int = 1,
        include_initial: bool = True,
        progress: bool = True,
    ) -> tuple[Array, Array]:
        self.setup(dt)
        t0, _, nsteps = self.resolve_time(dt, steps=steps, trange=trange)
        if u_hat is None:
            u_hat = self.initial_coefficients()

        stride = max(1, int(snapshot_stride))
        states: list[Array] = []
        times: list[float] = []
        if include_initial:
            states.append(u_hat)
            times.append(t0)

        iterator = (
            tqdm.trange(nsteps, desc="Integrating", unit="step")
            if progress
            else range(nsteps)
        )
        for i in iterator:
            u_hat = self.step(u_hat, dt)
            t = t0 + (i + 1) * dt
            if (i + 1) % stride == 0 or i == nsteps - 1:
                states.append(u_hat)
                times.append(t)

        return jnp.stack(states), jnp.array(times)
