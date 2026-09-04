# Rayleigh-Benard convection in a periodic channel
#
#   u_t + Grad(u)⋅u = -Grad(p) + nu*Div(Grad(u)) + T*j
#   T_t + Div(u*T)  = kappa*Div(Grad(T))
#   Div(u)          = 0
#
# on (x, y) in [0, Lx] x (-1, 1), periodic in x, no-slip walls, T(-1) = 1 and
# T(+1) = 0, with nu = sqrt(Pr/Ra) and kappa = 1/sqrt(Ra*Pr). Gravity acts along
# -y, so the buoyancy term drives the wall-normal (y) momentum equation.
#
# ChannelFlow2D.py carries the whole velocity formulation -- the pressure-free
# fourth-order equation for v, the continuity solve for u, the k=0 mean flow, and
# the stage loop. See its header for all of that. This file adds only what makes
# the problem Rayleigh-Benard, through the four extension hooks:
#
#   scalar_integrators  the temperature equation, appended to the state
#   scalar_initial      the conducting profile plus a perturbation
#   scalar_terms        -Div(u*T), the temperature's explicit right-hand side
#   buoyancy            +T,xx, the extra explicit forcing on the v equation
#
# The buoyancy term is what the pressure elimination leaves of T*j: applying
# Div(Grad(.)) to the y-momentum equation and substituting Div(Grad(p)) from the
# divergence of the momentum equation turns T*j into T,xx.
#
# THE TEMPERATURE BOUNDARY CONDITIONS ARE A STATIC LIFTING
#
# T lives in a DirectSumTPS whose inhomogeneous Dirichlet data is baked in at
# construction, so the coefficients carry only the departure from the conducting
# profile and a zero initial state *is* the conducting profile. Two consequences,
# both asserted below: the lifting is static, so it contributes nothing to dT/dt
# and the mass term's forcing must be dropped; and it is linear in y, so its
# Laplacian vanishes and the diffusion term's forcing is exactly zero.
#
# RESOLUTION AND STEP SIZE ARE COUPLED
#
# Only the diffusive terms are implicit, so dt is limited by the advective
# Courant number and has to come down roughly in step with the wall-normal
# resolution, about like 1/N^2. Measured at Ra = 1e6, Pr = 0.7, to t = 20:
#
#   128 x  64   dt = 0.025     runs
#   192 x  96   dt = 0.025     diverges     (so does 128 x 96: it is N that binds)
#   192 x  96   dt = 0.0125    runs
#   192 x  96   dt = 0.00625   runs, identical to dt = 0.0125 (Nu, max|u| to 4 sf)
#
# Those were measured on Legendre; the limit is the advective Courant number, so
# it is a property of the node spacing rather than of the basis, and Chebyshev
# behaves the same. Note that a run past the limit does not merely give a wrong
# answer -- it goes to NaN, and a NaN run is faster than a healthy one, so never
# read a timing off a run you have not checked with `jnp.isfinite`.
#
# so the failure at dt = 0.025 is a stability limit, not an accuracy one -- half
# the step is already time-converged. The defaults below (128 x 64, dt = 0.025)
# are shenfun's for this problem. They leave the top third of the temperature
# spectrum at ~6e-3 of its peak, which is marginal rather than comfortable: it is
# also the regime where skipping the wall-normal dealiasing stops being free (see
# "DEALIASING" in ChannelFlow2D.py). Raise both M, N and lower dt together if you
# need a converged Nusselt number -- at these defaults it reads 14.7 +- 5.4,
# where the spread is genuine turbulent fluctuation, not drift.
#
# VERIFICATION
#
# The onset of convection against linear stability theory: with the free-fall
# scaling on a channel of height H = 2, rescaling to unit height gives
# Ra_eff = Ra*H^3 = 8*Ra, so the classical rigid-rigid result Ra_c = 1707.762 at
# a_c = 3.117 predicts neutral stability at Ra = 213.47 and Lx = 4.0316 here.
# `critical_rayleigh` measures it. The velocity solver itself is verified
# separately, and more sharply, by the Orr-Sommerfeld run in OrrSommerfeld.py.
#
# THE TEMPERATURE FOLLOWS THE VELOCITY'S BASIS
#
# `polynomial` and `kind` are forwarded to `KMM2D` and reused for T, so
# the whole solver stays in one basis. Two places where that
# has to be got right, both silent if wrong:
#
#   * T carries a wall-normal second derivative, so under Petrov-Galerkin it
#     needs a test space of its own, exactly as the v equation does. Without one
#     the Chebyshev diffusion operator goes from 4 diagonals to N/2 of them.
#   * Buoyancy is a right-hand side of the *v* equation, so `C_T` is tested
#     against `self.PB` -- whatever the v equation itself uses -- not against
#     `self.VB`. Under Galerkin those are the same object; under PG they are not.
#
# See "CHOICE OF BASIS AND TEST SPACE" in ChannelFlow2D.py for which pairings are
# worth using. The temperature shifts the balance a little -- it adds a pair of
# transforms, favouring Chebyshev, and one more banded solve, favouring Legendre
# -- but not enough to change the recommendation either way.
#
# Spatial discretization: Fourier x (Legendre Galerkin | Chebyshev Petrov-Galerkin)
# Time discretization: any globally stiffly accurate IMEX Runge-Kutta tableau
# ruff: noqa: E402
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import jax

# Before any jaxfun import: the verification below resolves growth rates against
# an O(1) base flow, which float32 cannot separate. See ChannelFlow2D.py.
jax.config.update("jax_enable_x64", True)

# Likewise before any jaxfun import, and for a related reason: `jaxfun.sharding`
# builds its device mesh at import time, so the other processes' devices have to
# be visible by then. A no-op outside an MPI launcher or on a single rank; under
# an MPI launcher, mpi4py is required.
from spmd_bootstrap import echo, initialize_distributed, is_leader, to_host

initialize_distributed()

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from ChannelFlow2D import (
    ASSEMBLE,
    KMM2D,
    SOLVE,
    growth_rate_of,
    linear_operator,
    snapshot_times,
)
from flax import nnx
from matplotlib.animation import FuncAnimation

from jaxfun.galerkin import (
    DirectSumTPS,
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.integrators import ARS443, IMEXRungeKutta, IMEXTableau
from jaxfun.operators import Constant, Div, Grad
from jaxfun.typing import Array, PolynomialKind, TestSpaceKind

M, N = 128, 64  # Fourier modes (x), wall-normal modes (y)
LX = 2 * float(sp.pi)  # periodic box width
RA, PR = 1e6, 0.7
DT, T_END = 0.025, 60.0
N_SNAPSHOTS = 120
SEED = 0
AMPLITUDE = 1e-3  # initial temperature perturbation
TABLEAU = ARS443
MODE = 0
CRITICAL = True  # run the linear-stability verification
# Wall-normal basis and test space, forwarded to KMM2D and reused for the
# temperature. Pair CHEBYSHEV with PG and LEGENDRE with GALERKIN -- see "CHOICE
# OF BASIS" in ChannelFlow2D.py. Chebyshev gains as N grows; where the two cross
# depends on the machine, so measure before caring.
POLYNOMIAL = PolynomialKind.CHEBYSHEV
KIND = TestSpaceKind.PETROV_GALERKIN

if "PYTEST" in os.environ:
    M, N, T_END, N_SNAPSHOTS, CRITICAL = 16, 16, 0.2, 2, False


class RayleighBenard(KMM2D):
    """Rayleigh-Benard convection: Navier-Stokes plus a buoyant temperature.

    The state is `(v_hat, u0, T_hat)` -- the velocity pair inherited from
    `KMM2D`, with the temperature appended as its transported scalar.
    """

    def __init__(
        self,
        M: int,
        N: int,
        Lx: float,
        Ra: float,
        Pr: float,
        *,
        amplitude: float = AMPLITUDE,
        seed: int = SEED,
        mode: int = MODE,
        tableau: IMEXTableau = TABLEAU,
        time: tuple[float, float] | None = None,
        padding: tuple[int, int] | None = None,
        polynomial: PolynomialKind = PolynomialKind.LEGENDRE,
        kind: TestSpaceKind = TestSpaceKind.GALERKIN,
    ) -> None:
        """Assemble the velocity solver, then the temperature on top of it.

        Args:
            M: Number of Fourier modes along the periodic direction.
            N: Number of wall-normal modes.
            Lx: Width of the periodic box.
            Ra: Rayleigh number.
            Pr: Prandtl number.
            amplitude: Size of the initial temperature perturbation.
            seed: PRNG seed for the perturbation.
            mode: 0, 1 or 2. The first two disturbs an initial linear profile
                with broadband noise; the last starts with zero temperature
                throughout the domain, perturbed by noise.
            tableau: Any globally stiffly accurate IMEX Runge-Kutta tableau.
            time: Optional default integration interval.
            padding: Shape of real space, as in `KMM2D`.
            polynomial: Wall-normal basis, passed straight through to
                `KMM2D` and reused for the temperature.
            kind: GALERKIN or PETROV_GALERKIN, likewise.
        """
        nu = float((Pr / Ra) ** 0.5)
        super().__init__(
            M,
            N,
            Lx,
            nu,
            tableau=tableau,
            time=time,
            padding=padding,
            polynomial=polynomial,
            kind=kind,
        )
        self.Ra, self.Pr = nnx.static(Ra), nnx.static(Pr)
        self.amplitude = nnx.static(amplitude)
        self.seed, self.mode = nnx.static(seed), nnx.static(mode)

        bcT = {"left": {"D": 1}, "right": {"D": 0}}
        Tb = FunctionSpace(N, self.polspace, bcs=bcT, name="Tb")
        VT = TensorProduct(self.F, Tb, system=self.system, name="VT")
        # The wall values are inhomogeneous, so VT is a direct sum of the
        # homogeneous space and the Dirichlet lifting -- which is what
        # `scalar_initial` perturbs and what carries the conducting profile.
        assert isinstance(VT, DirectSumTPS)
        self.VT = nnx.static(VT)

        # The temperature carries a wall-normal second derivative, so under
        # Petrov-Galerkin it needs a test space of its own for the same reason
        # the v equation does. `DirectSum.get_testspace` forwards to the
        # homogeneous summand, which is the only part a test function sees --
        # the lifting is trial-side data.
        if self.testkind is TestSpaceKind.PETROV_GALERKIN:
            PT = TensorProduct(
                self.F,
                Tb.get_testspace("PG", name="TbP"),
                system=self.system,
                name="PT",
            )
        else:
            PT = VT

        x, y = self.system.base_scalars()
        t = self.system.base_time()
        kappa = Constant("kappa", float(1.0 / (Ra * Pr) ** 0.5))
        Tt = TrialFunction(VT, name="T", transient=True)
        s = TestFunction(PT, name="s")
        # Buoyancy is a right-hand side of the *v* equation, so it has to be
        # tested against whatever the v equation is tested against -- self.PB,
        # which is self.VB under Galerkin and the PG test space otherwise.
        q = TestFunction(self.PB, name="q")
        g = TrialFunction(self.Wo, name="g")

        eq_T = (Tt.diff(t) - kappa * Div(Grad(Tt))) * s
        self.gT = nnx.data(
            IMEXRungeKutta(
                eq_T,
                initial=jnp.zeros(VT.num_dofs, dtype=complex),
                tableau=tableau,
                solver_options=SOLVE,
                **ASSEMBLE,
            )
        )
        # The lifting is linear in y, so its Laplacian is analytically zero and
        # contributes no forcing. Numerically it is zero only to round-off, and
        # only exactly zero when the streamwise FFT is: at M = 128 this is 0.0,
        # at M = 126 it is 3e-18 and at M = 130 it is 2e-17, against a field of
        # order one. Test it as round-off rather than as an identity, so that
        # the assertion does not depend on M being a power of two.
        forcing = self.gT.linear_forcing
        assert forcing is None or float(jnp.abs(jnp.asarray(forcing)).max()) < 1e-14, (
            "the Dirichlet lifting is linear, so its Laplacian must vanish"
        )

        # Buoyancy into the v equation, and -Div(u*T) into the T equation.
        self.C_T = nnx.data(linear_operator(Tt.diff(x, 2) * q))
        self.C_ux = nnx.data(linear_operator(-g.diff(x, 1) * s))
        self.C_uy = nnx.data(linear_operator(-g.diff(y, 1) * s))

    # -- KMM2D extension hooks ---------------------------------------------

    @property
    def scalar_integrators(self) -> tuple[IMEXRungeKutta, ...]:
        """The temperature equation."""
        return (self.gT,)

    def buoyancy(self, scalars: tuple[Array, ...]) -> Array:
        """Return +T,xx, what the pressure elimination leaves of the T*j term."""
        return self.C_T @ scalars[0]

    def scalar_terms(
        self, u_p: Array, v_p: Array, scalars: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        """Return -Div(u*T), the temperature's explicit right-hand side.

        Transformed through the same batched half-spectrum helpers as the
        velocity. Those work in the orthogonal basis, and `to_orthogonal` is what
        carries the Dirichlet lifting into it -- VT is a direct sum, so the
        boundary values are part of the field, not a correction applied after.

        The two flux transforms share a batch here, but not with the velocity's:
        this hook is handed physical fields, by which point `explicit_terms` has
        already taken its own. Merging the two would mean a hook that contributes
        coefficient arrays to the batch instead, which is a larger change to the
        extension surface than the transform it would save.
        """
        (T_c,) = self._wall_normal(self.VT.to_orthogonal(scalars[0]))
        (T_p,) = self._streamwise(T_c)
        tx, ty = self._forward(u_p * T_p, v_p * T_p)
        return (self.C_ux @ tx + self.C_uy @ ty,)

    def scalar_initial(self) -> tuple[Array, ...]:
        """Return the conducting profile plus a perturbation.

        Zero coefficients *are* the conducting profile, that being exactly the
        boundary lifting, so only the perturbation is built here. It is
        multiplied by (1 - y^2) so it vanishes at the walls and is representable
        in the homogeneous space.
        """
        Vh = self.VT.get_homogeneous()
        xx, yy = Vh.mesh()
        if self.mode == 0:
            noise = jax.random.normal(jax.random.PRNGKey(self.seed), Vh.shape)
            T_hat = Vh.forward(self.amplitude * noise * (1 - yy**2))
        elif self.mode == 1:
            noise = jnp.cos(2 * jnp.pi * self.mode * xx / self.Lx) * jnp.ones_like(yy)
            T_hat = Vh.forward(self.amplitude * noise * (1 - yy**2))
        else:
            noise = jax.random.uniform(jax.random.PRNGKey(self.seed), self.VT.shape)
            T_hat = self.VT.forward(self.amplitude * noise * (1 - yy**2))

        return (T_hat.at[self.nyquist].set(0.0),)

    # -- diagnostics -------------------------------------------------------

    @jax.jit(static_argnums=0)
    def nusselt(self, state: tuple[Array, ...]) -> Array:
        """Return the instantaneous Nusselt number, total over conductive flux.

        The conducting state carries dT/dy = -1/2 across the height-2 channel, so
        its flux is kappa/2 and Nu = 1 + 2*<v T>/kappa.
        """
        v_p = self.VB.backward(state[0], N=self.pad)
        T_p = self.VT.backward(state[2], N=self.pad)
        return 1.0 + 2.0 * float((self.Ra * self.Pr) ** 0.5) * self.average(v_p * T_p)

    def extra_diagnostics(self, state: tuple[Array, ...]) -> dict[str, float]:
        """Return the temperature-specific checks."""
        T_hat = state[2]
        T_p = self.VT.backward(T_hat, N=self.pad)
        # Resolution check: the fraction of the temperature spectrum sitting in
        # the top third of the modes in either direction. A resolved spectral
        # solution decays to round-off there; a growing tail means aliasing or an
        # under-resolved boundary layer, and is the first thing to look at if the
        # run misbehaves. It reads 1.0 on the initial condition by construction,
        # the perturbation being broadband noise.
        # Only the wavenumbers, never the padding `RFourier` stores after them:
        # those rows are structurally zero, so leaving them in would push the
        # window off the top of the real spectrum and read a smaller tail on
        # more devices than on one.
        mag = jnp.abs(T_hat)[: self.F.n_real]
        m3, n3 = mag.shape[0] // 3, mag.shape[1] // 3
        # The streamwise spectrum is stored half, ordered 0 .. M/2, so its top
        # third is simply its tail -- no fftshift to bring the two halves together.
        tail = max(float(mag[-m3:].max()), float(mag[:, -n3:].max()))
        return {
            "tail": tail / max(float(mag.max()), 1e-300),
            "Trange": float(T_p.max() - T_p.min()),
            "Nu": float(self.nusselt(state)),
        }


# ---------------------------------------------------------------------------
# Linear stability: the critical Rayleigh number
# ---------------------------------------------------------------------------
RA_C_PREDICTED = 1707.762 / 8.0
LX_C_PREDICTED = 2 * float(sp.pi) / (3.117 / 2)


def growth_rate(Ra: float, Lx: float, *, m: int = 8, n: int = 32) -> float:
    """Return the measured exponential growth rate of an infinitesimal mode.

    One Fourier mode is seeded at amplitude 1e-6 -- small enough that the
    dynamics stay linear -- and the rate is read off the temperature norm over
    the second half of the run, once the initial transient has decayed onto the
    least-stable eigenfunction.
    """
    solver = RayleighBenard(
        m,
        n,
        Lx,
        Ra,
        PR,
        amplitude=1e-6,
        mode=1,
        polynomial=POLYNOMIAL,
        kind=KIND,
    )
    dt, steps, batches = 0.05, 2000, 40
    snaps = solver.solve(
        dt=dt,
        steps=steps,
        n_batches=batches,
        return_batch_snapshots=True,
        progress=False,
    )
    return growth_rate_of(
        jnp.abs(snaps[2]).max(axis=(1, 2)), snapshot_times(dt, steps, batches)
    )


def critical_rayleigh() -> None:
    """Verify the onset of convection against linear stability theory.

    Two independent statements. The sharp one is the growth rate *at* the
    predicted critical Rayleigh number, which must vanish; the linear fit through
    a bracket is a weaker cross-check, slightly biased because the growth rate is
    not exactly linear in Ra. The critical wavenumber is confirmed separately by
    perturbing the box width in both directions, which must reduce the rate --
    Lx_c is where the neutral curve is at its minimum.
    """
    echo(f"\nlinear stability at Lx = {LX_C_PREDICTED:.4f} (a_c = 3.117 / H)")
    ras = [RA_C_PREDICTED * f for f in (0.96, 0.98, 1.0, 1.02, 1.04)]
    rates = [growth_rate(Ra, LX_C_PREDICTED) for Ra in ras]
    for Ra, rate in zip(ras, rates, strict=True):
        echo(f"  Ra = {Ra:9.4f}   growth rate = {rate:+.9f}")
    slope, intercept = jnp.polyfit(jnp.asarray(ras), jnp.asarray(rates), 1)
    Ra_c = float(-intercept / slope)
    neutral = rates[len(rates) // 2]
    echo(f"  rate at the predicted Ra_c   = {neutral:+.3e}  (must vanish)")
    echo(
        f"  neutral from the fit: Ra_c = {Ra_c:.4f} -> Ra_c*H^3 = {8 * Ra_c:.3f}"
        f"   linear theory 1707.762  ({100 * abs(8 * Ra_c / 1707.762 - 1):.3f}% off)"
    )
    for f in (0.85, 1.15):
        rate = growth_rate(RA_C_PREDICTED, LX_C_PREDICTED * f)
        echo(f"  Lx = {LX_C_PREDICTED * f:.4f}   rate = {rate:+.9f}  (must decay)")
        assert rate < neutral, "Lx_c must minimise the neutral curve"
    assert abs(neutral) < 1e-5, f"not neutral at the predicted Ra_c: {neutral:+.3e}"
    assert abs(8 * Ra_c / 1707.762 - 1) < 5e-3, f"Ra_c off by {8 * Ra_c:.2f}"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main() -> None:
    """Integrate the convection problem, verify it, and plot the result."""
    echo(f"M={M} N={N} Ra={RA:g} Pr={PR} dt={DT} T={T_END}")
    padding = (3 * M // 2, N)
    solver = RayleighBenard(
        M,
        N,
        LX,
        RA,
        PR,
        time=(0.0, T_END),
        padding=padding,
        polynomial=POLYNOMIAL,
        kind=KIND,
        mode=2,
    )
    echo(
        f"  dofs: v {solver.VB.num_dofs} T {solver.VT.num_dofs} u0 {solver.D1.num_dofs}"
    )

    d0 = solver.diagnostics(solver.initial_coefficients())
    echo("  initial " + "  ".join(f"{k}={v:.3e}" for k, v in d0.items()))

    _t = time.time()
    snaps = solver.solve(
        dt=DT,
        n_batches=N_SNAPSHOTS,
        return_batch_snapshots=True,
        progress=is_leader(),
    )
    echo(f"  run {time.time() - _t:.1f}s")

    final: tuple[Array, Array, Array] = (snaps[0][-1], snaps[1][-1], snaps[2][-1])
    d1 = solver.diagnostics(final)
    echo("  final   " + "  ".join(f"{k}={v:.3e}" for k, v in d1.items()))
    echo(f"  Courant = {solver.courant(final, DT):.2f}  (advection is explicit)")

    # Nu fluctuates, so the number worth comparing against another code is a time
    # average taken once the flow is statistically steady -- here the second half.
    nus = jnp.asarray(
        [solver.nusselt(tuple(s[i] for s in snaps)) for i in range(len(snaps[0]))]
    )
    q = len(nus) // 4
    every = max(1, len(nus) // 12)
    echo("  Nu(t) " + " ".join(f"{float(v):.1f}" for v in nus[::every]))
    line = (
        f"  Nu = {float(nus[2 * q :].mean()):.3f} +- {float(nus[2 * q :].std()):.3f}"
        f"  over t > {T_END / 2:g}"
    )
    if q:
        # Quarter by quarter, so a residual spin-up trend shows up instead of hiding
        # inside a single mean: these two must agree to within the spread.
        line += (
            f"   [3rd quarter {float(nus[2 * q : 3 * q].mean()):.2f},"
            f" 4th {float(nus[3 * q :].mean()):.2f}]"
        )
    echo(line)

    assert d1["div"] < 1e-10, f"divergence not satisfied: {d1['div']:.3e}"
    # Exactly zero, unlike the Orr-Sommerfeld run: the fluid starts at rest, so v is
    # identically zero and nothing ever drives its k=0 mode.
    assert d1["v[k=0]"] == 0.0, "the wall-normal mean mode must stay exactly zero"
    assert d1["u[k=0]-u0"] < 1e-10, "the mean flow must be recovered exactly"

    if "PYTEST" in os.environ:
        assert all(bool(jnp.isfinite(s).all()) for s in final)
        sys.exit(0)

    if CRITICAL:
        critical_rayleigh()

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    # VT carries the inhomogeneous wall temperatures, so it is a direct sum, and
    # a direct sum declines to batch while sharding is active: the boundary
    # lifting it adds in is placed on the space's sharding, which the batch axis
    # has no counterpart for. Plotting is a one-off, so transforming snapshot by
    # snapshot there costs nothing worth avoiding. Indexed rather than iterated:
    # a global array spanning another process's devices refuses `__iter__`.
    n_snaps = snaps[2].shape[0]
    T_phys = (
        solver.VT.backward_batch(snaps[2])
        if len(jax.devices()) == 1
        else jnp.stack([solver.VT.backward(snaps[2][i]) for i in range(n_snaps)])
    )
    u_final = solver.VD.backward(solver.velocity(final[0], final[1]))
    v_final = solver.VB.backward(final[0])
    x_plot, y_plot = solver.VT.mesh(broadcast=False)

    # Matplotlib reads these element by element, which a distributed array
    # cannot serve, so they have to come back to the host first. That gather is
    # a *collective* -- every process has to reach it -- which is why it happens
    # here and the rank-0 guard comes after it rather than before.
    T_phys, u_final, v_final, x_plot, y_plot = to_host(
        (T_phys, u_final, v_final, x_plot, y_plot)
    )

    # Every process ran the same solve, so only one should draw anything or
    # write the animation out.
    if not is_leader():
        return

    times = jnp.linspace(0.0, T_END, snaps[0].shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    c1 = axes[0].contourf(x_plot, y_plot, T_phys[-1].T, levels=40, cmap="RdBu_r")
    axes[1].quiver(x_plot[::4], y_plot[::2], u_final.T[::2, ::4], v_final.T[::2, ::4])
    axes[0].set_title(f"T(x, y, t={float(times[-1]):.1f}),  Ra={RA:g} Pr={PR}")
    axes[1].set_title("Velocity")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.colorbar(c1, ax=axes[0], shrink=0.9)

    fig_anim, ax_anim = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    im = ax_anim.imshow(
        T_phys[0].T,
        origin="lower",
        extent=(
            float(x_plot[0]),
            float(x_plot[-1]),
            float(y_plot[0]),
            float(y_plot[-1]),
        ),
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="bilinear",
    )
    fig_anim.colorbar(im, ax=ax_anim, shrink=0.9)
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    title = ax_anim.set_title(f"Rayleigh-Benard (t={float(times[0]):.2f})")

    def update(frame: int):
        im.set_data(T_phys[frame].T)
        title.set_text(f"Rayleigh-Benard (t={float(times[frame]):.2f})")
        return (im,)

    _anim = FuncAnimation(fig_anim, update, frames=len(times), interval=40, blit=False)
    _anim.save("rayleighbenard.gif", writer="pillow", fps=24)
    plt.show()


if __name__ == "__main__":
    main()
