# Turbulent channel flow with the three-dimensional KMM solver
#
# Fully developed turbulence between two parallel walls, driven by a constant
# mean pressure gradient, at a friction Reynolds number Re_tau = u_tau*h/nu of
# 180 -- the case of Kim, Moin & Moser (JFM 177:133-166, 1987), whose method
# ChannelFlow3D.py implements. The solver is used unchanged; this module adds
# what a turbulence run needs around it: an initial condition that becomes
# turbulent quickly, plane- and time-averaged statistics, and restartable
# checkpoints.
#
# UNITS
#
# Everything is scaled by the half-height h and the friction velocity u_tau, so
# nu = 1/Re_tau and the driving pressure gradient is exactly 1. In a
# statistically steady state the wall shear then balances the force at
# u_tau = 1, and profiles in wall units are the raw profiles: y+ = (1 -|z|)*Re_tau
# and U+ = U. Measuring u_tau from the wall shear of the mean profile is the
# first check that a run has settled -- it must come back to 1.
#
# GETTING TO TURBULENCE QUICKLY
#
# A channel at Re_tau = 180 is subcritical: small noise on a laminar profile
# decays, and a large one wastes several eddy turnovers reorganizing itself. Two
# things shorten that.
#
# The mean profile starts turbulent. Under the same forcing the *laminar*
# profile peaks at Re_tau/2 = 90, five times the turbulent centreline velocity,
# and a run started from it spends its spin-up decelerating. Reichardt's
# composite law of the wall is close to the final profile everywhere.
#
# The fluctuations start as the structures that sustain wall turbulence rather
# than as noise: near-wall streaks spaced about 100 wall units apart, with
# streamwise vortices sitting over them in the phase that lifts slow fluid off
# the wall, and a sinuous waviness along x of wavelength ~400 wall units, the
# instability through which streaks break down. Low-amplitude broadband noise
# on top breaks the remaining symmetries.
#
# All of it is seeded through the solver's own state variables: the streaks
# through the wall-normal vorticity g, the vortices through the wall-normal
# velocity w. The horizontal velocities are recovered from continuity, so the
# initial field is divergence-free and satisfies no-slip by construction,
# whatever the seed.
#
# STATISTICS
#
# Plane averages are taken on the 3/2-padded mesh, where the quadratic products
# are exact, by the same transform path `KMM3D.explicit_terms` uses -- sharded
# where the solver is. Only the small (10, N) array of plane averages ever
# leaves the devices. `ChannelStatistics` accumulates them over time and reports
# the mean profile, the Reynolds stresses and the total shear stress, with the
# two channel halves folded together.
#
# The total shear stress nu*dU/dz - <u'w'> is the built-in check. Averaging the
# streamwise momentum equation over planes and time leaves d/dz of it balancing
# the unit force, so it must equal -z exactly in a statistically steady state.
# Its deviation measures how far the average is from converged, and a
# deviation that does not decay with averaging time means something is wrong.
#
# CHECKPOINTS
#
# Checkpoints are written with Orbax. Each process writes its own shards of the
# state, so no array is gathered anywhere. Writes are asynchronous and overlap
# with the next stretch of time stepping. A checkpoint becomes visible only once
# it is complete, so the latest one on disk is always usable. The running
# statistics and the configuration travel with the state.
#
# A restart may change the grid. The state is truncated or zero-padded directly
# in the composite coefficients. Every composite basis function satisfies the
# boundary conditions, so no-slip survives exactly. Use this to spin up on a
# coarse grid and continue on a finer one. Averaging restarts after a change of
# grid or Re_tau.
#
# SNAPSHOTS
#
# Checkpoints restart the solver; they hold spectral coefficients and nothing a
# visualization tool can read. For pictures, `output.snapshot_every` writes the
# physical velocity to one HDF5 file per run with an XDMF sidecar beside it,
# which is what ParaView opens (see jaxfun.utils.hdf5file for why XDMF rather
# than VTKHDF). A restart appends to the same file, and the sidecar is rewritten
# after every snapshot, so a run killed at any point leaves a file that opens.
#
# The snapshots are written on the unpadded quadrature mesh: the same points a
# plain backward transform lands on, with no interpolation, 2.25 times smaller
# than the padded mesh the nonlinear terms use. `snapshot_closed` adds the two
# walls and closes the periodic ends, which costs nothing here -- the velocity
# is exactly zero at z = +-1 for every composite basis function -- and is only
# cosmetic: the Gauss points exclude the endpoints, so without it ParaView draws
# a slab with a seam and no skin at the walls.
#
# CASE FILES
#
# A run is described by a TOML case file (see channel_case.py for the keys, and
# cases/re180.toml for a commented template). Relative output paths -- the
# checkpoint directory and the profile figure -- are resolved against the case
# file's own directory, so a case can be copied anywhere and run from there
# while this module stays under version control. Without a case file the
# built-in defaults run, writing to the working directory. `--set section.key=
# value` overrides a single entry, and `--t-end`, `--seed` and
# `--checkpoint-dir` override theirs.
#
# The device count belongs to the run, not the case -- a checkpoint restarts on
# any number of devices -- so it is set in the environment. On CPU it matters:
# XLA:CPU runs this solver's FFTs single-threaded, so several devices, not the
# thread count, are what put several cores to work. About one device per
# performance (on a Mac) core is right:
#
#   JAX_NUM_CPU_DEVICES=6 python TurbulentChannel3D.py case.toml
#
# Usage:
#
#   python TurbulentChannel3D.py                         # built-in case
#   python TurbulentChannel3D.py ~/runs/re180/case.toml  # fresh, or resume latest
#   python TurbulentChannel3D.py case.toml --t-end 100   # keep going
#   python TurbulentChannel3D.py case.toml --set grid.Nz=96 --set time.dt=5e-4
#   python TurbulentChannel3D.py case.toml --set output.snapshot_every=10
#   python TurbulentChannel3D.py fine.toml --restart-from coarse/turbulent_channel_ckpt
#
# Spatial discretization: Fourier x Fourier x (Chebyshev GR | Legendre Galerkin)
# Time discretization: ARS443 IMEX Runge-Kutta
# ruff: noqa: E402
import os
import sys
import tempfile
import time
from dataclasses import replace

_here = os.path.dirname(os.path.abspath(__file__))
# `spmd_bootstrap` sits one level up, shared with the 2D solver; ChannelFlow3D is
# alongside.
sys.path[:0] = [_here, os.path.dirname(_here)]

import jax

# Before any jaxfun import, so nothing is built at the wrong precision.
jax.config.update("jax_enable_x64", True)

# Likewise before any jaxfun import: `jaxfun.sharding` builds its device mesh at
# import time. A no-op outside an MPI launcher or on a single rank.
from spmd_bootstrap import echo, initialize_distributed, is_leader, to_host

initialize_distributed()

from typing import Any

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from channel_case import ChannelCase, parse_cli, pytest_case, to_dict
from ChannelFlow3D import KMM3D, VelocityKind
from flax import nnx

from jaxfun import galerkin, integrators
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.integrators import IMEXTableau
from jaxfun.sharding import state_sharding
from jaxfun.typing import Array, PolynomialKind, TestSpaceKind
from jaxfun.utils.hdf5file import HDF5File

MOMENTS = ("u", "v", "w", "uu", "vv", "ww", "uv", "uw", "vw", "dudz")


def reichardt(yplus: Array, kappa: float = 0.41) -> Array:
    """Return Reichardt's law of the wall, U+ as a function of y+."""
    return jnp.log1p(kappa * yplus) / kappa + 7.8 * (
        1 - jnp.exp(-yplus / 11) - yplus / 11 * jnp.exp(-yplus / 3)
    )


def integration_weights(polspace: type[OrthogonalSpace], n: int) -> Array:
    """Return weights q_k for the unweighted integral over (-1, 1).

    int_{-1}^{1} f(z) dz ~ sum_k q_k f(z_k) on the n Gauss points z_k of the
    wall-normal basis, exact for polynomials of degree below n. For Legendre
    these are the Gauss weights themselves. The Chebyshev quadratures carry the
    weight function of their family, so they get Fejer's rules instead, which
    keep the same points but integrate without it: the first rule on the
    Chebyshev-Gauss points, the second on the Chebyshev-U-Gauss points. See

      L. Fejer, Mechanische Quadraturen mit positiven Cotesschen Zahlen.
      Math. Z. 37:287-309, 1933. doi:10.1007/BF01474575

      J. Waldvogel, Fast construction of the Fejer and Clenshaw-Curtis
      quadrature rules. BIT 46:195-202, 2006. doi:10.1007/s10543-006-0045-4

    Both rules are symmetric in z, so the ordering of the points is immaterial.
    """
    if polspace is galerkin.Legendre.Legendre:
        return jnp.asarray(np.polynomial.legendre.leggauss(n)[1])
    if polspace is galerkin.Chebyshev.Chebyshev:
        theta = (2 * np.arange(n) + 1) * np.pi / (2 * n)
        j = np.arange(1, n // 2 + 1)[:, None]
        q = 2 / n * (1 - 2 * (np.cos(2 * j * theta) / (4 * j**2 - 1)).sum(axis=0))
        return jnp.asarray(q)
    if polspace is galerkin.ChebyshevU.ChebyshevU:
        theta = (np.arange(n) + 1) * np.pi / (n + 1)
        j = np.arange(1, (n + 1) // 2 + 1)[:, None]
        s = (np.sin((2 * j - 1) * theta) / (2 * j - 1)).sum(axis=0)
        return jnp.asarray(4 * np.sin(theta) / (n + 1) * s)
    raise NotImplementedError(f"no unweighted quadrature for {polspace.__name__}")


class TurbulentChannel(KMM3D):
    """KMM3D in wall units, with turbulence initialization and statistics.

    The flow is driven by a unit mean pressure gradient with nu = 1/Re_tau,
    so a statistically steady state has u_tau = 1 and wall units are the
    solver's own units.
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        Nz: int,
        Lx: float,
        Ly: float,
        Re_tau: float,
        **kwargs: Any,
    ) -> None:
        """Build the solver.

        Args:
            Nx: Number of Fourier modes along the streamwise direction.
            Ny: Number of Fourier modes along the spanwise direction.
            Nz: Number of modes along the wall-normal direction.
            Lx: Streamwise period, in units of the half-height.
            Ly: Spanwise period, in units of the half-height.
            Re_tau: Friction Reynolds number.
            **kwargs: Passed on to `KMM3D`: `padding`, `polynomial`,
                `kind`, `tableau`, `time`.
        """
        super().__init__(Nx, Ny, Nz, Lx, Ly, 1.0 / Re_tau, body_force=1.0, **kwargs)
        self.Re_tau = nnx.static(float(Re_tau))
        self.grid = nnx.static((Nx, Ny, Nz))
        self.q_z = nnx.data(integration_weights(self.polspace, self.pad[2]))
        self.Dx = nnx.data(self.D1.evaluate_basis_derivative(jnp.array([-1.0, 1.0]), 1))

    # -- initial condition -------------------------------------------------

    def turbulent_initial_state(
        self,
        seed: int = 1,
        streak_amp: float = 4.0,
        vortex_amp: float = 1.0,
        noise_amp: float = 0.1,
    ) -> tuple[Array, ...]:
        """Return a state that develops into turbulence quickly.

        A Reichardt mean profile plus near-wall streaks, streamwise vortices
        over them, a sinuous streamwise waviness and broadband noise. See
        "GETTING TO TURBULENCE QUICKLY" in the header.

        Args:
            seed: Seed for the broadband noise.
            streak_amp: Streak amplitude, in units of u_tau.
            vortex_amp: Wall-normal velocity of the vortices, in units of u_tau.
            noise_amp: Noise amplitude relative to the structured part of each
                field.
        """
        Re = float(self.Re_tau)
        Lx, Ly = float(self.Lx), float(self.Ly)
        X, Y, Z = self.VD.mesh()
        yp_lo, yp_up = (1 + Z) * Re, (1 - Z) * Re

        # Streak spacing ~100 and streak wavelength ~400 wall units, rounded
        # to what the box can hold.
        beta = 2 * np.pi * max(1, round(Ly * Re / 100)) / Ly
        alpha = 2 * np.pi * max(1, round(Lx * Re / 400)) / Lx
        wavy = 0.5 * jnp.sin(alpha * X)
        # The two walls get independent spanwise phases.
        th_lo = beta * Y + wavy
        th_up = beta * Y - wavy + np.pi / 2

        def streak(yp: Array) -> Array:
            """Near-wall envelope: 0 at the wall, peaking at 1 at y+ = 15."""
            return yp / 15 * jnp.exp(1 - yp / 15)

        def vortex(yp: Array) -> Array:
            """Like `streak` but with a double zero, so w_z = 0 at the wall too."""
            return (yp / 30) ** 2 * jnp.exp(2 * (1 - yp / 30))

        # u' = A f cos(theta) has wall-normal vorticity g = -u'_y = A beta f
        # sin(theta). The vortices lift fluid away from each wall (w > 0 at the
        # lower, w < 0 at the upper) over its low-speed streaks, cos(theta) = -1.
        g_p = (
            streak_amp
            * beta
            * (streak(yp_lo) * jnp.sin(th_lo) + streak(yp_up) * jnp.sin(th_up))
        )
        w_p = -vortex_amp * (
            vortex(yp_lo) * jnp.cos(th_lo) - vortex(yp_up) * jnp.cos(th_up)
        )
        w_hat = self.VB.forward(w_p)
        g_hat = self.VD.forward(g_p)

        key_w, key_g = jax.random.split(jax.random.key(seed))
        w_hat = w_hat + self._noise(key_w, self.VB, noise_amp * float(abs(w_p).max()))
        g_hat = g_hat + self._noise(key_g, self.VD, noise_amp * float(abs(g_p).max()))

        z1 = self.D1.mesh()
        u0 = self.D1.forward(reichardt((1 - jnp.abs(z1)) * Re))
        v0 = jnp.zeros(self.D1.num_dofs)
        return self._zero_nyquist((w_hat, g_hat, u0, v0))

    def _noise(self, key: Array, space: Any, amplitude: float) -> Array:
        """Return low-pass random coefficients in `space`, of max `amplitude`.

        Drawn as a real physical field, so the half spectrum is Hermitian, and
        filtered in composite coefficients, so the boundary conditions hold.
        """
        shape = tuple(len(m) for m in space.mesh(broadcast=False))
        field = jax.random.normal(key, shape)
        c = space.forward(field)
        kx = jnp.arange(c.shape[0])[:, None, None]
        ky = jnp.abs(jnp.fft.fftfreq(c.shape[1], 1 / c.shape[1]))[None, :, None]
        kz = jnp.arange(c.shape[2])[None, None, :]
        # The (0,0) mode is left empty: the solver needs it at zero in w and g.
        keep = (kx <= 8) & (ky <= 8) & (kz < c.shape[2] // 2) & (kx + ky > 0)
        c = jnp.where(keep, c, 0)
        return c * amplitude / float(abs(space.backward(c)).max())

    # -- averages ----------------------------------------------------------

    def plane_moments(self, state: tuple[Array, ...]) -> Array:
        """Return plane averages of the velocity moments, shape (10, Nz).

        Rows are named by `MOMENTS`: the three mean velocities, the six second
        moments and the mean shear dU/dz, at the wall-normal quadrature points.
        The second moments are raw; `ChannelStatistics` subtracts the means.
        """
        w_hat, g_hat, u0, v0 = state[:4]
        u_hat, v_hat = self.velocity(w_hat, g_hat, u0, v0)
        cw = self.VB.to_orthogonal(w_hat)
        cu, cv = jax.vmap(self.VD.to_orthogonal)(jnp.stack((u_hat, v_hat)))
        u, v, w = self._horizontal(*self._wall_normal(cu, cv, cw))
        dudz = self.D1.backward_primitive(u0, k=1, N=self.pad[2])
        products = jnp.stack((u, v, w, u * u, v * v, w * w, u * v, u * w, v * w))
        return jnp.concatenate((products.mean(axis=(1, 2)), dudz[None]))

    def z_mean(self, profile: Array) -> Array:
        """Return the wall-normal average of a profile on the quadrature points.

        Unweighted, whatever the basis, and exact for polynomials of degree
        below the number of points; see `integration_weights`.
        """
        return 0.5 * jnp.sum(self.q_z * profile)

    def wall_units(self, state: tuple[Array, ...]) -> dict[str, float]:
        """Return the measured friction and bulk quantities of the mean flow.

        u_tau is measured at each wall from the shear of the mean profile u0,
        and reported as the average of the two. C_f = 2/U_b^2 because the
        wall shear is 1 in these units.
        """
        u0 = state[2]
        shear = self.Dx @ u0
        nu = float(self.nu)
        u_tau = 0.5 * (jnp.sqrt(nu * abs(shear[0])) + jnp.sqrt(nu * abs(shear[1])))
        U_b = float(self.z_mean(self.D1.backward(u0, N=self.pad[2])))
        return {
            "u_tau": float(u_tau),
            "Re_tau": float(u_tau) * float(self.Re_tau),
            "U_b": U_b,
            "C_f": 2.0 / U_b**2,
        }

    def extra_diagnostics(self, state: tuple[Array, ...]) -> dict[str, float]:
        """Add the measured Re_tau and bulk velocity to `diagnostics`."""
        wall = self.wall_units(state)
        return {"Re_tau(meas)": wall["Re_tau"], "U_b": wall["U_b"]}


def _member(enum: type[Any], value: str) -> Any:
    """Return the member `value` names, as a case file may spell it."""
    try:
        return enum.coerce(value)
    except ValueError:
        return enum.coerce(value.upper())


def _tableau(name: str) -> IMEXTableau:
    """Return the IMEX tableau `jaxfun.integrators` exports as `name`."""
    tableau = getattr(integrators, name, None)
    if not isinstance(tableau, IMEXTableau):
        valid = sorted(
            k for k, v in vars(integrators).items() if isinstance(v, IMEXTableau)
        )
        raise ValueError(f"unknown tableau {name!r}; expected one of {valid}")
    return tableau


def build_solver(case: ChannelCase, **overrides: Any) -> TurbulentChannel:
    """Build the solver `case` describes; `overrides` replace case fields."""
    case = replace(case, **overrides) if overrides else case
    return TurbulentChannel(
        case.Nx,
        case.Ny,
        case.Nz,
        case.Lx,
        case.Ly,
        case.Re_tau,
        padding=case.padded,
        # See "CHOICE OF BASIS AND TEST SPACE" in ChannelFlow3D.py for the
        # pairing rule.
        polynomial=_member(PolynomialKind, case.polynomial),
        kind=_member(TestSpaceKind, case.kind),
        tableau=_tableau(case.tableau),
    )


@jax.jit
def plane_moments(solver: TurbulentChannel, state: tuple[Array, ...]) -> Array:
    """Compiled `TurbulentChannel.plane_moments`, with the solver traced."""
    return solver.plane_moments(state)


@jax.jit
def physical_velocity(
    solver: TurbulentChannel, state: tuple[Array, ...]
) -> tuple[Array, ...]:
    """Compiled (u, v, w) on the quadrature mesh, with the solver traced.

    Unpadded on purpose: `pad=None` evaluates the spectral field exactly at the
    points `VD.mesh()` returns, which is the mesh the snapshot file stores.
    """
    return solver.velocity_from_state(state, kind=VelocityKind.PHYSICAL)


def fluctuation_energy(solver: TurbulentChannel, moments: Array) -> float:
    """Return the volume-averaged turbulent kinetic energy of one sample."""
    m = dict(zip(MOMENTS, moments, strict=True))
    k = 0.5 * (m["uu"] - m["u"] ** 2 + m["vv"] - m["v"] ** 2 + m["ww"] - m["w"] ** 2)
    return float(solver.z_mean(k))


class ChannelStatistics:
    """Running time average of plane moments, reported in wall units.

    Holds plain host arrays, so it lives outside the solver. Averages are over
    time and the horizontal planes together, and fluctuations are taken about
    that joint mean.
    """

    def __init__(self, z: Array, Re_tau: float) -> None:
        self.z = np.asarray(z)
        self.Re_tau = float(Re_tau)
        self.reset()

    def reset(self) -> None:
        """Discard every sample."""
        self.sums = np.zeros((len(MOMENTS), len(self.z)))
        self.count = 0
        self.t_first: float | None = None
        self.t_last: float | None = None

    def sample(self, moments: Array, t: float) -> None:
        """Add one sample of `TurbulentChannel.plane_moments`, taken at `t`."""
        self.sums += np.asarray(moments)
        self.count += 1
        if self.t_first is None:
            self.t_first = t
        self.t_last = t

    def mean(self) -> dict[str, np.ndarray]:
        """Return the time average of every moment, keyed as `MOMENTS`."""
        if not self.count:
            raise ValueError("no samples yet")
        return dict(zip(MOMENTS, self.sums / self.count, strict=True))

    def profiles(self) -> dict[str, np.ndarray]:
        """Return the averaged profiles over one half channel, in wall units.

        The two halves are folded together: the mean velocity and the normal
        stresses are even in z, the shear stress <u'w'> odd. Keys: `y+`, `U+`,
        `urms+`, `vrms+`, `wrms+` (streamwise, spanwise, wall-normal), `-uw+`
        and `total`, the total shear stress nu*dU/dz - <u'w'>.
        """
        m = self.mean()

        def even(f: np.ndarray) -> np.ndarray:
            return 0.5 * (f + f[::-1])

        def odd(f: np.ndarray) -> np.ndarray:
            return 0.5 * (f - f[::-1])

        lower = self.z < 0
        U = m["u"]
        uw = m["uw"] - m["u"] * m["w"]
        return {
            "y+": ((1 + self.z) * self.Re_tau)[lower],
            "U+": even(U)[lower],
            "urms+": np.sqrt(even(m["uu"] - U**2))[lower],
            "vrms+": np.sqrt(even(m["vv"] - m["v"] ** 2))[lower],
            "wrms+": np.sqrt(even(m["ww"] - m["w"] ** 2))[lower],
            "-uw+": -odd(uw)[lower],
            "total": odd(m["dudz"] / self.Re_tau - uw)[lower],
        }

    def total_stress_error(self) -> float:
        """Return max |nu*dU/dz - <u'w'> + z|, which vanishes once converged."""
        m = self.mean()
        total = m["dudz"] / self.Re_tau - (m["uw"] - m["u"] * m["w"])
        return float(np.abs(total + self.z).max())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy, for a checkpoint."""
        return {
            "sums": self.sums.tolist(),
            "count": self.count,
            "t_first": self.t_first,
            "t_last": self.t_last,
        }

    def load_dict(self, d: dict[str, Any]) -> None:
        """Restore what `to_dict` returned."""
        self.sums = np.asarray(d["sums"], dtype=float)
        self.count = int(d["count"])
        self.t_first, self.t_last = d["t_first"], d["t_last"]

    def save(self, path: str | os.PathLike) -> None:
        """Write the statistics to an .npz file, which `load` reads back.

        Besides what `load` needs (`z`, `Re_tau`, `sums`, `count`, `t_first`,
        `t_last`), the file holds the `profiles` under their own keys once there
        are samples, so `np.load(path)["U+"]` works without this class. A time
        that is not set yet is stored as NaN.
        """
        data: dict[str, Any] = {
            "z": self.z,
            "Re_tau": self.Re_tau,
            "sums": self.sums,
            "count": self.count,
            "t_first": np.nan if self.t_first is None else self.t_first,
            "t_last": np.nan if self.t_last is None else self.t_last,
        }
        if self.count:
            data |= self.profiles()
        np.savez(path, **data)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChannelStatistics":
        """Return the statistics `save` wrote to `path`."""

        def time(t: np.ndarray) -> float | None:
            return None if np.isnan(t) else float(t)

        with np.load(path) as d:
            stats = cls(d["z"], float(d["Re_tau"]))
            stats.load_dict(
                {
                    "sums": d["sums"],
                    "count": d["count"],
                    "t_first": time(d["t_first"]),
                    "t_last": time(d["t_last"]),
                }
            )
        return stats


# -- checkpoints ---------------------------------------------------------------

FIELDS = ("w_hat", "g_hat", "u0", "v0")


def state_shapes(solver: TurbulentChannel) -> tuple[tuple[int, ...], ...]:
    """Return the array shape of each field of `solver`'s state."""
    n1 = (solver.D1.num_dofs,)
    return (tuple(solver.VB.num_dofs), tuple(solver.VD.num_dofs), n1, n1)


def regrid(
    state: tuple[Array, ...], old_grid: tuple[int, int, int], new: TurbulentChannel
) -> tuple[Array, ...]:
    """Return `state`, saved on `old_grid` = ((Nx, Ny, Nz)), on `new`'s grid.

    Truncates or zero-pads the composite coefficients, which keeps the boundary
    conditions exact. The Nyquist mode of the smaller grid is dropped on both
    Fourier axes, because the solver holds it at zero. Also used when only the
    device count differs, which changes the padding of the half spectrum.
    """
    Nxo, Nyo, _ = old_grid
    Nxn, Nyn, _ = new.grid
    kx = min(Nxo, Nxn) // 2  # rows 0 .. kx-1 of the half spectrum
    ky = min(Nyo, Nyn) // 2  # wavenumbers -(ky-1) .. ky-1 of the full one
    shapes = state_shapes(new)

    def resize3(a: Array, shape: tuple[int, ...]) -> Array:
        nz = min(a.shape[2], shape[2])
        out = jnp.zeros(shape, dtype=a.dtype)
        out = out.at[:kx, :ky, :nz].set(a[:kx, :ky, :nz])
        if ky > 1:
            out = out.at[:kx, -(ky - 1) :, :nz].set(a[:kx, -(ky - 1) :, :nz])
        return out

    def resize1(a: Array, shape: tuple[int, ...]) -> Array:
        n = min(a.shape[0], shape[0])
        return jnp.zeros(shape, dtype=a.dtype).at[:n].set(a[:n])

    def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
        return (
            resize3(state[0], shapes[0]),
            resize3(state[1], shapes[1]),
            resize1(state[2], shapes[2]),
            resize1(state[3], shapes[3]),
        )

    shardings = tuple(state_sharding(s) for s in shapes)
    return jax.jit(body, out_shardings=shardings)(tuple(state))


class ChannelCheckpointer:
    """Asynchronous, sharded checkpoints of a turbulent channel run.

    Each checkpoint holds the state, the time, a step counter, the running
    statistics and the configuration. Every process writes and reads only its
    own shards of the state.
    """

    def __init__(self, directory: str | os.PathLike, max_to_keep: int = 3) -> None:
        self.directory = os.path.abspath(directory)
        self.manager = ocp.CheckpointManager(
            self.directory,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep, enable_async_checkpointing=True
            ),
        )

    def latest_step(self) -> int | None:
        """Return the step of the newest complete checkpoint, if any."""
        return self.manager.latest_step()

    def save(
        self,
        step: int,
        solver: TurbulentChannel,
        state: tuple[Array, ...],
        t: float,
        stats: ChannelStatistics,
        case: ChannelCase,
    ) -> None:
        """Start writing a checkpoint and return before it is on disk.

        `case` is stored with it, as a record of what produced the run; a
        restart does not read it back.
        """
        meta = {
            "t": t,
            "step": step,
            "grid": list(solver.grid),
            "shapes": [list(s.shape) for s in state[:4]],
            "Lx": float(solver.Lx),
            "Ly": float(solver.Ly),
            "Re_tau": float(solver.Re_tau),
            "polynomial": solver.polspace.__name__,
            "kind": str(solver.testkind),
            "stats": stats.to_dict(),
            "case": to_dict(case),
        }
        self.manager.save(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(dict(zip(FIELDS, state[:4], strict=True))),
                meta=ocp.args.JsonSave(meta),
            ),
        )

    def meta(self, step: int | None = None) -> dict[str, Any]:
        """Return a checkpoint's metadata, without reading the state.

        The keys are those `save` writes: the time, the step, the grid, the box,
        the statistics (as `ChannelStatistics.to_dict`) and the case.
        """
        step = self.manager.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"no checkpoint in {self.directory}")
        return self.manager.restore(
            step, args=ocp.args.Composite(meta=ocp.args.JsonRestore())
        )["meta"]

    def restore(
        self,
        solver: TurbulentChannel,
        stats: ChannelStatistics,
        step: int | None = None,
    ) -> tuple[tuple[Array, ...], float, int, bool]:
        """Restore a checkpoint into `solver`'s layout.

        Loads `stats` too when the checkpoint was written on the same grid and
        at the same Re_tau, and leaves it untouched otherwise.

        Args:
            solver: The solver to restart; its grid may differ from the saved one.
            stats: Receives the saved statistics when they are compatible.
            step: Which checkpoint; the latest when omitted.

        Returns:
            The state, the time, the step counter, and whether the statistics
            were restored.
        """
        meta = self.meta(step)
        step = meta["step"]
        for key, have in (
            ("Lx", float(solver.Lx)),
            ("Ly", float(solver.Ly)),
            ("polynomial", solver.polspace.__name__),
        ):
            if meta[key] != have:
                raise ValueError(
                    f"checkpoint {key}={meta[key]!r} does not match this run's {have!r}"
                )

        dtypes = (complex, complex, float, float)
        target = {
            name: jax.ShapeDtypeStruct(
                tuple(shape), jnp.dtype(dt), sharding=state_sharding(tuple(shape))
            )
            for name, shape, dt in zip(FIELDS, meta["shapes"], dtypes, strict=True)
        }
        saved = self.manager.restore(
            step, args=ocp.args.Composite(state=ocp.args.StandardRestore(target))
        )["state"]
        state = tuple(saved[name] for name in FIELDS)
        same_grid = tuple(meta["grid"]) == tuple(solver.grid)
        if tuple(s.shape for s in state) != state_shapes(solver):
            state = regrid(state, tuple(meta["grid"]), solver)

        compatible = same_grid and meta["Re_tau"] == float(solver.Re_tau)
        if compatible:
            stats.load_dict(meta["stats"])
        return state, float(meta["t"]), int(meta["step"]), compatible

    def close(self) -> None:
        """Wait for pending writes and release the manager."""
        self.manager.wait_until_finished()
        self.manager.close()


# -- driver --------------------------------------------------------------------


def run(
    solver: TurbulentChannel,
    state: tuple[Array, ...],
    t: float,
    step: int,
    case: ChannelCase,
    stats: ChannelStatistics,
    checkpointer: ChannelCheckpointer | None,
    *,
    averaging_from: float,
    snapshots: HDF5File | None = None,
) -> tuple[tuple[Array, ...], float, int]:
    """Advance from `t` to `case.t_end`, sampling statistics and checkpointing.

    Integrates in chunks of `case.sample_every` steps. After each chunk the
    plane moments are sampled once `t` has reached `averaging_from`; every
    `case.checkpoint_every` chunks a checkpoint is started and every
    `case.snapshot_every` chunks a velocity snapshot is written. Returns early if
    the state stops being finite.
    """
    dt, every = case.dt, case.sample_every
    chunks = max(0, int(round((case.t_end - t) / (dt * every))))
    wall_time = time.time()
    for i in range(1, chunks + 1):
        state = solver.solve(
            dt,
            steps=every,
            state0=state,
            trange=(t, t + every * dt),
            n_batches=1,
            progress=False,
        )
        step += every
        t = t + every * dt
        if not all(bool(jnp.isfinite(s).all()) for s in state):
            echo(f"  t={t:.3f}: the state is no longer finite; stopping")
            break
        # A collective; every process has to reach it.
        moments = to_host(plane_moments(solver, state))
        if t >= averaging_from - 1e-12:
            stats.sample(moments, t)
        if i % case.log_every == 0 or i == chunks:
            wall = solver.wall_units(state)
            echo(
                f"  t={t:8.3f}  Re_tau={wall['Re_tau']:7.2f}  U_b={wall['U_b']:6.3f}"
                f"  k={fluctuation_energy(solver, moments):.3e}"
                f"  CFL={solver.courant(state, dt):.2f}  samples={stats.count}"
                f"  ({(time.time() - wall_time) / (case.log_every * every):.3f}"
                " s/step)"
            )
            wall_time = time.time()
        if case.snapshot_every and (i % case.snapshot_every == 0 or i == chunks):
            # Another collective. The interval is tested against `case`, which
            # every process has, and not against `snapshots`, which only the
            # leader has: a process that skipped `to_host` would hang the rest.
            u, v, w = to_host(physical_velocity(solver, state))
            if snapshots is not None:
                snapshots.write({"U": np.stack((u, v, w))}, time=t, step=step)
        if checkpointer is not None and (i % case.checkpoint_every == 0 or i == chunks):
            checkpointer.save(step, solver, state, t, stats, case)
    return state, t, step


def plot(stats: ChannelStatistics, Re_tau: float, path: str | os.PathLike) -> None:
    """Plot the averaged profiles against the law of the wall."""
    import matplotlib.pyplot as plt

    p = stats.profiles()
    yp = p["y+"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.semilogx(yp, p["U+"], "k", label="U+")
    ax.semilogx(yp[yp < 12], yp[yp < 12], "b--", label="y+")
    log = yp[yp > 20]
    ax.semilogx(log, np.log(log) / 0.41 + 5.2, "r--", label="log law")
    ax.set_xlabel("y+")
    ax.set_title(f"Mean velocity, Re_tau = {Re_tau:g}")
    ax.legend()
    ax = axes[1]
    for key in ("urms+", "vrms+", "wrms+"):
        ax.plot(yp, p[key], label=key)
    ax.set_xlabel("y+")
    ax.set_title("RMS velocity fluctuations")
    ax.legend()
    ax = axes[2]
    z = yp / Re_tau - 1
    ax.plot(yp, p["-uw+"], label="-<u'w'>+")
    ax.plot(yp, p["total"], label="total shear stress")
    ax.plot(yp, -z, "k--", label="exact total, -z")
    ax.set_xlabel("y+")
    ax.set_title("Shear stress")
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.show()


def check(solver: TurbulentChannel, state: tuple[Array, ...]) -> None:
    """Assert the solver's structural invariants."""
    d = solver.diagnostics(state)
    echo("  " + "  ".join(f"{k}={v:.3e}" for k, v in d.items()))
    assert all(bool(jnp.isfinite(s).all()) for s in state)
    assert d["div"] < 1e-10, f"divergence not satisfied: {d['div']:.3e}"
    assert d["w[k=0]"] < 1e-12 * d["max|w|"], "the (0,0) mode of w must not be driven"
    assert d["g[k=0]"] < 1e-12 * d["max|u|"], "the (0,0) mode of g must not be driven"


def pytest_checks(
    solver: TurbulentChannel,
    state: tuple[Array, ...],
    t: float,
    step: int,
    stats: ChannelStatistics,
    case: ChannelCase,
) -> None:
    """Checkpoint round-trip, regrid identity and statistics sanity."""
    check(solver, state)
    assert stats.count > 0, "no statistics were collected"
    for key, value in stats.profiles().items():
        assert value.shape == (int((stats.z < 0).sum()),), key
        assert np.isfinite(value).all(), key

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ChannelCheckpointer(os.path.join(tmp, "ckpt"))
        ckpt.save(step, solver, state, t, stats, case)
        ckpt.manager.wait_until_finished()
        loaded = ChannelStatistics(jnp.asarray(stats.z), stats.Re_tau)
        state1, t1, step1, restored = ckpt.restore(solver, loaded)
        ckpt.close()
    assert restored and t1 == t and step1 == step
    for a, b in zip(state, state1, strict=True):
        assert bool(jnp.array_equal(a, b)), "the checkpoint round-trip is not exact"
    assert np.array_equal(loaded.sums, stats.sums) and loaded.count == stats.count
    echo("  checkpoint round-trip exact")

    fine = build_solver(case, Nx=case.Nx + 8, Ny=case.Ny + 8, Nz=case.Nz + 8)
    back = regrid(regrid(state, solver.grid, fine), fine.grid, solver)
    for a, b in zip(state, back, strict=True):
        assert bool(jnp.array_equal(a, b)), "regrid up and back is not the identity"
    echo("  regrid up and back exact")


def main(case: ChannelCase, args: Any) -> KMM3D:
    """Run, or continue, the turbulent channel simulation `case` describes.

    Args:
        case: The case, with any command-line overrides already applied.
        args: The remaining run options from `channel_case.parse_cli`:
            `restart_from`, `step`, `average_from`, `reset_stats`.
    """
    solver = build_solver(case)
    echo(
        f"Re_tau={case.Re_tau:g}  box {case.Lx:.3f} x {case.Ly:.3f} x 2"
        f"  grid {case.Nx} x {case.Ny} x {case.Nz}  dt={case.dt}"
        f"  devices={jax.device_count()}  sharded={bool(solver.sharded)}"
    )
    if args.case is not None:
        echo(f"  case {os.path.abspath(args.case)}")
    z = solver.VD.mesh(N=solver.pad, broadcast=False)[2]
    stats = ChannelStatistics(to_host(z), case.Re_tau)

    in_pytest = "PYTEST" in os.environ
    checkpointer = (
        None if in_pytest else ChannelCheckpointer(case.path("checkpoint_dir"))
    )

    snapshots = None
    if not in_pytest and case.snapshot_every:
        # `to_host` is a collective, so every process computes the mesh; only
        # the leader goes on to open the file.
        mesh = to_host(solver.VD.mesh(broadcast=False))
        if is_leader():
            snapshots = HDF5File.from_coords(
                case.path("snapshot_file"),
                mesh,
                domains=[(0.0, case.Lx), (0.0, case.Ly), (-1.0, 1.0)],
                dtype=np.float64 if case.snapshot_float64 else np.float32,
                wrap_axes=(0, 1) if case.snapshot_closed else (),
                wall_axes=(2,) if case.snapshot_closed else (),
            )
            echo(f"  snapshots -> {snapshots.xdmf_path}")

    source = (
        ChannelCheckpointer(args.restart_from)
        if args.restart_from is not None
        else checkpointer
    )
    average_from = case.t_transient
    if source is not None and source.latest_step() is not None:
        state, t, step, restored = source.restore(solver, stats, args.step)
        echo(f"  restarted from {source.directory} at t={t:.3f} (step {step})")
        if not restored:
            echo("  new grid or Re_tau: statistics start over")
            average_from = max(case.t_transient, t + case.regrid_transient)
        if source is not checkpointer:
            source.close()
    else:
        state, t, step = solver.turbulent_initial_state(seed=case.seed), 0.0, 0
        echo("  fresh start from the turbulent initial condition")
    if args.reset_stats:
        stats.reset()
    if args.average_from is not None:
        average_from = args.average_from

    state, t, step = run(
        solver,
        state,
        t,
        step,
        case,
        stats,
        checkpointer,
        averaging_from=average_from,
        snapshots=snapshots,
    )
    if checkpointer is not None:
        checkpointer.close()
    if snapshots is not None:
        snapshots.close()

    if in_pytest:
        pytest_checks(solver, state, t, step, stats, case)
        sys.exit(0)

    check(solver, state)
    wall = solver.wall_units(state)
    echo(
        f"  final  Re_tau={wall['Re_tau']:.2f}  U_b={wall['U_b']:.3f}"
        f"  C_f={wall['C_f']:.3e}"
    )
    if not stats.count:
        echo("  no statistics collected yet")
        return solver
    echo(
        f"  statistics over t in [{stats.t_first:.2f}, {stats.t_last:.2f}]"
        f" ({stats.count} samples): total stress error"
        f" {stats.total_stress_error():.3e}"
    )
    if is_leader():
        plot(stats, case.Re_tau, case.path("plot"))

    return solver


if __name__ == "__main__":
    if "PYTEST" in os.environ:
        case, args = pytest_case(), parse_cli([])[1]
    else:
        case, args = parse_cli()
    solver = main(case, args)
