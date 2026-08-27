# Verification of the channel solver: an Orr-Sommerfeld eigenmode on Poiseuille
# flow
#
# The base flow U(y) = 1 - y^2 is an exact steady solution of ChannelFlow2D.py's
# mean-flow equation once the body force balances its own diffusion. Superposing
# the least-stable Orr-Sommerfeld eigenmode of OrrSommerfeld_eigs.py at an
# amplitude small enough to stay linear, the perturbation must grow like
# exp(alfa*Im(c)*t) with c the eigenvalue -- so the measured growth rate is a
# sharp test of the whole solver at once: advection, the biharmonic operator,
# continuity and the mean flow. It is what the tableau comparison in
# ChannelFlow2D.py's header was measured on.
#
# The companion demo is RayleighBenard.py, which verifies the same solver against
# the onset of convection instead.
#
# Spatial discretization: Fourier x (Legendre Galerkin | Chebyshev Petrov-Galerkin)
# Time discretization: any globally stiffly accurate IMEX Runge-Kutta tableau
# ruff: noqa: E402
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

# Before any jaxfun import, so nothing is built at the wrong precision. This is
# not optional: the eigenmode is seeded at amplitude 1e-7 on a base flow of order
# 1, and in float32 the measured growth rate comes out negative (-5.5e-04 against
# a theoretical +2.7e-03). It has to sit at column 0 -- tests/test_demos.py finds
# the float64-only demos by looking for exactly that.
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_num_cpu_devices", 2)

# Likewise before any jaxfun import, and for a related reason: `jaxfun.sharding`
# builds its device mesh at import time, so the other processes' devices have to
# be visible by then. A no-op without mpi4py or on a single rank, which is what
# keeps this demo an ordinary script.
from spmd_bootstrap import echo, initialize_distributed, is_leader

initialize_distributed()

import jax.numpy as jnp
import sympy as sp
from ChannelFlow2D import KMM2D, VelocityKind, growth_rate_of, snapshot_times
from OrrSommerfeld_eigs import OrrSommerfeld

from jaxfun.galerkin.inner import project
from jaxfun.typing import Array, PolynomialKind, TestSpaceKind

M, N = 32, 128  # Fourier modes (x), wall-normal modes (y)
# Any M runs on any number of devices: the half spectrum stores M // 2 + 1
# coefficients, which is odd for every power-of-two M, and `RFourier` pads that
# up to a multiple of the device count itself. The padding is empty, so a power
# of two here buys the fast FFT without costing anything to distribute.
RE, ALFA = 8000.0, 1.0  # Reynolds number, streamwise wavenumber
DT, T_END = 0.02, 0.2
AMPLITUDE = 1e-7  # eigenmode amplitude; small enough that the dynamics stay linear
N_OS = 100  # modes in the Orr-Sommerfeld eigenproblem itself
# Wall-normal basis and test space, as in RayleighBenard.py. See "CHOICE OF BASIS
# AND TEST SPACE" in ChannelFlow2D.py: at N=128 Legendre-Galerkin is the faster of
# the two recommended pairings.
POLYNOMIAL = PolynomialKind.LEGENDRE
KIND = TestSpaceKind.GALERKIN

if "PYTEST" in os.environ:
    M, N, T_END = 16, 48, 1.0


def orr_sommerfeld_state(
    solver: KMM2D,
    Re: float,
    alfa: float,
    amplitude: float,
    n_os: int = 100,
    t: float = 0.0,
) -> tuple[tuple[Array, ...], complex, Array]:
    """Return the initial state, the eigenvalue, and phi' for cross-checking."""
    problem = OrrSommerfeld(alfa=alfa, Re=Re, N=n_os)
    eigvals, eigvectors = problem.solve()
    xm, ym = solver.VB.mesh(broadcast=False)
    eigval, phi, dphidy = problem.interp(ym, eigvals, eigvectors, eigval=1)
    wave = jnp.exp(1j * alfa * (xm - eigval * t))[:, None]
    v_p = amplitude * (-1j * alfa * phi[None, :] * wave).real
    (yd,) = solver.D1.system.base_scalars()
    return (
        (solver.VB.forward(v_p), project(1 - yd**2, solver.D1)),
        complex(eigval),
        amplitude * (dphidy[None, :] * wave).real,
    )


def os_vel(
    solver: KMM2D, t: float, Re: float, alfa: float, amplitude: float
) -> tuple[Array, Array, complex]:
    state, eigval, _ = orr_sommerfeld_state(solver, Re, alfa, amplitude, 128, t)
    u_p, v_p = solver.velocity_from_state(state, kind=VelocityKind.PHYSICAL)
    return u_p, v_p, eigval


def solution_error(
    solver: KMM2D,
    state: tuple[Array, ...],
    t: float,
    Re: float,
    alfa: float,
    amplitude: float,
) -> tuple[Array, Array, Array, Array]:
    "Compute same error metrics as Shenfun for comparison."
    uh, vh, u, v = solver.velocity_from_state(state, kind=VelocityKind.BOTH)
    ex, ey, eigval = os_vel(solver, t, Re, alfa, amplitude)
    w0, w1 = solver.VD.weights()
    e2 = jnp.sum(w0 * w1 * ((u - ex) ** 2 + (v - ey) ** 2))
    exact = jnp.exp(2 * jnp.imag(alfa * eigval) * t)
    xi, yj = solver.VD.mesh()
    ux = 1 - yj**2
    e1 = jnp.sum(w0 * w1 * ((u - ux) ** 2 + v**2))
    ex, ey, eigval = os_vel(solver, 0.0, Re, alfa, amplitude)
    e0 = jnp.sum(w0 * w1 * ((ex - ux) ** 2 + ey**2))
    return e0, e1, e2, exact


def main() -> KMM2D:
    """Evolve the eigenmode and compare its growth rate with linear theory."""
    dt, t_end, amplitude = DT, T_END, AMPLITUDE
    nu = 1.0 / RE
    padding = (3 * M // 2, N)
    solver = KMM2D(
        M,
        N,
        2 * float(sp.pi) / ALFA,
        nu,
        body_force=2 * nu,
        time=(0.0, t_end),
        padding=padding,
        kind=KIND,
        polynomial=POLYNOMIAL,
    )
    state0, eigval, u_expected = orr_sommerfeld_state(
        solver, RE, ALFA, amplitude, N_OS, 0.0
    )
    echo(f"Orr-Sommerfeld  Re={RE:g} alfa={ALFA:g}  M={M} N={N} dt={dt} T={t_end}")
    echo(f"  eigenvalue {eigval:.16f}")
    # The streamwise perturbation is not seeded; continuity has to produce it.
    # What is left over is the error of projecting the eigenfunction onto N
    # wall-normal modes, not of the continuity solve, so it converges spectrally
    # and the tolerance has to track N. Measured (Re=8000, alfa=1):
    #
    #   N        48        64        96        128       160
    #   rel err  1.7e-05   3.4e-08   1.0e-12   7.6e-12   2.2e-11
    #
    # i.e. it hits the round-off floor by N=96, limited by the eigenvector's own
    # conditioning and the biharmonic mass solve inside `VB.forward`.
    u_hat = solver.velocity(state0[0], jnp.zeros(solver.D1.num_dofs))
    u_got = solver.VD.backward(u_hat).real
    err = float(jnp.abs(u_got - u_expected).max() / jnp.abs(u_expected).max())
    echo(f"  continuity recovers u = phi'(y)*exp(i*alfa*x) to {err:.3e}")
    assert err < (1e-8 if N >= 96 else 1e-3), (
        "the continuity solve must reproduce the eigenmode's u"
    )
    d0 = solver.diagnostics(state0)
    echo("  initial " + "  ".join(f"{k}={v:.3e}" for k, v in d0.items()))
    steps, batches = int(round(t_end / dt)), 50

    snaps = solver.solve(
        dt=dt,
        state0=state0,
        n_batches=batches,
        return_batch_snapshots=True,
        progress=is_leader(),
    )
    final = tuple(s[-1] for s in snaps)
    d1 = solver.diagnostics(final)
    echo("  final   " + "  ".join(f"{k}={v:.3e}" for k, v in d1.items()))
    echo(f"  Courant = {solver.courant(final, dt):.2f}")
    # The base flow has no wall-normal velocity, so |v| is pure perturbation.
    rate = growth_rate_of(
        jnp.abs(snaps[0]).max(axis=(1, 2)), snapshot_times(dt, steps, batches)
    )
    expected = ALFA * eigval.imag
    echo(f"\n  growth rate measured {rate:+.12f}")
    echo(
        f"  linear theory        {expected:+.12f}   (rel error {abs(rate / expected - 1):.2e})"  # noqa: E501
    )
    assert d1["div"] < 1e-10, f"divergence not satisfied: {d1['div']:.3e}"
    # Relative, not exact: the seeded eigenmode's k=0 Fourier component is
    # analytically zero but comes out of the FFT at round-off, so k=0 starts at
    # ~1e-24 rather than at 0. It is never *driven* -- every term on the v
    # equation's right-hand side carries a d/dx -- so it only decays from there.
    # RayleighBenard, which starts from v = 0 exactly, does hold it at exactly 0.
    assert d1["v[k=0]"] < 1e-14 * d1["max|v|"], "the k=0 mode must not be driven"

    e0, e1, e2, exact = solution_error(solver, final, t_end, RE, ALFA, amplitude)
    assert abs(e1 / e0 - exact) < 1e-6
    assert jnp.sqrt(e2) < 1e-11

    if "PYTEST" not in os.environ:
        assert abs(rate / expected - 1) < 0.01, "growth rate off by more than 1%"

    return solver


if __name__ == "__main__":
    solver = main()
