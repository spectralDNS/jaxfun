# Verification of the 3D channel solver: an Orr-Sommerfeld eigenmode, twice
#
# The base flow U(z) = 1 - z^2 along a horizontal direction n = (cos th, sin th)
# is an exact steady solution of ChannelFlow3D.py's two mean-flow equations once
# the body force balances its own diffusion, which for a tilted profile means a
# tilted force f = 2*nu*n. Superposing the least-stable Orr-Sommerfeld eigenmode
# in the (n, z) plane at an amplitude small enough to stay linear, the
# perturbation must grow like exp(alfa*Im(c)*t) with c the eigenvalue -- so the
# measured growth rate is a sharp test of the whole solver at once: advection,
# the biharmonic operator, the 2x2 recovery and the mean flow.
#
# WHY THE EIGENVALUE IS THE SAME FOR BOTH ANGLES
#
# Nothing in the configuration depends on the horizontal direction perpendicular
# to n: the base flow is entirely along n, and the seeded mode varies only along
# xi = n⋅x. So the problem in the (xi, z) plane *is* plane Poiseuille flow at the
# same Re and the same alfa, and the eigenvalue is the two-dimensional one --
# which is why both tiers below reuse OrrSommerfeld_eigs.py unchanged, with no
# spanwise wavenumber anywhere in it.
#
# TWO TIERS
#
#   th = 0      The mode is aligned with x and constant in y. This is the 2-D
#               problem embedded in the 3-D solver, and it must reproduce the 2-D
#               solver's answer, that slice of this solver being the 2-D solver.
#               Everything spanwise must stay identically quiet: v, the
#               wall-normal vorticity, and the mean spanwise profile.
#
#   th = pi/4   The same mode rotated into the diagonal, with the base flow and
#               the driving force rotated with it. Same eigenvalue, but now both
#               Fourier axes carry the wave and both rows of the 2x2 recovery are
#               loaded. The sharp check here is that the flow stays planar:
#               u*sin(th) - v*cos(th) must vanish everywhere, for the mean and
#               the perturbation together.
#
# WHAT NEITHER TIER COVERS
#
# A rotated two-dimensional mode has omega_z identically zero -- a plane flow has
# no wall-normal vorticity, whatever plane it lies in. Both tiers therefore
# assert that g stays at round-off, which is a real check on the cancellation of
# H_x,y against H_y,x (the two are individually large), but neither exercises the
# Squire coupling -i*beta*U'*w by which w drives g in a genuinely
# three-dimensional mode. That needs the coupled Orr-Sommerfeld / Squire
# eigenproblem, which OrrSommerfeld_eigs.py does not solve.
#
# SquireMode3D.py covers the vorticity equation from the other side: it drives g
# directly and checks its exact decay, along with every assembled forcing
# operator against a symbolic reference.
#
# Spatial discretization: Fourier x Fourier x (Legendre Galerkin | Chebyshev PG)
# Time discretization: any globally stiffly accurate IMEX Runge-Kutta tableau
# ruff: noqa: E402
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
# `spmd_bootstrap` and `OrrSommerfeld_eigs` sit one level up, shared with the 2D
# solver; ChannelFlow3D is alongside.
sys.path[:0] = [_here, os.path.dirname(_here)]

import jax

# Before any jaxfun import, so nothing is built at the wrong precision. This is
# not optional: the eigenmode is seeded at amplitude 1e-7 on a base flow of order
# 1, and in float32 the measured growth rate comes out negative. It has to sit at
# column 0 -- tests/test_demos.py finds the float64-only demos by looking for
# exactly that.
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_num_cpu_devices", 2)

# Likewise before any jaxfun import, and for a related reason: `jaxfun.sharding`
# builds its device mesh at import time, so the other processes' devices have to
# be visible by then. A no-op outside an MPI launcher or on a single rank.
from spmd_bootstrap import echo, initialize_distributed, is_leader

initialize_distributed()

import jax.numpy as jnp
import sympy as sp
from ChannelFlow3D import KMM3D, VelocityKind, growth_rate_of, snapshot_times
from OrrSommerfeld_eigs import OrrSommerfeld

from jaxfun.galerkin.inner import project
from jaxfun.typing import Array, PolynomialKind, TestSpaceKind

M, MY, N = 32, 32, 96  # Fourier modes (x), Fourier modes (y), wall-normal modes
# Any M and MY run on any number of devices: the half spectrum on axis 0 stores
# M // 2 + 1 coefficients, which is odd for every power-of-two M, and `RFourier`
# pads that up to a multiple of the device count itself. What has to divide is
# the *padded spanwise* count, 3*MY/2, which a power of two times 8 always does.
MY_ALIGNED = 8  # spanwise modes for the aligned tier, which is constant in y
RE, ALFA = 8000.0, 1.0  # Reynolds number, in-plane wavenumber
DT, T_END = 0.02, 50.0
AMPLITUDE = 1e-7  # eigenmode amplitude; small enough that the dynamics stay linear
N_OS = 100  # modes in the Orr-Sommerfeld eigenproblem itself
# Wall-normal basis and test space; see "CHOICE OF BASIS AND TEST SPACE" in
# ChannelFlow3D.py for the pairing rule.
POLYNOMIAL = PolynomialKind.CHEBYSHEV
KIND = TestSpaceKind.PETROV_GALERKIN

if "PYTEST" in os.environ:
    M, MY, N, T_END = 16, 16, 48, 1.0


def periods(alfa: float, theta: float) -> tuple[float, float]:
    """Return the box periods that make the tilted mode a single Fourier mode.

    The physical wavevector is alfa*(cos th, sin th), and a box of period L
    carries wavenumbers 2*pi*n/L, so one period per direction puts the mode at
    index 1 on each axis. At th = 0 the spanwise wavenumber is zero, the mode is
    constant in y, and the spanwise period is arbitrary.
    """
    two_pi = 2 * float(sp.pi)
    Lx = two_pi / (alfa * float(jnp.cos(theta)))
    if theta == 0.0:
        return Lx, two_pi
    return Lx, two_pi / (alfa * float(jnp.sin(theta)))


def orr_sommerfeld_state(
    solver: KMM3D,
    Re: float,
    alfa: float,
    theta: float,
    amplitude: float,
    n_os: int = 100,
    t: float = 0.0,
) -> tuple[tuple[Array, ...], complex, Array, Array]:
    """Return the initial state, the eigenvalue, and the expected u and v.

    The eigenmode is a streamfunction mode in the (xi, z) plane, xi = n⋅x:
    w = -dpsi/dxi = -i*alfa*phi(z)*E and the in-plane horizontal velocity is
    dpsi/dz = phi'(z)*E, whose Cartesian components are that times cos th and
    sin th. Only w and the mean profiles are seeded; the recovery has to produce
    both horizontal perturbations, which is what the returned arrays check.
    """
    problem = OrrSommerfeld(alfa=alfa, Re=Re, N=n_os)
    eigvals, eigvectors = problem.solve()
    xm, ym, zm = solver.VB.mesh(broadcast=False)
    eigval, phi, dphidz = problem.interp(zm, eigvals, eigvectors, eigval=1)
    ct, st = float(jnp.cos(theta)), float(jnp.sin(theta))
    xi = xm[:, None, None] * ct + ym[None, :, None] * st
    wave = jnp.exp(1j * alfa * (xi - eigval * t))
    w_p = amplitude * (-1j * alfa * phi[None, None, :] * wave).real
    in_plane = amplitude * (dphidz[None, None, :] * wave).real

    (zd,) = solver.D1.system.base_scalars()
    base = 1 - zd**2
    state = (
        solver.VB.forward(w_p),
        # A plane flow has no wall-normal vorticity, whatever plane it lies in.
        jnp.zeros(solver.VD.num_dofs, dtype=complex),
        jnp.asarray(project(base * ct, solver.D1)),
        jnp.asarray(project(base * st, solver.D1)),
    )
    return state, complex(eigval), in_plane * ct, in_plane * st


def os_vel(
    solver: KMM3D,
    t: float,
    Re: float,
    alfa: float,
    theta: float,
    amplitude: float,
) -> tuple[Array, Array, Array, complex]:
    """Return the analytic velocity field at time `t`, base flow included."""
    state, eigval, _, _ = orr_sommerfeld_state(
        solver, Re, alfa, theta, amplitude, 128, t
    )
    u_p, v_p, w_p = solver.velocity_from_state(state, kind=VelocityKind.PHYSICAL)
    return u_p, v_p, w_p, eigval


def solution_error(
    solver: KMM3D,
    state: tuple[Array, ...],
    t: float,
    Re: float,
    alfa: float,
    theta: float,
    amplitude: float,
) -> tuple[Array, Array, Array, Array]:
    """Compute the same error metrics as shenfun, extended with w."""
    u, v, w = solver.velocity_from_state(state, kind=VelocityKind.PHYSICAL)
    ex, ey, ez, eigval = os_vel(solver, t, Re, alfa, theta, amplitude)
    w0, w1, w2 = solver.VD.weights()
    weight = w0 * w1 * w2
    e2 = jnp.sum(weight * ((u - ex) ** 2 + (v - ey) ** 2 + (w - ez) ** 2))
    exact = jnp.exp(2 * jnp.imag(alfa * eigval) * t)
    _, _, zj = solver.VD.mesh()
    ct, st = float(jnp.cos(theta)), float(jnp.sin(theta))
    ub, vb = (1 - zj**2) * ct, (1 - zj**2) * st
    e1 = jnp.sum(weight * ((u - ub) ** 2 + (v - vb) ** 2 + w**2))
    ex, ey, ez, _ = os_vel(solver, 0.0, Re, alfa, theta, amplitude)
    e0 = jnp.sum(weight * ((ex - ub) ** 2 + (ey - vb) ** 2 + ez**2))
    return e0, e1, e2, exact


def run(label: str, theta: float, My: int) -> float:
    """Evolve the eigenmode at one angle and return the measured growth rate."""
    dt, t_end, amplitude = DT, T_END, AMPLITUDE
    nu = 1.0 / RE
    Lx, Ly = periods(ALFA, theta)
    ct, st = float(jnp.cos(theta)), float(jnp.sin(theta))
    padding = (3 * M // 2, 3 * My // 2, N)
    solver = KMM3D(
        M,
        My,
        N,
        Lx,
        Ly,
        nu,
        # Tilted with the base flow, so that each mean profile's diffusion is
        # balanced by its own component of the force.
        body_force=(2 * nu * ct, 2 * nu * st),
        time=(0.0, t_end),
        padding=padding,
        kind=KIND,
        polynomial=POLYNOMIAL,
    )
    n_dev = len(jax.devices())
    assert solver.sharded or n_dev == 1, (
        f"{n_dev} devices but the solver took the local path; the sizes here are "
        "meant to keep the distributed path engaged"
    )
    state0, eigval, u_expected, v_expected = orr_sommerfeld_state(
        solver, RE, ALFA, theta, amplitude, N_OS, 0.0
    )
    echo(f"\n== {label} ==")
    echo(
        f"  Re={RE:g} alfa={ALFA:g} theta={theta:.4f}  M={M} My={My} N={N} "
        f"dt={dt} T={t_end}  sharded={bool(solver.sharded)}"
    )
    echo(f"  eigenvalue {eigval:.16f}")

    # Neither horizontal perturbation is seeded; the 2x2 recovery has to produce
    # both. What is left over is the error of projecting the eigenfunction onto N
    # wall-normal modes, not of the recovery, so it converges spectrally and the
    # tolerance has to track N -- measured in 2D at Re=8000, alfa=1:
    #
    #   N        48        64        96        128       160
    #   rel err  1.7e-05   3.4e-08   1.0e-12   7.6e-12   2.2e-11
    #
    # i.e. it hits the round-off floor by N=96, limited by the eigenvector's own
    # conditioning and the biharmonic mass solve inside `VB.forward`.
    zero = jnp.zeros(solver.D1.num_dofs)
    u_hat, v_hat = solver.velocity(state0[0], state0[1], zero, zero)
    tol = 1e-8 if N >= 96 else 1e-3
    for name, got_hat, expected in (
        ("u", u_hat, u_expected),
        ("v", v_hat, v_expected),
    ):
        size = float(jnp.abs(expected).max())
        if size == 0.0:  # the aligned tier has no spanwise perturbation at all
            got = float(jnp.abs(solver.VD.backward(got_hat)).max())
            echo(f"  recovery {name}: identically zero, got {got:.3e}")
            assert got < 1e-20, f"{name} should vanish but is {got:.3e}"
            continue
        err = float(jnp.abs(solver.VD.backward(got_hat).real - expected).max() / size)
        echo(f"  recovery {name}: rel err {err:.3e}")
        assert err < tol, f"the recovery must reproduce the eigenmode's {name}"

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

    # The base flow has no wall-normal velocity, so |w| is pure perturbation.
    rate = growth_rate_of(
        jnp.abs(snaps[0]).max(axis=(1, 2, 3)), snapshot_times(dt, steps, batches)
    )
    expected_rate = ALFA * eigval.imag
    echo(f"  growth rate measured {rate:+.12f}")
    echo(
        f"  linear theory        {expected_rate:+.12f}   "
        f"(rel error {abs(rate / expected_rate - 1):.2e})"
    )

    assert d1["div"] < 1e-10, f"divergence not satisfied: {d1['div']:.3e}"
    # Relative, not exact: the seeded eigenmode's (0,0) Fourier component is
    # analytically zero but comes out of the FFT at round-off, so it starts at
    # ~1e-24 rather than at 0. It is never *driven* -- every term on the w
    # equation's right-hand side carries a d/dx or d/dy -- so it only decays.
    assert d1["w[k=0]"] < 1e-14 * d1["max|w|"], "the (0,0) mode must not be driven"
    u_p, v_p, w_p = solver.velocity_from_state(
        final, pad=solver.pad, kind=VelocityKind.PHYSICAL
    )
    # The whole field -- mean and perturbation together -- stays in the (n, z)
    # plane. A global check on both rows of the recovery at once.
    planar = float(jnp.abs(u_p * st - v_p * ct).max())
    scale = max(float(jnp.abs(u_p).max()), float(jnp.abs(v_p).max()))
    echo(f"  |u*sin(th) - v*cos(th)| {planar:.3e}  (scale {scale:.3e})")
    assert planar < 1e-12 * scale, f"the flow left its plane: {planar:.3e}"

    # A plane flow has no wall-normal vorticity, whatever plane it lies in, and
    # H_x,y and H_y,x are individually large here -- so this is a real check that
    # they cancel rather than a restatement of the initial condition.
    #
    # The reference is the velocity scale, which is the O(1) base flow, and not
    # the perturbation. g starts at exactly zero and its floor is set by
    # round-off in the products that carry the *mean*: H_x contains -omega_z*v
    # and H_y contains +omega_z*u, so an O(eps) omega_z is multiplied by an O(1)
    # velocity. Measured, max|g_hat| comes out at ~1e-18 whatever the eigenmode
    # amplitude -- across A = 1e-7, 1e-6 and 1e-5 it does not move -- so a bound
    # relative to the perturbation would tighten as the amplitude grew and mean
    # nothing. A sign error in either forcing operator drives g to the order of
    # the perturbation's own vorticity, six orders above this bound.
    g_size = float(jnp.abs(final[1]).max())
    echo(f"  max|g_hat| {g_size:.3e}  against the velocity scale {scale:.3e}")
    assert g_size < 1e-14 * scale, f"wall-normal vorticity was driven: {g_size:.3e}"

    # The tilted force holds each mean profile against its own diffusion.
    for name, got, comp in (("u0", final[2], ct), ("v0", final[3], st)):
        prof = solver.D1.backward(got)
        want = (1 - solver.D1.mesh() ** 2) * comp
        err = float(jnp.abs(prof - want).max())
        echo(f"  mean {name}: max dev from (1-z^2)*{comp:.4f} is {err:.3e}")
        assert err < 1e-8, f"the mean profile {name} drifted by {err:.3e}"

    e0, e1, e2, exact = solution_error(solver, final, t_end, RE, ALFA, theta, amplitude)
    echo(
        f"  sqrt(e2) {float(jnp.sqrt(e2)):.3e}   "
        f"e1/e0 - exact {float(abs(e1 / e0 - exact)):.3e}"
    )
    assert jnp.sqrt(e2) < 1e-11, jnp.sqrt(e2)
    assert abs(e1 / e0 - exact) < 1e-5, abs(e1 / e0 - exact)

    if "PYTEST" not in os.environ:
        assert abs(rate / expected_rate - 1) < 0.01, "growth rate off by more than 1%"
    return rate


def main() -> None:
    """Run both tiers and check they agree with each other and with theory."""
    aligned = run("aligned with x (the 2D problem embedded)", 0.0, MY_ALIGNED)
    rotated = run("rotated 45 degrees", float(jnp.pi) / 4, MY)
    echo("\n== summary ==")
    echo(f"  aligned {aligned:+.12f}")
    echo(f"  rotated {rotated:+.12f}")
    # Same eigenvalue, same discretization in the wall-normal direction, so the
    # two must agree far more closely than either agrees with linear theory.
    echo(f"  difference {abs(aligned - rotated):.3e}")
    if "PYTEST" not in os.environ:
        assert abs(aligned / rotated - 1) < 1e-3, (
            "the two orientations must give the same growth rate"
        )


if __name__ == "__main__":
    main()
