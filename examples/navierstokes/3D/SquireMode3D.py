# Verification of the 3D channel solver: the pieces the 2D solver does not have
#
# ChannelFlow3D.py adds two things to the two-dimensional method: a transport
# equation for the wall-normal vorticity, and a 2x2 recovery of both horizontal
# velocities in place of a single continuity solve. This demo checks exactly
# those, cheaply enough to run on every change, in two stages.
#
# ONE SHOT, NO TIME STEPPING
#
# The first check does not integrate anything. It builds a manufactured field
# that is divergence-free by construction and satisfies the boundary conditions,
# differentiates H = omega x u symbolically with sympy, and compares every
# assembled right-hand side against it:
#
#   the 2x2 recovery       against the analytic u and v
#   NL_w                   against d/dz(H_x,x + H_y,y) - H_z,xx - H_z,yy
#   NL_g                   against H_x,y - H_y,x
#   NL_u0, NL_v0           against -<H_x>, -<H_y>
#   the (0,0) pin          against the profile it is asked to inject
#
# This is what catches a wrong sign or a swapped axis, and it catches it in
# seconds with no chance of a plausible-looking wrong answer: the streamwise and
# spanwise wavenumbers are deliberately different (2 and 3), so a solver that
# confuses the two directions cannot pass. Everything here is a polynomial of low
# degree times a single Fourier mode, and the quadratic products land on modes
# well inside the grid, so the comparison is exact rather than merely converged
# and the tolerances below are round-off, not truncation.
#
# A SQUIRE MODE, WHICH DECAYS EXACTLY
#
# The second check integrates the one initial condition whose evolution is known
# in closed form. Take a quiescent channel -- no base flow, no body force -- and
# seed the wall-normal vorticity alone:
#
#   g = A*cos(kx*x + ky*y)*sin(pi*(z+1)/2),   w = 0,   u0 = v0 = 0
#
# The horizontal velocity the recovery returns is u_h = (-ky, kx)*G(z)*sin(theta)
# /(kx^2 + ky^2), which is perpendicular to the wavevector. That is what makes
# this exact: u_h depends on position only through theta = kx*x + ky*y, and
# u_h is orthogonal to Grad(theta), so (u_h⋅Grad_h) of anything vanishes
# identically -- and with w = 0 the whole convective derivative (u⋅Grad)u is
# exactly zero, not merely small. Then
#
#   H = omega x u = (u⋅Grad)u - Grad(|u|^2/2) = -Grad(|u|^2/2)
#
# is a pure gradient, and both forcing combinations annihilate gradients term by
# term: H_x,y - H_y,x is the curl of a gradient, and d/dz(H_x,x + H_y,y) -
# H_z,xx - H_z,yy is Lap_h(phi_z) - Lap_h(phi_z). So the nonlinear terms are not
# small here, they cancel -- which is a much sharper statement about the sign and
# axis relations in `explicit_terms` than any tolerance on a small residual.
#
# What is left is linear diffusion of a Laplacian eigenmode, so
#
#   g(t) = g(0)*exp(-nu*(kx^2 + ky^2 + (pi/2)^2)*t)
#
# and w, u0 and v0 must stay at round-off while it happens. The reference rate
# above is the analytic one; the discretization's own eigenvalue differs from it
# by 3.3e-13 at N=32, which is far below the time-integration error, so the
# tolerance is set by ARS443 and not by the mesh.
#
# WHAT THIS DEMO DOES NOT COVER
#
# The biharmonic w equation is never excited here -- w stays zero by design --
# and neither is advection by a base flow. OrrSommerfeld3D.py does both.
#
# Spatial discretization: Fourier x Fourier x (Legendre Galerkin | Chebyshev PG)
# Time discretization: any globally stiffly accurate IMEX Runge-Kutta tableau
# ruff: noqa: E402
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
# `spmd_bootstrap` sits one level up, shared with the 2D solver; ChannelFlow3D is
# alongside.
sys.path[:0] = [_here, os.path.dirname(_here)]

import jax

# Before any jaxfun import, so nothing is built at the wrong precision. The
# cancellations checked below are asserted at round-off, which in float32 they
# are not. It has to sit at column 0 -- tests/test_demos.py finds the
# float64-only demos by looking for exactly that.
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

from jaxfun.typing import Array, PolynomialKind, TestSpaceKind
from jaxfun.utils.common import lambdify

# The manufactured check is unpadded on purpose: its quadratic products land on
# modes well inside the grid either way, and with no padding there is nothing
# between the analytic field and the assembled operators to explain a
# discrepancy. 2 and 3 rather than 1 and 1 so that an x/y confusion cannot pass.
M_CHECK, MY_CHECK, N_CHECK = 16, 16, 24
KX_CHECK, KY_CHECK = 2, 3

# The Squire mode. `pad[1]` is a multiple of 8 so the sharded path engages at 2,
# 4 and 8 devices; see the sharding notes in ChannelFlow3D.py's header.
M, MY, N = 16, 16, 32
PAD = (24, 24, 32)
KX, KY = 1, 2
NU = 0.01
AMPLITUDE = 1e-6
DT, T_END = 0.05, 2.0

POLYNOMIAL = PolynomialKind.LEGENDRE
KIND = TestSpaceKind.GALERKIN

if "PYTEST" in os.environ:
    T_END = 0.5


def manufactured_field(
    solver: KMM3D, kx: int, ky: int
) -> tuple[dict[str, sp.Expr], tuple[Array, Array]]:
    """Return a divergence-free manufactured mode, symbolic and as coefficients.

    The wall-normal velocity and vorticity are chosen freely -- everything else
    follows from them, which is the point: the returned `u` and `v` are what the
    solver's 2x2 recovery has to reproduce, derived here independently of it.

    The two are given different phases (cosine and sine). With the same phase
    every quadratic product of H would be an in-quadrature pair with zero
    horizontal mean, and `NL_u0`/`NL_v0` would be tested against zero -- which
    they pass whatever sign they carry.
    """
    x, y, z = solver.VD.system.base_scalars()
    th = kx * x + ky * y
    kh2 = kx**2 + ky**2
    # w in the biharmonic space: both it and its z-derivative vanish at the
    # walls. g in the Dirichlet space.
    W = (1 - z**2) ** 2
    G = 1 - z**2
    w_e = W * sp.cos(th)
    g_e = G * sp.sin(th)
    # kh2*u = i*kx*w_z + i*ky*g and kh2*v = i*ky*w_z - i*kx*g, taken as real
    # parts with the phases above.
    Wz = sp.diff(W, z)
    u_e = (-kx * Wz * sp.sin(th) + ky * G * sp.cos(th)) / kh2
    v_e = (-ky * Wz * sp.sin(th) - kx * G * sp.cos(th)) / kh2

    omx_e = sp.diff(w_e, y) - sp.diff(v_e, z)
    omy_e = sp.diff(u_e, z) - sp.diff(w_e, x)
    omz_e = sp.diff(v_e, x) - sp.diff(u_e, y)
    fields = {
        "u": u_e,
        "v": v_e,
        "w": w_e,
        "g": g_e,
        "Hx": omy_e * w_e - omz_e * v_e,
        "Hy": omz_e * u_e - omx_e * w_e,
        "Hz": omx_e * v_e - omy_e * u_e,
    }

    # Sanity, before anything is compared against it. `sp.simplify` cannot walk
    # CoordSys base scalars, so the identities are checked on plain symbols.
    a, b, c = sp.symbols("a b c", real=True)
    sub = {x: a, y: b, z: c}
    _u, _v, _w, _g = (fields[k].subs(sub) for k in ("u", "v", "w", "g"))
    assert sp.simplify(sp.diff(_u, a) + sp.diff(_v, b) + sp.diff(_w, c)) == 0, (
        "the manufactured field is not divergence-free"
    )
    assert sp.simplify(sp.diff(_v, a) - sp.diff(_u, b) - _g) == 0, (
        "the manufactured g is not v_x - u_y"
    )

    mesh = solver.VD.mesh(broadcast=True)
    w_hat = solver.VB.forward(lambdify((x, y, z), w_e)(*mesh))
    g_hat = solver.VD.forward(lambdify((x, y, z), g_e)(*mesh))
    return fields, (w_hat, g_hat)


#: Plain wall-normal symbol for the reference profiles below. A `CoordSys` base
#: scalar cannot be used: each space carries its own, so substituting one into
#: another's expression silently does nothing, and plain sympy's `lambdify`
#: cannot print what is left.
_Z = sp.Symbol("zz", real=True)


def horizontal_mean(expr: sp.Expr, solver: KMM3D, Lx: float, Ly: float) -> sp.Expr:
    """Return the exact horizontal average of `expr`, as a function of `_Z`."""
    x, y, z = solver.VD.system.base_scalars()
    a, b = sp.symbols("a b", real=True)
    flat = sp.expand_trig(sp.expand(expr.subs({x: a, y: b, z: _Z})))
    mean = sp.integrate(sp.integrate(flat, (a, 0, Lx)), (b, 0, Ly)) / (Lx * Ly)
    return sp.simplify(mean)


def check_manufactured(Lx: float, Ly: float) -> None:
    """Compare every assembled right-hand side against a symbolic reference."""
    echo("\n-- manufactured mode, one shot ------------------------------------")
    solver = KMM3D(
        M_CHECK,
        MY_CHECK,
        N_CHECK,
        Lx,
        Ly,
        NU,
        polynomial=POLYNOMIAL,
        kind=KIND,
    )
    fields, (w_hat, g_hat) = manufactured_field(solver, KX_CHECK, KY_CHECK)
    x, y, z = solver.VD.system.base_scalars()
    zero = jnp.zeros(solver.D1.num_dofs)

    # -- the 2x2 recovery ----------------------------------------------------
    u_hat, v_hat = solver.velocity(w_hat, g_hat, zero, zero)
    mesh = solver.VD.mesh(broadcast=True)
    for name, got_hat in (("u", u_hat), ("v", v_hat)):
        exact = lambdify((x, y, z), fields[name])(*mesh)
        err = float(
            jnp.abs(solver.VD.backward(got_hat) - exact).max() / jnp.abs(exact).max()
        )
        echo(f"  recovery {name}: rel err {err:.3e}")
        assert err < 1e-12, f"recovered {name} is wrong: {err:.3e}"

    state = (w_hat, g_hat, zero, zero)
    d = solver.diagnostics(state)
    echo(f"  div {d['div']:.3e}   w[k=0] {d['w[k=0]']:.3e}   g[k=0] {d['g[k=0]']:.3e}")
    assert d["div"] < 1e-12, f"divergence not satisfied: {d['div']:.3e}"

    # -- the nonlinear terms, against the symbolic H -------------------------
    NL_w, NL_g, NL_u0, NL_v0, _ = solver.explicit_terms(u_hat, v_hat, w_hat, g_hat, ())
    Hx, Hy, Hz = fields["Hx"], fields["Hy"], fields["Hz"]
    src_w = (
        sp.diff(Hx, x, 1, z, 1)
        + sp.diff(Hy, y, 1, z, 1)
        - sp.diff(Hz, x, 2)
        - sp.diff(Hz, y, 2)
    )
    src_g = sp.diff(Hx, y) - sp.diff(Hy, x)
    for name, got, src, space in (
        ("NL_w", NL_w, src_w, solver.PB),
        ("NL_g", NL_g, src_g, solver.PD),
    ):
        ref = space.scalar_product(
            lambdify((x, y, z), src)(*space.mesh(broadcast=True))
        )
        err = float(jnp.abs(got - ref).max() / jnp.abs(ref).max())
        echo(f"  {name}: rel err {err:.3e}")
        assert err < 1e-11, f"{name} does not match the symbolic source: {err:.3e}"

    # The mean-flow forcing is -<H_x> and -<H_y>, tested in z alone.
    zq = solver.P1.mesh()
    for name, got, H in (("NL_u0", NL_u0, Hx), ("NL_v0", NL_v0, Hy)):
        mean = horizontal_mean(H, solver, Lx, Ly)
        profile = sp.lambdify(_Z, mean, "numpy")(zq)
        ref = solver.P1.scalar_product(-jnp.broadcast_to(profile, zq.shape))
        size = float(jnp.abs(ref).max())
        assert size > 1e-6, (
            f"the {name} reference is ~zero ({size:.3e}), so this assertion "
            "would pass whatever sign the operator carries"
        )
        err = float(jnp.abs(got - ref).max() / size)
        echo(f"  {name}: rel err {err:.3e}  (reference size {size:.3e})")
        assert err < 1e-11, f"{name} does not match -<H>: {err:.3e}"

    # -- the (0,0) pin -------------------------------------------------------
    # Injecting a mean profile must reproduce it exactly and disturb nothing.
    (zd,) = solver.D1.system.base_scalars()
    from jaxfun.galerkin.inner import project

    u0 = jnp.asarray(project(1 - zd**2, solver.D1))
    u_hat_b, _ = solver.velocity(w_hat, g_hat, u0, zero)
    inj = float(jnp.abs(u_hat_b[0, 0].real - u0).max())
    off = float(jnp.abs(u_hat_b - u_hat).at[0, 0].set(0.0).max())
    echo(f"  pin: u_hat[0,0]-u0 {inj:.3e}   max change elsewhere {off:.3e}")
    assert inj < 1e-14, f"the pin did not inject the mean profile: {inj:.3e}"
    assert off == 0.0, f"the pin disturbed other wavenumbers by {off:.3e}"
    echo("  manufactured checks passed")


def squire_state(solver: KMM3D, kx: int, ky: int, amplitude: float):
    """Return the initial state for a pure wall-normal-vorticity mode."""
    x, y, z = solver.VD.system.base_scalars()
    g_e = amplitude * sp.cos(kx * x + ky * y) * sp.sin(sp.pi * (z + 1) / 2)
    g_hat = solver.VD.forward(lambdify((x, y, z), g_e)(*solver.VD.mesh(broadcast=True)))
    zero = jnp.zeros(solver.D1.num_dofs)
    return (
        jnp.zeros(solver.VB.num_dofs, dtype=complex),
        g_hat,
        zero,
        zero,
    )


def main() -> KMM3D:
    Lx, Ly = 2 * float(sp.pi), 2 * float(sp.pi)
    check_manufactured(Lx, Ly)

    echo("\n-- Squire mode, exact decay --------------------------------------")
    dt, t_end = DT, T_END
    solver = KMM3D(
        M,
        MY,
        N,
        Lx,
        Ly,
        NU,
        time=(0.0, t_end),
        padding=PAD,
        polynomial=POLYNOMIAL,
        kind=KIND,
    )
    n_dev = len(jax.devices())
    assert solver.sharded or n_dev == 1, (
        f"{n_dev} devices but the solver took the local path; the sizes here are "
        "meant to keep the distributed path engaged"
    )
    echo(f"  devices {n_dev}, sharded {bool(solver.sharded)}")

    state0 = squire_state(solver, KX, KY, AMPLITUDE)
    # Physical wavenumbers: the domain is 2*pi wide in both directions, so these
    # are the integers above.
    kx_p = 2 * float(sp.pi) * KX / Lx
    ky_p = 2 * float(sp.pi) * KY / Ly
    expected = -NU * (kx_p**2 + ky_p**2 + (float(sp.pi) / 2) ** 2)

    steps, batches = int(round(t_end / dt)), 20
    snaps = solver.solve(
        dt=dt,
        state0=state0,
        n_batches=batches,
        return_batch_snapshots=True,
        progress=is_leader(),
    )
    final = tuple(s[-1] for s in snaps)
    times = snapshot_times(dt, steps, batches)
    rate = growth_rate_of(jnp.abs(snaps[1]).max(axis=(1, 2, 3)), times)
    echo(f"  decay rate {rate:.12e}  expected {expected:.12e}")
    echo(f"  relative error {abs(rate / expected - 1):.3e}")
    assert abs(rate / expected - 1) < 1e-6, (
        f"decay rate off: {rate:.6e} against {expected:.6e}"
    )

    # The amplitude itself, not just its slope.
    decayed = float(jnp.abs(final[1]).max() / jnp.abs(state0[1]).max())
    exact = float(jnp.exp(expected * t_end))
    echo(f"  amplitude ratio {decayed:.12e}  expected {exact:.12e}")
    assert abs(decayed / exact - 1) < 1e-6, "decayed amplitude off"

    d = solver.diagnostics(final)
    echo("  " + "  ".join(f"{k} {v:.3e}" for k, v in d.items()))
    u_p, v_p, w_p = solver.velocity_from_state(
        final, pad=PAD, kind=VelocityKind.PHYSICAL
    )
    scale = float(jnp.abs(u_p).max())
    assert scale > 0.0, "the recovered velocity vanished"
    # w is never driven: H is exactly a gradient for this field, so both the
    # w-equation and mean-flow forcings cancel term by term.
    assert float(jnp.abs(w_p).max()) < 1e-12 * scale, (
        f"w was driven: {float(jnp.abs(w_p).max()):.3e} against {scale:.3e}"
    )
    assert d["max|u0|"] < 1e-14, f"u0 was driven: {d['max|u0|']:.3e}"
    assert d["max|v0|"] < 1e-14, f"v0 was driven: {d['max|v0|']:.3e}"
    assert d["div"] < 1e-12, f"divergence not satisfied: {d['div']:.3e}"
    assert d["w[k=0]"] == 0.0, "the w (0,0) mode must stay exactly zero"
    assert d["g[k=0]"] < 1e-16 * float(jnp.abs(final[1]).max()), (
        "the g (0,0) mode must stay at round-off"
    )
    # The Squire velocity is perpendicular to the horizontal wavevector.
    perp = float(jnp.abs(kx_p * u_p + ky_p * v_p).max())
    echo(f"  |kx*u + ky*v| {perp:.3e}  (max|u| {scale:.3e})")
    assert perp < 1e-12 * scale, f"the recovered velocity is not solenoidal: {perp:.3e}"
    echo("  Squire checks passed")
    return solver


if __name__ == "__main__":
    solver = main()
