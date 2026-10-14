# Rayleigh-Benard convection in a periodic channel, as one coupled system
#
#   u_t + Grad(u)⋅u = -Grad(p) + nu*Div(Grad(u)) + T*j
#   T_t + Div(u*T)  = kappa*Div(Grad(T))
#   Div(u)          = 0
#
# on (x, y) in [0, Lx] x (-1, 1), periodic in x, no-slip walls, T(-1) = 1 and
# T(+1) = 0, with nu = sqrt(Pr/Ra) and kappa = 1/sqrt(Ra*Pr).
#
# The same problem as RayleighBenard.py, but in primitive variables: velocity,
# pressure and temperature are the components of one mixed space, and
# CoupledIMEXRungeKutta advances them together. Every term linear in a single
# field is implicit -- diffusion, the pressure gradient, continuity and buoyancy
# -- and the two advection terms are explicit. The pressure has no time
# derivative; it is the Lagrange multiplier of continuity, so the tableau has to
# be globally stiffly accurate (ARS443).
#
# Each implicit stage is one block solve, decoupled per Fourier wavenumber into
# a dense system for (u, v, p, T) along y.
#
# The wall-normal basis is Legendre with Galerkin test functions, and the
# pressure has as many modes as the velocity components. Continuity is then
# tested against every polynomial of the divergence's degree, so Div(u) = 0
# holds pointwise and not only weakly: the discrete velocity is the same
# divergence-free one as in RayleighBenard.py. With N - 2 pressure modes instead,
# a mean wall-normal velocity of the size of the truncation error would appear.
# At k = 0 the constant and the highest pressure mode are undetermined, and are
# pinned.
#
# Chebyshev works too, with Galerkin-recombined test spaces, which keep every
# block banded: those of the velocity and temperature spaces for the momentum
# and temperature rows, and that of the pressure space for continuity; set
# POLYNOMIAL = "chebyshev". The pressure gradient is not
# integrated by parts, which the Chebyshev weight would not allow.
#
# On several devices, or under mpirun, the wavenumbers are split across them, and
# each device factorizes and solves only its own.
#
# Note that this is a monolithic solver and it is not as fast as the segregated one
# in RayleighBenard.py, which should be the method of choice for production runs.
#
# ruff: noqa: E402
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))  # spmd_bootstrap sits one level up

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

# Before any jaxfun import, which builds its device mesh from the devices visible
# then. A no-op outside an MPI launcher.
from spmd_bootstrap import echo, initialize_distributed, is_leader, to_host

initialize_distributed()

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import (
    CartesianProduct,
    Chebyshev,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.Fourier import RFourier
from jaxfun.integrators import ARS443, CoupledIMEXRungeKutta
from jaxfun.operators import Div, Dot, Grad

M, N = 128, 64  # Fourier modes (x), wall-normal modes (y)
LX = 2 * float(sp.pi)
RA, PR = 1e6, 0.7
DT, T_END = 0.025, 60.0
N_SNAPSHOTS = 120
TABLEAU = ARS443
SEED = 0
AMPLITUDE = 1e-3  # initial temperature perturbation
MODE = 0  # the kind of perturbation; see `initial_temperature`
POLYNOMIAL = "chebyshev"

if "PYTEST" in os.environ:
    M, N, T_END, N_SNAPSHOTS = 16, 16, 0.2, 2


def rayleigh_benard(
    M: int, N: int, Lx: float, Ra: float, Pr: float, poly: str, mode: int = MODE
):
    """Return the coupled RB integrator and its spaces `(W, VT)`."""
    nu, kappa = (Pr / Ra) ** 0.5, 1 / (Ra * Pr) ** 0.5
    family = {"legendre": Legendre.Legendre, "chebyshev": Chebyshev.Chebyshev}[poly]
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    F = FunctionSpace(M, RFourier, domain=Domain(0, Lx), name="F")
    D = FunctionSpace(N, family, bcs=hom, name="D")
    Tb = FunctionSpace(N, family, bcs={"left": {"D": 1}, "right": {"D": 0}}, name="Tb")
    P = FunctionSpace(N, family, name="P")
    VD = TensorProduct(F, D, name="VD")
    Q = TensorProduct(F, P, name="Q")
    VT = TensorProduct(F, Tb, name="VT")
    W = CartesianProduct(CartesianProduct(VD, VD, name="V", rank=1), Q, VT, name="W")
    u, p, T = TrialFunction(W, name="upT", transient=True)

    if poly == "chebyshev":
        DG = TensorProduct(F, D.get_testspace("GR", name="DG"), name="DG")
        TG = TensorProduct(F, Tb.get_testspace("GR", name="TG"), name="TG")
        QG = TensorProduct(F, P.get_testspace("GR", name="QG"), name="QG")
        WT = CartesianProduct(CartesianProduct(DG, DG, rank=1), QG, TG, name="WT")
        v, q, s = TestFunction(WT, name="vqs")
    else:
        v, q, s = TestFunction(W, name="vqs")

    t = W.system.base_time()
    j = W.system.base_vectors()[1]
    equations = (
        Dot(u.diff(t), v)
        + Dot(Dot(Grad(u), u), v)
        - nu * Dot(Div(Grad(u)), v)
        - T * Dot(j, v)
        + Dot(Grad(p), v),
        Div(u) * q,
        (T.diff(t) + Dot(u, Grad(T)) - kappa * Div(Grad(T))) * s,
    )

    solver = CoupledIMEXRungeKutta(
        equations,
        tableau=TABLEAU,
        initial=(0, None, initial_temperature(VT, M, Lx, mode)),
        time=(0.0, T_END),
        # Legendre would find these two itself, from their empty rows and
        # columns; the Chebyshev test functions leave the rows nonempty.
        pin={p: [(0, 0), (0, N - 1)]},
        sparse=True,
    )
    return solver, W, VT


def initial_temperature(VT, M: int, Lx: float, mode: int, seed: int = SEED):
    """Return the initial temperature coefficients, as RayleighBenard.py makes them.

    Modes 0 and 1 perturb the conducting profile, with broadband noise and with
    the single Fourier mode cos(2 pi x / Lx); mode 2 starts from T = 0, perturbed
    by noise. The coefficients are those of the homogeneous part, so zero is the
    conducting profile, which is exactly the boundary lifting.

    A single mode grows and saturates everywhere at once, into one coherent burst
    whose overshoot drives <v T>, and with it the instantaneous Nu, negative for
    a while. Noise releases plumes at different places and times.
    """
    Vh = VT.get_homogeneous()
    xx, yy = Vh.mesh()
    if mode == 0:
        noise = jax.random.normal(jax.random.PRNGKey(seed), Vh.shape)
        T_hat = Vh.forward(AMPLITUDE * noise * (1 - yy**2))
    elif mode == 1:
        noise = jnp.cos(2 * jnp.pi * xx / Lx) * jnp.ones_like(yy)
        T_hat = Vh.forward(AMPLITUDE * noise * (1 - yy**2))
    else:
        noise = jax.random.uniform(jax.random.PRNGKey(seed), VT.shape)
        T_hat = VT.forward(AMPLITUDE * noise * (1 - yy**2))
    return T_hat.at[M // 2].set(0.0)  # the Nyquist mode, as RayleighBenard.py


def nusselt(solver, state, W, VT, Ra: float, Pr: float, pad) -> float:
    """Return Nu = 1 + 2 sqrt(Ra Pr) <v T>, the total over the conductive flux."""
    vy = W.flatten()[1].backward(state[1], N=pad)
    T = VT.backward(state[3], N=pad)
    # Unweighted along y, also for Chebyshev, whose scalar product is weighted.
    _, q = VT.basespaces[1].integration_points_and_weights(pad[1])
    mean = jnp.mean(vy * T, axis=0) @ q / jnp.sum(q)
    return 1.0 + 2.0 * (Ra * Pr) ** 0.5 * float(mean)


def main() -> None:
    echo(f"M={M} N={N} Ra={RA:g} Pr={PR} dt={DT} T={T_END} {POLYNOMIAL}")
    pad = (3 * M // 2, N)
    _t = time.time()
    solver, W, VT = rayleigh_benard(M, N, LX, RA, PR, POLYNOMIAL)
    echo(f"  assembled in {time.time() - _t:.1f}s, dofs {W.num_dofs}")

    _t = time.time()
    snaps = solver.solve(
        DT,
        n_batches=N_SNAPSHOTS,
        N=pad,
        return_batch_snapshots=True,
        progress=is_leader(),
    )
    echo(f"  run {time.time() - _t:.1f}s")

    final = tuple(s[-1] for s in snaps)
    continuity = float(jnp.abs(solver.apply_linear(final)[2]).max())
    nus = [
        nusselt(solver, tuple(s[i] for s in snaps), W, VT, RA, PR, pad)
        for i in range(len(snaps[0]))
    ]
    every = max(1, len(nus) // 12)
    echo("  Nu(t) " + " ".join(f"{v:.2f}" for v in nus[::every]))
    echo(f"  weak continuity residual {continuity:.2e}")
    eps = float(jnp.finfo(final[0].dtype).eps)
    assert continuity < 1e4 * eps * max(1.0, float(jnp.abs(final[0]).max()))

    if "PYTEST" in os.environ:
        assert all(bool(jnp.isfinite(s).all()) for s in final)
        sys.exit(0)

    # Gathered to the host by every process, before only one goes on to plot.
    T_phys, x, y = to_host(
        (VT.backward(final[3], N=pad), *VT.mesh(N=pad, broadcast=False))
    )
    if not is_leader():
        return solver
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    c = ax.contourf(x, y, T_phys.T, levels=40, cmap="RdBu_r")
    ax.set_title(f"T(x, y, t={T_END:g}),  Ra={RA:g} Pr={PR}")
    ax.set_aspect("equal")
    fig.colorbar(c, ax=ax, shrink=0.9)
    plt.show()
    return solver


if __name__ == "__main__":
    solver = main()
