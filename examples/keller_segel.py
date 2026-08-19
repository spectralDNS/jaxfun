# Solve the volume-filling Keller-Segel chemotaxis system with IMEX Runge-Kutta
#
#   u_t = D*Div(Grad(u)) - chi*Div(u*(1 - u)*Grad(v)),
#   0   = Div(Grad(v)) - v + u,
#   (x, y) in [0, G] x [0, G] periodic
#
# Cells `u` climb the gradient of a chemoattractant `v` that they produce
# themselves. The attractant diffuses so much faster than the cells move that it
# is taken to be in quasi-steady state, which is the standard parabolic-elliptic
# limit of the model -- and leaves `v` with no time derivative at all.
#
# So this is a *constraint* equation: `v` is not integrated but solved for from
# `u` at every Runge-Kutta stage, in between the implicit stage solves and the
# evaluation of the nonlinear terms. It is therefore never lagged -- at every
# stage `v` is the exact solution of its own equation for the `u` of that same
# stage. Compare sandbox/schnakenberg.py, which is the same machinery with two
# transported species and no constraint.
#
# Two details make this a well-behaved constraint. The `-v` term removes the
# Laplacian's null space, so the operator is invertible (and, being Fourier,
# diagonal) with no pressure-like pinning needed. And `v` appears linearly and
# is never multiplied by `u`, so the constraint assembles as one square operator
# in `v` alone, with `u` entering only through the explicit part.
#
# Without the volume-filling factor (1 - u) the classical Keller-Segel model
# blows up in finite time above a critical mass, which it does here: at chi=5
# and mass 404 the run reaches inf within t=2. The factor caps the chemotactic
# flux as u approaches 1 and the aggregates saturate instead.
#
# Both terms of the transported equation are divergences on a periodic domain,
# so the total number of cells is conserved. The run reports the drift, which
# turns out to be a good resolution diagnostic rather than a pure round-off
# check. At chi=10, T=60 the drift falls from 1.1e-07 at N=64 to 5.0e-11 at
# N=96 and 7.2e-11 at N=128, i.e. it reaches the round-off floor once N=96
# resolves the aggregates. Pushing to chi=20 sharpens them past what N=64 can
# carry: u then overshoots [0, 1] by 0.5% and the drift rises to 1.2e-04. That
# is genuinely spatial -- cutting dt fourfold changes it in no digit, and
# padding the dealiasing from 3N/2 to 3N changes only the fourth.
#
# REFERENCES
#
#   Keller & Segel (1970), J. Theor. Biol. 26, 399-415 -- the original model.
#   Hillen & Painter (2001), Adv. Appl. Math. 26, 280-301 -- the density-
#     limiting factor, and global existence once it is there.
#   Painter & Hillen (2002), Can. Appl. Math. Q. 10, 501-543 -- "volume
#     filling", derived from a space-limited random walk.
#   Hillen & Painter (2009), J. Math. Biol. 58, 183-217 -- survey; source of
#     the instability condition chi*ubar*(1 - ubar)/D > 1 + k^2 that CHI and
#     UBAR below are chosen against.
#   Painter & Hillen (2011), Physica D 240, 363-375 -- the same model in 2D on
#     a periodic square, closest to what this file does.
#   Jaeger & Luckhaus (1992), Trans. AMS 329, 819-824 -- the parabolic-elliptic
#     system and its finite-time blow-up without volume filling.
#
# The equations are from the literature, but G, D, CHI and UBAR are tuned here
# for a resolved picture at N=96 and are not taken from any of these papers. The
# sharp critical-mass results are for the unscreened whole-plane problem
# (0 = lap(v) + u), so they explain why the classical model blows up here
# without fixing a threshold for this version.
#
# Spatial discretization: 2D Fourier Galerkin (spectral)
# Time discretization: SystemIMEXRungeKutta (ARS443)
# ruff: noqa: E402
import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ARS443, SystemIMEXRungeKutta

N = 96
G = 20.0
D = 1.0
CHI = 10.0
UBAR = 0.5  # mean cell density; the volume-filling model saturates at u = 1
T = 60.0
steps = 30000
n_states = 100
# N and CHI are a pair: raising CHI sharpens the aggregates and needs a larger N
# to keep them resolved. See the mass-drift note at the top.

if "PYTEST" in os.environ:
    N = 24
    T = 2e-2
    steps = 8
    n_states = 4

F = Fourier(N, Domain(0, G))
V = TensorProduct(F, F, name="V")
w = TestFunction(V, name="w")
q = TestFunction(V, name="q")
u = TrialFunction(V, name="u", transient=True)
v = TrialFunction(V, name="v")  # constrained: no transient=True

x, y = V.system.base_scalars()
t = V.system.base_time()

chemotaxis = Div(u * (1 - u) * Grad(v))
eq1 = (u.diff(t) - D * Div(Grad(u)) + CHI * chemotaxis) * w
eq2 = (Div(Grad(v)) - v + u) * q

# A random perturbation of the uniform state. The uniform state is an exact
# equilibrium, so something has to break it; noise rather than a single mode, so
# the aggregates have to find their own arrangement instead of inheriting one.
key = jax.random.PRNGKey(0)
u0 = V.forward(UBAR + 0.01 * jax.random.normal(key, V.shape))

integrator = SystemIMEXRungeKutta(
    (eq1, eq2),
    tableau=ARS443,
    time=(0.0, T),
    initial=(u0, None),  # None: v is solved for, not initialized
    sparse=True,
)

dealias = (3 * N // 2, 3 * N // 2)

u_hats, v_hats = integrator.solve(
    dt=T / steps,
    n_batches=n_states,
    return_batch_snapshots=True,
    N=dealias,
    progress=True,
)

times = jnp.linspace(0.0, T, u_hats.shape[0])


@jax.jit
def backward_saved_states(coefficients):
    return jax.vmap(lambda u_hat: V.backward(u_hat).real)(coefficients)


x_plot, y_plot = V.mesh(broadcast=False)
u_states = backward_saved_states(u_hats)
v_states = backward_saved_states(v_hats)

# Cell count, conserved by the transported equation. See the note at the top:
# the drift is a resolution diagnostic, and sits at the round-off floor once the
# aggregates are resolved.
cell = (G / N) ** 2
mass = jnp.sum(u_states, axis=(1, 2)) * cell
drift = float(jnp.abs(mass[-1] - mass[0]) / mass[0])
# Loose enough to pass at the resolved round-off floor (5e-11 in double
# precision, and single precision -- which the test suite runs in -- loses about
# 1e-07 to summing the quadrature alone), tight enough that a constraint solve
# injecting or removing cells would fail it outright.
mass_tol = max(1e-8, 1e3 * float(jnp.finfo(u_states.dtype).eps))
print(f"mass {float(mass[0]):.8f} -> {float(mass[-1]):.8f}")
print(f"relative drift {drift:.3e}  (tolerance {mass_tol:.1e})")
print(f"u in [{float(u_states[-1].min()):.5f}, {float(u_states[-1].max()):.5f}]")

if "PYTEST" in os.environ:
    assert u_hats.shape == (n_states + 1,) + V.num_dofs
    assert v_hats.shape == (n_states + 1,) + V.num_dofs
    assert bool(jnp.isfinite(u_states).all())
    assert bool(jnp.isfinite(v_states).all())
    assert drift < mass_tol
    sys.exit(0)

extent = (
    float(x_plot[0]),
    float(x_plot[-1]),
    float(y_plot[0]),
    float(y_plot[-1]),
)
limits = {
    "u": (float(u_states.min()), float(u_states.max())),
    "v": (float(v_states.min()), float(v_states.max())),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
for row, (name, field) in enumerate((("u", u_states), ("v", v_states))):
    vmin, vmax = limits[name]
    for col, frame in enumerate((0, -1)):
        im = axes[row, col].imshow(
            field[frame].T,
            origin="lower",
            extent=extent,
            cmap="magma" if name == "u" else "viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row, col].set_title(f"{name}(x, y, t={times[frame]:.3g})")
        axes[row, col].set_xlabel("x")
        axes[row, col].set_ylabel("y")
    fig.colorbar(im, ax=axes[row, :], shrink=0.9)
fig.suptitle("Volume-filling Keller-Segel (IMEX RK, elliptic constraint)")

fig_anim, axes_anim = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
images = []
fields = (("u", u_states, "magma"), ("v", v_states, "viridis"))
for ax, (name, field, cmap) in zip(axes_anim, fields, strict=True):
    vmin, vmax = limits[name]
    image = ax.imshow(
        field[0].T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    fig_anim.colorbar(image, ax=ax, shrink=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(name)
    images.append(image)
title = fig_anim.suptitle(f"Keller-Segel (t={times[0]:.3g})")


def update(frame: int):
    images[0].set_data(u_states[frame].T)
    images[1].set_data(v_states[frame].T)
    title.set_text(f"Keller-Segel (t={times[frame]:.3g})")
    return tuple(images)


_anim = FuncAnimation(fig_anim, update, frames=len(times), interval=50, blit=False)
_anim.save("keller_segel.gif", writer="pillow", fps=24)
plt.show()
