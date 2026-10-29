# Solve the 2D focusing nonlinear Schrodinger equation with ETDRK4 in time
#
#   i psi_t + Δpsi + |psi|^2 psi = 0,  (x, y) in [-L, L]^2 periodic
#
# starting from the Gaussian psi(x, y, 0) = A exp(-(x^2 + y^2)). In 2D the cubic
# NLS is mass critical: a Gaussian whose mass pi A^2 / 2 is above the Townes
# threshold (~11.70) self-focuses and collapses in finite time. A = 3 gives
# mass 9 pi / 2 ~ 14.1 and collapse near t = 0.4, so the run stops at t = 0.38,
# while the narrowing peak is still resolved on the grid.
#
# Spatial discretization: 2D Fourier Galerkin (spectral)
# Time discretization: ETDRK4 (explicit)
# ruff: noqa: E402
import os
import sys
from pathlib import Path

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ETDRK4

N = 256
L = 8.0
A = 3.0
T = 0.38
steps = 1520
n_states = 152

if "PYTEST" in os.environ:
    N = 32
    T = 0.1
    steps = 40
    n_states = 4
elif "NLS_FAST" in os.environ:
    N = 128
    T = 0.36
    steps = 360
    n_states = 36

dt = T / steps

F = Fourier(N, Domain(-L, L))
V = TensorProduct(F, F, name="V")
v = TestFunction(V, name="v")
psi = TrialFunction(V, name="psi", transient=True)

x, y = V.system.base_scalars()
t = V.system.base_time()

psi0 = A * sp.exp(-(x**2 + y**2))
weak_form = v * (psi.diff(t) - sp.I * Div(Grad(psi)) - sp.I * sp.Abs(psi) ** 2 * psi)

integrator = ETDRK4(
    weak_form,
    time=(0.0, T),
    initial=psi0,
    sparse=True,
    sparse_tol=1000,
)
states = integrator.solve(
    dt=dt,
    steps=steps,
    n_batches=n_states,
    return_batch_snapshots=True,
    progress="PYTEST" not in os.environ,
)
times = jnp.linspace(0.0, T, states.shape[0])

xj, yj = V.mesh(broadcast=False)
psi_states = V.backward_batch(states)
density_states = jnp.abs(psi_states) ** 2
real_states = psi_states.real
cell_area = float((xj[1] - xj[0]) * (yj[1] - yj[0]))
mass_history = jnp.sum(density_states, axis=(1, 2)) * cell_area
mass_drift = float(jnp.max(jnp.abs(mass_history / mass_history[0] - 1)))
peak_history = jnp.max(density_states, axis=(1, 2))
peak_growth = float(peak_history[-1] / peak_history[0])

if "PYTEST" in os.environ:
    assert states.shape == (n_states + 1,) + V.num_dofs
    assert bool(jnp.isfinite(psi_states).all())
    assert peak_growth > 1.2, peak_growth
    assert mass_drift < 1e-4, mass_drift
    sys.exit(0)

gif_path = Path(
    os.environ.get("NLS_GIF_PATH", Path(__file__).with_name("nls2d_etdrk4.gif"))
)

print("Initial mass =", float(mass_history[0]))
print("Max relative mass drift =", mass_drift)
print("Peak density growth =", peak_growth)
print(f"Saving animation to {gif_path}")

plt.rcParams.update(
    {
        "axes.facecolor": "#fbf6ef",
        "figure.facecolor": "#f3ede4",
        "axes.edgecolor": "#3b2f2f",
        "axes.labelcolor": "#231815",
        "xtick.color": "#231815",
        "ytick.color": "#231815",
        "grid.color": "#d9c9b6",
        "text.color": "#231815",
    }
)

fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.05))
ax_density = fig.add_subplot(grid[:, 0])
ax_real = fig.add_subplot(grid[:, 1])
ax_profile = fig.add_subplot(grid[0, 2])
ax_peak = fig.add_subplot(grid[1, 2])

density_color = "#2a9d8f"
fill_color = "#8ecae6"
peak_color = "#355c7d"
time_color = "#e76f51"

# The collapse happens near the origin, so the images zoom in on it
window = 3.0
extent = (float(xj[0]), float(xj[-1]), float(yj[0]), float(yj[-1]))
mid = N // 2  # index of x = 0 and y = 0

density_image = ax_density.imshow(
    density_states[0].T,
    origin="lower",
    extent=extent,
    cmap="magma",
    interpolation="bicubic",
)
fig.colorbar(density_image, ax=ax_density, shrink=0.8, label=r"$|\psi|^2$")
ax_density.set_title("Density")

real_image = ax_real.imshow(
    real_states[0].T,
    origin="lower",
    extent=extent,
    cmap="RdBu_r",
    interpolation="bicubic",
)
fig.colorbar(real_image, ax=ax_real, shrink=0.8, label=r"Re$(\psi)$")
ax_real.set_title("Real part")

for ax in (ax_density, ax_real):
    ax.set_xlim(-window, window)
    ax.set_ylim(-window, window)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

density_max = float(peak_history.max())
fill = [
    ax_profile.fill_between(
        xj, density_states[0, :, mid], color=fill_color, alpha=0.55, linewidth=0
    )
]
(profile_line,) = ax_profile.plot(
    xj, density_states[0, :, mid], color=density_color, lw=2.5
)
ax_profile.plot(
    xj,
    density_states[0, :, mid],
    color="#6d6875",
    lw=1.0,
    ls="--",
    alpha=0.8,
    label="initial",
)
ax_profile.set_xlim(-window, window)
ax_profile.set_ylim(0.0, density_max * 1.1)
ax_profile.set_xlabel("x")
ax_profile.set_ylabel(r"$|\psi(x, 0)|^2$")
ax_profile.set_title("Centreline profile")
ax_profile.grid(alpha=0.35)
ax_profile.legend(loc="upper right")

ax_peak.semilogy(times, peak_history, color=peak_color, lw=2.0)
(peak_marker,) = ax_peak.plot(
    [float(times[0])], [float(peak_history[0])], "o", color=time_color, ms=8
)
ax_peak.set_xlim(0.0, T)
ax_peak.set_xlabel("time")
ax_peak.set_ylabel(r"$\max\, |\psi|^2$")
ax_peak.set_title("Peak density")
ax_peak.grid(alpha=0.35, which="both")

stats = ax_peak.text(
    0.03,
    0.94,
    "",
    transform=ax_peak.transAxes,
    va="top",
    ha="left",
    bbox={
        "boxstyle": "round,pad=0.35",
        "fc": "#fffaf2",
        "ec": "#c9b79c",
        "alpha": 0.95,
    },
)

fig.suptitle(
    f"2D Focusing NLS: Collapse of a Gaussian (mass {float(mass_history[0]):.2f}"
    r" > Townes mass $\approx$ 11.70)",
    fontsize=15,
)


def update(frame: int):
    # Rescale the colours every frame, since the peak grows by an order of
    # magnitude and a fixed scale would hide the early frames
    peak = float(peak_history[frame])
    amplitude = float(jnp.abs(real_states[frame]).max())
    density_image.set_data(density_states[frame].T)
    density_image.set_clim(0.0, peak)
    real_image.set_data(real_states[frame].T)
    real_image.set_clim(-amplitude, amplitude)
    fill[0].remove()
    fill[0] = ax_profile.fill_between(
        xj, density_states[frame, :, mid], color=fill_color, alpha=0.55, linewidth=0
    )
    profile_line.set_ydata(density_states[frame, :, mid])
    peak_marker.set_data([float(times[frame])], [peak])
    stats.set_text(
        f"t = {float(times[frame]):.3f}\n"
        f"peak |psi|^2 = {peak:.1f}\n"
        f"mass drift = {float(jnp.abs(mass_history[frame] / mass_history[0] - 1)):.1e}"
    )
    return density_image, real_image, fill[0], profile_line, peak_marker, stats


anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
anim.save(gif_path, writer="pillow", fps=24)

plt.show()
