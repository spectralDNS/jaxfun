# Solve the 1D focusing nonlinear Schrodinger equation with ETDRK4 in time
#
#   i psi_t + psi_xx + |psi|^2 psi = 0,  x in [-L, L] periodic
#
# Spatial discretization: Fourier Galerkin (spectral)
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

from jaxfun import Domain
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier as space
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ETDRK4

N = 192
L = 24.0
T = 4.5
steps = 9000
n_states = 180

alpha1 = 1.0
shift1 = -6.5
velocity1 = 1.6
phase1 = 0.0

alpha2 = 0.8
shift2 = 6.0
velocity2 = -1.3
phase2 = sp.pi / 3

if "PYTEST" in os.environ:
    N = 48
    T = 0.4
    steps = 192
    n_states = 8
elif "NLS_FAST" in os.environ:
    N = 64
    T = 0.6
    steps = 240
    n_states = 24

dt = T / steps

V = FunctionSpace(N, space, domain=Domain(-L, L), name="V", fun_str="E")
v = TestFunction(V, name="v")
psi = TrialFunction(V, name="psi", transient=True)

(x,) = V.system.base_scalars()
t = V.system.base_time()


def bright_soliton(
    alpha: float, shift: float, velocity: float, phase: sp.Expr | float
) -> sp.Expr:
    envelope = sp.sqrt(2) * alpha * sp.sech(alpha * (x - shift))
    carrier = sp.exp(sp.I * (velocity * (x - shift) / 2 + phase))
    return envelope * carrier


psi0 = bright_soliton(alpha1, shift1, velocity1, phase1) + bright_soliton(
    alpha2, shift2, velocity2, phase2
)
weak_form = v * (psi.diff(t) - sp.I * psi.diff(x, 2) - sp.I * sp.Abs(psi) ** 2 * psi)

integrator = ETDRK4(
    V,
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
    progress=False,
)
times = jnp.linspace(0.0, T, states.shape[0])


@jax.jit
def backward_saved_states(coefficients):
    return jax.vmap(V.backward)(coefficients)


xj = V.mesh()
psi_states = backward_saved_states(states)
density_states = jnp.abs(psi_states) ** 2
real_states = psi_states.real
imag_states = psi_states.imag
mass_history = jnp.trapezoid(density_states, xj, axis=1)
mass_drift = float(jnp.max(jnp.abs(mass_history - mass_history[0])))
density_change = float(
    jnp.linalg.norm(density_states[-1] - density_states[0])
    / jnp.linalg.norm(density_states[0])
)

if "PYTEST" in os.environ:
    assert states.shape == (n_states + 1, V.num_dofs)
    assert jnp.isfinite(psi_states).all()
    assert density_change > 0.1, density_change
    assert mass_drift < 1e-3, mass_drift
    sys.exit(1)

gif_path = Path(
    os.environ.get("NLS_GIF_PATH", Path(__file__).with_name("nls1d_etdrk4.gif"))
)

print("Max mass drift =", mass_drift)
print("Density change =", density_change)
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

fig = plt.figure(figsize=(13, 6), constrained_layout=True)
grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.35), height_ratios=(1.0, 0.9))
ax_density = fig.add_subplot(grid[0, 0])
ax_wave = fig.add_subplot(grid[1, 0], sharex=ax_density)
ax_heat = fig.add_subplot(grid[:, 1])

density_color = "#2a9d8f"
fill_color = "#8ecae6"
real_color = "#355c7d"
imag_color = "#e76f51"
time_color = "#f4a261"

density_max = float(density_states.max())
wave_max = float(jnp.max(jnp.abs(psi_states)))

fill = [
    ax_density.fill_between(
        xj,
        density_states[0],
        color=fill_color,
        alpha=0.55,
        linewidth=0,
    )
]
(density_line,) = ax_density.plot(xj, density_states[0], color=density_color, lw=2.5)
ax_density.plot(
    xj,
    density_states[0],
    color="#6d6875",
    lw=1.0,
    ls="--",
    alpha=0.8,
    label="initial density",
)
ax_density.set_xlim(float(xj[0]), float(xj[-1]))
ax_density.set_ylim(0.0, density_max * 1.1)
ax_density.set_ylabel(r"$|\psi|^2$")
ax_density.set_title("Interfering Bright Solitons")
ax_density.grid(alpha=0.35)

(real_line,) = ax_wave.plot(
    xj, real_states[0], color=real_color, lw=2.0, label="Re(psi)"
)
(imag_line,) = ax_wave.plot(
    xj, imag_states[0], color=imag_color, lw=2.0, label="Im(psi)"
)
ax_wave.axhline(0.0, color="#6d6875", lw=1.0, alpha=0.5)
ax_wave.set_xlim(float(xj[0]), float(xj[-1]))
ax_wave.set_ylim(-wave_max * 1.15, wave_max * 1.15)
ax_wave.set_xlabel("x")
ax_wave.set_ylabel("wave field")
ax_wave.grid(alpha=0.35)
ax_wave.legend(loc="upper right")

heat = ax_heat.imshow(
    density_states.T,
    origin="lower",
    aspect="auto",
    extent=(0.0, T, float(xj[0]), float(xj[-1])),
    cmap="magma",
    interpolation="bicubic",
    vmin=0.0,
    vmax=density_max,
)
time_line = ax_heat.axvline(float(times[0]), color=time_color, lw=2.5, alpha=0.95)
ax_heat.set_xlabel("time")
ax_heat.set_ylabel("x")
ax_heat.set_title("Density Evolution")
fig.colorbar(heat, ax=ax_heat, label=r"$|\psi(x, t)|^2$")

stats = ax_density.text(
    0.02,
    0.95,
    "",
    transform=ax_density.transAxes,
    va="top",
    ha="left",
    bbox={
        "boxstyle": "round,pad=0.35",
        "fc": "#fffaf2",
        "ec": "#c9b79c",
        "alpha": 0.95,
    },
)


def update(frame: int):
    fill[0].remove()
    fill[0] = ax_density.fill_between(
        xj,
        density_states[frame],
        color=fill_color,
        alpha=0.55,
        linewidth=0,
    )
    density_line.set_ydata(density_states[frame])
    real_line.set_ydata(real_states[frame])
    imag_line.set_ydata(imag_states[frame])
    time_line.set_xdata([float(times[frame]), float(times[frame])])
    stats.set_text(
        f"t = {float(times[frame]):.2f}\n"
        f"peak |psi|^2 = {float(density_states[frame].max()):.2f}\n"
        f"mass drift = {float(jnp.abs(mass_history[frame] - mass_history[0])):.2e}"
    )
    return fill[0], density_line, real_line, imag_line, time_line, stats


anim = FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
anim.save(gif_path, writer="pillow", fps=24)

fig.suptitle("1D Focusing NLS with Two Counter-Propagating Solitons", fontsize=15)
plt.show()
