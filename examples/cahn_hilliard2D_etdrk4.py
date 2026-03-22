# Solve the 2D Cahn-Hilliard equation with ETDRK4 in time
#
#   u_t - Δ(nu * u + alpha * u^3 + mu * Δu) = 0,
#   (x, y) in [0, 1] x [0, 1] periodic
#
# Spatial discretization: 2D Fourier Galerkin (spectral)
# Time discretization: ETDRK4 (explicit exponential integrator)
# ruff: noqa: E402
import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProductSpace
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ETDRK4
from jaxfun.operators import Constant


def sample_initial_condition_ch_2d(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return a smooth, zero-mean random initial condition on a periodic grid."""
    Mx, My = x.size, y.size
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    kx = np.fft.fftfreq(Mx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(My, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2

    noise = rng.standard_normal((Mx, My))
    k0 = rng.uniform(5.0, 20.0)
    spectral_filter = np.exp(-K2 / (2 * k0**2))

    u0 = np.fft.ifft2(np.fft.fft2(noise) * spectral_filter).real
    return np.clip(u0 - u0.mean(), -1.0, 1.0)


N = 96
T = 5e-2  # * 20
steps = 320  # * 20
n_states = 80  # * 20
seed = 42

nu = Constant("nu", -1.0)
alpha = Constant("alpha", 1.0)
# alpha = Constant("alpha", 0.5)
mu = Constant("mu", -1.5e-2)
# mu = Constant("mu", -0.75e-2)

if "PYTEST" in os.environ:
    N = 24
    T = 2e-3
    steps = 8
    n_states = 4
elif "CH_FAST" in os.environ:
    N = 48
    T = 1e-2
    steps = 32
    n_states = 8

dt = T / steps

F = Fourier(N, Domain(0.0, 1.0))
V = TensorProductSpace((F, F), name="V")
v = TestFunction(V, name="v")
u = TrialFunction(V, name="u", transient=True)

x, y = V.system.base_scalars()
t = V.system.base_time()

Laplace = lambda u: Div(Grad(u))

equation = u.diff(t) - Laplace(nu * u + alpha * u**3 + mu * Laplace(u))
weak_form = v * equation

x_plot, y_plot = V.mesh(broadcast=False)
rng = np.random.default_rng(seed)
u0_phys = jnp.asarray(
    sample_initial_condition_ch_2d(np.asarray(x_plot), np.asarray(y_plot), rng)
)
u0_hat = V.forward(u0_phys)

integrator = ETDRK4(
    V,
    weak_form,
    time=(0.0, T),
    initial=u0_hat,
    sparse=True,
    sparse_tol=1000,
)
states = integrator.solve(
    dt=dt,
    steps=steps,
    n_batches=n_states,
    return_batch_snapshots=True,
    progress=True,
)
times = jnp.linspace(0.0, T, states.shape[0])


@jax.jit
def backward_saved_states(coefficients):
    return jax.vmap(lambda u_hat: V.backward(u_hat).real)(coefficients)


u_states = backward_saved_states(states)
mean_history = jnp.mean(u_states, axis=(1, 2))
mean_drift = float(jnp.max(jnp.abs(mean_history - mean_history[0])))
relative_change = float(
    jnp.linalg.norm(u_states[-1] - u_states[0]) / jnp.linalg.norm(u_states[0])
)

if "PYTEST" in os.environ:
    assert states.shape == (n_states + 1,) + V.num_dofs
    assert bool(jnp.isfinite(u_states).all())
    assert mean_drift < 1e-5, mean_drift
    assert relative_change > 1e-2, relative_change
    sys.exit(0)

print("Mean drift =", mean_drift)
print("Relative change =", relative_change)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
vmin = float(u_states.min())
vmax = float(u_states.max())
# vmin = None
# vmax = None

im0 = axes[0].imshow(
    u_states[0].T,
    origin="lower",
    extent=(float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])),
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    aspect="equal",
)
im1 = axes[1].imshow(
    u_states[-1].T,
    origin="lower",
    extent=(float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])),
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    aspect="equal",
)
im2 = axes[2].imshow(
    (u_states[-1] - u_states[0]).T,
    origin="lower",
    extent=(float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])),
    cmap="coolwarm",
    aspect="equal",
)

axes[0].set_title("u(x, y, 0)")
axes[1].set_title(f"u(x, y, T), T={times[-1]:.3e}")
axes[2].set_title("u(T) - u(0)")
for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.colorbar(im1, ax=axes[:2], shrink=0.9)
fig.colorbar(im2, ax=axes[2], shrink=0.9)
fig.suptitle("2D Cahn-Hilliard Equation (ETDRK4)")

fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
im = ax_anim.imshow(
    u_states[0].T,
    origin="lower",
    extent=(float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])),
    cmap="RdBu_r",
    # cmap="jet",
    vmin=vmin,
    vmax=vmax,
    aspect="equal",
    interpolation="nearest",
)
fig_anim.colorbar(im, ax=ax_anim, shrink=0.9)
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")
title = ax_anim.set_title(f"2D Cahn-Hilliard (t={times[0]:.3e})")


def update(frame: int):
    im.set_data(u_states[frame].T)
    title.set_text(
        f"2D Cahn-Hilliard (t={times[frame]:.3e}, mean drift={mean_drift:.2e})"
    )
    return (im,)


_anim = FuncAnimation(fig_anim, update, frames=len(times), interval=50, blit=False)
plt.show()
