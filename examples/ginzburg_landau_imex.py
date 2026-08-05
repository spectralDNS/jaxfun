# Solve the Ginzburg-Landau equation with IMEX Runge-Kutta in time
#
#   u_t - Δu - u + (1 + 1.5j) u * |u|^2 = 0,
#   (x, y) in [-50, 50] x [-50, 50] periodic
#
# Spatial discretization: 2D Fourier Galerkin (spectral)
# Time discretization: IMEXRungeKutta (ARS443)
# ruff: noqa: E402
import os
import sys
from typing import cast

import jax
import jax.numpy as jnp

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProduct
from jaxfun.galerkin.arguments import JAXFunction, TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ARS443, IMEXRungeKutta

N = 128
T = 100.0
steps = 5000
n_states = 200

if "PYTEST" in os.environ:
    N = 24
    T = 2e-3
    steps = 8
    n_states = 4

dt = T / steps

F = Fourier(N, Domain(-50, 50))
V = TensorProduct(F, F, name="V")
v = TestFunction(V, name="v")
u = TrialFunction(V, name="u", transient=True)

x, y = V.system.base_scalars()
t = V.system.base_time()

equation = u.diff(t) - Div(Grad(u)) - u + (1 + 1.5j) * u * sp.Abs(u) ** 2
weak_form = v * equation

ue = (x + y) * sp.exp(-0.03 * (x**2 + y**2))
u0_hat = JAXFunction(ue, V, name="u0")

integrator = IMEXRungeKutta(
    V,
    weak_form,
    tableau=ARS443,
    time=(0.0, T),
    initial=cast(jax.Array, u0_hat.array),
    sparse=True,
    sparse_tol=1000,
)
states = integrator.solve(
    dt=dt,
    steps=steps,
    n_batches=n_states,
    return_batch_snapshots=True,
    # N=(196, 196),
    progress=True,
)
times = jnp.linspace(0.0, T, states.shape[0])


@jax.jit
def backward_saved_states(coefficients):
    return jax.vmap(lambda u_hat: V.backward(u_hat).real)(coefficients)


x_plot, y_plot = V.mesh(broadcast=False)
u_states = backward_saved_states(states)

if "PYTEST" in os.environ:
    assert states.shape == (n_states + 1,) + V.num_dofs
    assert bool(jnp.isfinite(u_states).all())
    sys.exit(0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
vmin = float(u_states.min())
vmax = float(u_states.max())

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

axes[0].set_title("u(x, y, 0)")
axes[1].set_title(f"u(x, y, T), T={times[-1]:.3e}")
for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig.colorbar(im1, ax=axes[:2], shrink=0.9)
fig.suptitle("Ginzburg-Landau Equation (IMEX RK)")

fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
im = ax_anim.imshow(
    u_states[0].T,
    origin="lower",
    extent=(float(x_plot[0]), float(x_plot[-1]), float(y_plot[0]), float(y_plot[-1])),
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    aspect="equal",
    interpolation="nearest",
)
fig_anim.colorbar(im, ax=ax_anim, shrink=0.9)
ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")
title = ax_anim.set_title(f"Ginzburg-Landau (t={times[0]:.3e})")


def update(frame: int):
    im.set_data(u_states[frame].T)
    title.set_text(f"Ginzburg-Landau (t={times[frame]:.3e})")
    return (im,)


_anim = FuncAnimation(fig_anim, update, frames=len(times), interval=50, blit=False)
plt.show()
