# Solve the Schnakenberg reaction-diffusion system with IMEX Runge-Kutta in time
#
#   u_t - eu*Δu - gamma*(a - u + u^2 v) = 0,
#   v_t - ev*Δv - gamma*(b - u^2 v)     = 0,
#   (x, y) in [0, G] x [0, G] periodic
#
# The two equations couple only through the nonlinear reaction term u^2*v, so
# each is advanced by its own IMEXRungeKutta: the stiff diffusion operators stay
# decoupled and are factorized once per equation, while the reaction term is
# evaluated explicitly from both fields after every stage.
#
# The uniform state u = a + b, v = b/(a+b)^2 is an exact equilibrium; the initial
# condition below perturbs it locally, and diffusion-driven (Turing) instability
# grows the perturbation into a pattern.
#
# Spatial discretization: 2D Fourier Galerkin (spectral)
# Time discretization: SystemIMEXRungeKutta (ARS443)
# ruff: noqa: E402
import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from jax import Array
from matplotlib.animation import FuncAnimation

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProduct
from jaxfun.galerkin.arguments import JAXFunction, TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ARS443, SystemIMEXRungeKutta

N = 128
T = 500.0
steps = 1000
n_states = 100
G = 30
a = 0.1
b = 0.9
gamma = 3
eu = 1.0
ev = 10.0

if "PYTEST" in os.environ:
    N = 24
    T = 2e-2
    steps = 8
    n_states = 4

F = Fourier(N, Domain(0, G))
V = TensorProduct(F, F, name="V")
w = TestFunction(V, name="w")
u = TrialFunction(V, name="u", transient=True)
q = TestFunction(V, name="q")
v = TrialFunction(V, name="v", transient=True)

x, y = V.system.base_scalars()
t = V.system.base_time()

eq1 = (u.diff(t) - eu * Div(Grad(u)) - gamma * (a - u + u**2 * v)) * w
eq2 = (v.diff(t) - ev * Div(Grad(v)) - gamma * (b - u**2 * v)) * q

ue = 1 - sp.exp(-2 * ((x - G / 2.15) ** 2 + (y - G / 2.15) ** 2))
ve = b / (a + b) ** 2 + sp.exp(-2 * ((x - G / 2) ** 2 + 2 * (y - G / 2) ** 2))
u0_hat = JAXFunction(ue, V)
v0_hat = JAXFunction(ve, V)

integrator = SystemIMEXRungeKutta(
    (eq1, eq2),
    tableau=ARS443,
    time=(0.0, T),
    initial=(cast(Array, u0_hat.array), cast(Array, v0_hat.array)),
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


x_plot, y_plot = V.mesh(broadcast=False)
u_states = V.backward_batch(u_hats).real
v_states = V.backward_batch(v_hats).real

if "PYTEST" in os.environ:
    assert u_hats.shape == (n_states + 1,) + V.num_dofs
    assert v_hats.shape == (n_states + 1,) + V.num_dofs
    assert bool(jnp.isfinite(u_states).all())
    assert bool(jnp.isfinite(v_states).all())
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
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[row, col].set_title(f"{name}(x, y, t={times[frame]:.3g})")
        axes[row, col].set_xlabel("x")
        axes[row, col].set_ylabel("y")
    fig.colorbar(im, ax=axes[row, :], shrink=0.9)
fig.suptitle("Schnakenberg System (IMEX RK)")

fig_anim, axes_anim = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
images = []
fields = (("u", u_states), ("v", v_states))
for ax, (name, field) in zip(axes_anim, fields, strict=True):
    vmin, vmax = limits[name]
    image = ax.imshow(
        field[0].T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
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
title = fig_anim.suptitle(f"Schnakenberg (t={times[0]:.3e})")


def update(frame: int):
    images[0].set_data(u_states[frame].T)
    images[1].set_data(v_states[frame].T)
    title.set_text(f"Schnakenberg (t={times[frame]:.3e})")
    return tuple(images)


_anim = FuncAnimation(fig_anim, update, frames=len(times), interval=50, blit=False)
_anim.save("schnakenberg.gif", writer="pillow", fps=24)
plt.show()
