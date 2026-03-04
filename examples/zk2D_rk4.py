# Solve the 2D Zakharov-Kuznetsov equation with ETDRK4 in time
#
#   u_t + (u^2 / 2)_x + mu^2 (u_xx + u_yy)_x = 0,
#   (x, y) in [-1, 1] x [-1, 1] periodic
#
# Spatial discretization: 2D Fourier Galerkin (spectral)
# Time discretization: ETDRK4 (explicit)
# ruff: noqa: E402
import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation

from jaxfun import Div, Domain, Grad
from jaxfun.galerkin import TensorProductSpace
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ETDRK4
from jaxfun.operators import Constant

N = 64
mu = Constant("mu", sp.Rational(3, 20))
T = 0.3
steps = 12000
n_states = 120

if "PYTEST" in os.environ:
    N = 24
    T = 0.01
    steps = 24
    n_states = 6

dt = T / steps

F = Fourier(N, Domain(-1, 1))
V = TensorProductSpace((F, F), name="V")
v = TestFunction(V, name="v")
u = TrialFunction(V, name="u", transient=True)

x, y = V.system.base_scalars()
t = V.system.base_time()

u0 = sp.cos(sp.pi * x)  # * (sp.cos(sp.pi * y) + 1) / 2
laplace_u = Div(Grad(u))
weak_form = v * (u.diff(t) + (u * u).diff(x) / 2 + mu**2 * laplace_u.diff(x))

integrator = ETDRK4(
    V,
    weak_form,
    time=(0.0, T),
    initial=u0,
    sparse=True,
    sparse_tol=1000,
)

states = integrator.solve(
    dt=dt,
    steps=steps,
    n_batches=n_states,
    return_each_step=True,
)
times = jnp.linspace(0.0, T, states.shape[0])

if "PYTEST" in os.environ:
    u0_phys = V.backward(states[0]).real
    uT_phys = V.backward(states[-1]).real
    assert jnp.isfinite(uT_phys).all()
    assert float(jnp.linalg.norm(uT_phys - u0_phys)) > 1e-8
    sys.exit(1)

states_phys = jnp.array([V.backward(uhat).real for uhat in states])
x_plot, y_plot = V.mesh(broadcast=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
vmin = float(states_phys.min())
vmax = float(states_phys.max())
levels = 40

# V.backward returns arrays with axis order (x, y), while Matplotlib's
# contourf/imshow with 1D x/y inputs expects (y, x).
_c0 = axes[0].contourf(
    x_plot,
    y_plot,
    states_phys[0].T,
    levels=levels,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
)
c1 = axes[1].contourf(
    x_plot,
    y_plot,
    states_phys[-1].T,
    levels=levels,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
)
c2 = axes[2].contourf(
    x_plot,
    y_plot,
    (states_phys[-1] - states_phys[0]).T,
    levels=levels,
    cmap="coolwarm",
)

axes[0].set_title("u(x, y, 0)")
axes[1].set_title(f"u(x, y, T), T={times[-1]:.3f}")
axes[2].set_title("u(T) - u(0)")
for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

fig.colorbar(c1, ax=axes[:2], shrink=0.9)
fig.colorbar(c2, ax=axes[2], shrink=0.9)
fig.suptitle("2D Zakharov-Kuznetsov equation (ETDRK4)")

fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
im = ax_anim.imshow(
    states_phys[0].T,
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
title = ax_anim.set_title(f"2D ZK (t={times[0]:.3f})")


def update(frame: int):
    im.set_data(states_phys[frame].T)
    title.set_text(f"2D ZK (t={times[frame]:.3f})")
    return (im,)


_anim = FuncAnimation(fig_anim, update, frames=len(times), interval=40, blit=False)
_anim.save("zk2d_etdrk4.gif", writer="pillow", fps=24)
plt.show()
