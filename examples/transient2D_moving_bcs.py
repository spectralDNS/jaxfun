# Solve a coupled transient/constraint system whose boundary value moves in time
#
#   u_t = -u - v,
#   0   = Div(Grad(v)) - v + u,
#   (x, y) in [-1, 1] x [-1, 1],  with v = (1 - x^2) * exp(-t) on the y = 1 wall
#   and v = 0 on the other three.
#
# `v` carries no time derivative, so it is not integrated but solved for from `u`
# at every Runge-Kutta stage -- the parabolic-elliptic pattern of
# `keller_segel.py`, except that here the *constrained* field is the one holding
# the inhomogeneous wall, and that wall decays as the run proceeds.
#
# Two things follow from that. Being algebraic, the constraint contributes no
# `-<q, dB/dt>` term the way an integrated field does -- but its lifting still
# has to be rebuilt at each stage time instead of reused from construction. And
# the coefficients the solver returns are the homogeneous part alone, so turning
# a whole run of snapshots back into fields takes one time per snapshot, which is
# what `backward_batch(v_hats, t=times)` is for below. Reconstructing them
# against a single frozen lifting would pin the wall at t=0 in every frame.
#
# Spatial discretization: Legendre Galerkin with a time-dependent Dirichlet BC
# Time discretization: IMEX Runge-Kutta with a constraint solve per stage

import os
from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib.animation import FuncAnimation

from jaxfun.coordinates import R
from jaxfun.galerkin import (
    DirectSumTPS,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.integrators import ARS443, SystemIMEXRungeKutta
from jaxfun.operators import Div, Grad

N = 64
T = 2.0
n_states = 50
if "PYTEST" in os.environ:
    N = 20
    T = 0.2
    n_states = 4

R2 = R(2)
x, _ = R2.base_scalars()
t = R2.base_time()
wall = (1 - x**2) * sp.exp(-t)
hom = {"left": {"D": 0}, "right": {"D": 0}}
mov = {"left": {"D": 0}, "right": {"D": wall}}
Lx = FunctionSpace(N, Legendre.Legendre, bcs=hom, name="Lx")
Ly = FunctionSpace(N, Legendre.Legendre, bcs=mov, name="Ly")
V: DirectSumTPS = cast(DirectSumTPS, TensorProduct(Lx, Ly, name="V"))
U = V.get_orthogonal()

v = TrialFunction(V, name="v")  # constrained: no transient=True
q = TestFunction(V, name="q")
u = TrialFunction(U, name="u", transient=True)
w = TestFunction(U, name="w")
eq_u = (u.diff(t) + u + v) * w
eq_v = (Div(Grad(v)) - v + u) * q
integrator = SystemIMEXRungeKutta(
    (eq_u, eq_v),
    tableau=ARS443,
    time=(0.0, T),
    initial=(sp.Integer(0), None),
    sparse=True,
    solver_options={"auto_threshold": 1000, "kron_method": "rcm"},
)
(constraint,) = integrator.constraints
assert constraint._boundary is not None, "the moving wall should force it"
first, last = constraint.forcing_at(0.0), constraint.forcing_at(1.0)
assert first is not None and last is not None
assert not jnp.allclose(first, last)
u_hats, v_hats = integrator.solve(
    dt=0.02,
    n_batches=n_states,
    return_batch_snapshots=True,
    progress="PYTEST" not in os.environ,
)

times = jnp.linspace(0.0, T, v_hats.shape[0])
u_states = U.backward_batch(u_hats)
v_states = V.backward_batch(v_hats, t=times)

if "PYTEST" in os.environ:
    # In pytest, just check that the solution is not NaN or Inf
    assert not jnp.isnan(u_states).any()
    assert not jnp.isinf(u_states).any()
    assert not jnp.isnan(v_states).any()
    assert not jnp.isinf(v_states).any()

x_plot, y_plot = V.mesh(broadcast=False)
vmin = float(v_states.min())
vmax = float(v_states.max())
umin = float(u_states.min())
umax = float(u_states.max())
levels = 40

fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
im = ax_anim.imshow(
    v_states[0].T,
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
title = ax_anim.set_title(f"2D (t={times[0]:.3f})")


def update(frame: int):
    im.set_data(v_states[frame].T)
    title.set_text(f"2D (t={times[frame]:.3f})")
    return (im,)


_anim = FuncAnimation(fig_anim, update, frames=len(times), interval=40, blit=False)
_anim.save("transient2D_moving_bcs.gif", writer="pillow", fps=24)
plt.show()
