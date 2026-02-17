# Solve the 1D KdV equation with RK4 in time
#
#   u_t + u u_x + mu^2 u_xxx = 0,  x in [0, 2π] periodic
#
# Spatial discretization: Fourier Galerkin (spectral)
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

from jaxfun import Domain
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ETDRK4
from jaxfun.operators import Constant

N = 512
mu = Constant("mu", sp.Rational(11, 500))
T = 1.0
steps = 100000
if "PYTEST" in os.environ:
    T = 0.02
    steps = 80

dt = T / steps

V = Fourier(N, Domain(-1, 1))
v = TestFunction(V, name="v")
u = TrialFunction(V, name="u", transient=True)

(x,) = V.system.base_scalars()
t = V.system.base_time()

u0 = sp.cos(sp.pi * x)
weak_form = v * (u.diff(t) + (u * u).diff(x) / 2 + mu**2 * u.diff(x, 3))

integrator = ETDRK4(
    V,
    weak_form,
    time=(0.0, T),
    initial=u0,
    sparse=True,
    sparse_tol=1000,
)

n_states = 200
states = integrator.solve(dt=dt, steps=steps, n_batches=n_states, return_each_step=True)
times = jnp.linspace(0.0, T, n_states + 1)

xj = V.mesh()

if "PYTEST" in os.environ:
    uhat_T = states[-1]
    u_num = V.backward(uhat_T).real
    assert jnp.isfinite(u_num).all()
    sys.exit(1)

states_phys = jnp.array([V.backward(uhat).real for uhat in states])

fig, ax = plt.subplots()
(line,) = ax.plot(xj, states_phys[0], "b")
ax.set_xlim(xj.min(), xj.max())
ax.set_ylim(states_phys.min() * 1.2, states_phys.max() * 1.2)  # ty:ignore[invalid-argument-type]
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("1D KdV equation (ETDRK4)")


def update(idx: int):
    line.set_ydata(states_phys[idx])
    ax.set_title(f"1D KdV equation (t={times[idx]:.3f})")
    return (line,)


_anim = FuncAnimation(fig, update, frames=len(times), interval=30, blit=True)
# _anim.save("kdv1d_etdrk4.gif", writer="pillow", fps=30)
plt.show()
