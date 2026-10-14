"""Tests for the monolithic IMEX Runge-Kutta integrator of coupled systems.

Three systems carry the tests:

- the Taylor-Green vortex on a periodic square, an exact Navier-Stokes solution
  whose nonlinear term is balanced by the pressure;
- a periodic channel, started from a divergence-free perturbation and driven by
  a body force, which must keep continuity at every step and settle on
  Poiseuille flow;
- a first-order wave system on a periodic line, coupled only through its
  implicit operator, which has no algebraic field and so takes every tableau.

The suite runs in float32 unless `--float64` is given, so tolerances are
chosen well above single-precision round-off.
"""

import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    CartesianProduct,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import project
from jaxfun.integrators import (
    ARK3_2_4L2SA,
    ARS443,
    IMEX_SSP2_222,
    CoupledIMEXRungeKutta,
)
from jaxfun.operators import Div, Dot, Grad

HOM = {"left": {"D": 0}, "right": {"D": 0}}


def navier_stokes(VD, Q, nu: float, force: float = 0.0, extra=0):
    """Return `(equations, (x, y), (i, j), t)` for Navier-Stokes over `VD`, `Q`."""
    W = CartesianProduct(CartesianProduct(VD, VD, name="V", rank=1), Q, name="W")
    u, p = TrialFunction(W, name="up", transient=True)
    v, q = TestFunction(W, name="vq")
    system = VD.system
    t = system.base_time()
    i, j = system.base_vectors()
    momentum = (
        Dot(u.diff(t), v)
        + Dot(Dot(Grad(u), u), v)
        - nu * Dot(Div(Grad(u)), v)
        - p * Div(v)
        - force * Dot(i, v)
        + extra * Dot(i, v)
    )
    return (momentum, Div(u) * q), system.base_scalars(), (i, j), t


def test_taylor_green_vortex() -> None:
    nu = 0.1
    F = FunctionSpace(8, Fourier, name="F")
    VD = TensorProduct(F, F, name="VD")
    Q = TensorProduct(F, F, name="Q", system=VD.system)
    equations, (x, y), (i, j), _ = navier_stokes(VD, Q, nu)
    u0 = sp.sin(x) * sp.cos(y) * i - sp.cos(x) * sp.sin(y) * j
    tg = CoupledIMEXRungeKutta(
        equations, tableau=ARS443, initial=(u0, None), sparse=True
    )
    assert tg.algebraic == (2,)

    # Two solves at different step sizes: the second retraces the nonlinear
    # terms, and nothing cached during the first trace may leak into it.
    state = tg.solve(0.01, 10, progress=False)
    ux, uy, p = tg.solve(0.02, 5, state0=state, trange=(0.1, 0.2), progress=False)
    decay = sp.exp(-2 * nu * 0.2)
    assert jnp.allclose(ux, project(sp.sin(x) * sp.cos(y) * decay, VD), atol=1e-5)
    assert jnp.allclose(uy, project(-sp.cos(x) * sp.sin(y) * decay, VD), atol=1e-5)
    # The pressure is a stage value balancing explicit terms: first order.
    pe = project((sp.cos(2 * x) + sp.cos(2 * y)) / 4 * decay**2, Q)
    assert jnp.abs(p - pe).max() < 1e-2 * jnp.abs(pe).max()


@pytest.mark.integration
def test_channel_keeps_continuity_and_reaches_poiseuille() -> None:
    nu, force = 1.0, 1.0
    F = FunctionSpace(4, Fourier, name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs=HOM, name="D")
    P = FunctionSpace(6, Legendre.Legendre, name="P")
    VD = TensorProduct(F, D, name="VD")
    Q = TensorProduct(F, P, name="Q", system=VD.system)
    equations, (x, y), (i, j), _ = navier_stokes(VD, Q, nu, force)
    u0 = -4 * y * (1 - y**2) * sp.sin(x) * i - (1 - y**2) ** 2 * sp.cos(x) * j
    channel = CoupledIMEXRungeKutta(
        equations, tableau=ARS443, initial=(u0, None), sparse=True
    )

    state = channel.solve(0.05, 4, progress=False)
    continuity = channel.apply_linear(state)[2]
    assert jnp.abs(continuity).max() < 1e-5 * jnp.abs(state[0]).max()

    ux, uy, p = channel.solve(0.25, 40, progress=False)
    assert jnp.allclose(ux, project(force * (1 - y**2) / (2 * nu), VD), atol=1e-5)
    assert jnp.abs(uy).max() < 1e-5
    assert jnp.abs(p).max() < 1e-5


@pytest.mark.parametrize(
    ("tableau", "tol"),
    [(ARS443, 1e-4), (ARK3_2_4L2SA, 1e-4), (IMEX_SSP2_222, 1e-3)],
    ids=["gsa", "implicit_sa", "general"],
)
def test_wave_system_every_combination_path(tableau, tol) -> None:
    """`u_t = w_x`, `w_t = u_x`, solved by `u = w = sin(x + t)`."""
    G = FunctionSpace(16, Fourier, name="G")
    W = CartesianProduct(G, G, name="W1")
    u, w = TrialFunction(W, name="uw", transient=True)
    a, b = TestFunction(W, name="ab")
    t = G.system.base_time()
    (x,) = G.system.base_scalars()
    equations = ((u.diff(t) - w.diff(x)) * a, (w.diff(t) - u.diff(x)) * b)
    wave = CoupledIMEXRungeKutta(
        equations, tableau=tableau, initial=(sp.sin(x), sp.sin(x)), sparse=True
    )
    assert wave.algebraic == ()
    su, sw = wave.solve(0.01, 50, progress=False)
    exact = project(sp.sin(x + 0.5), G)
    assert jnp.abs(su - exact).max() < tol
    assert jnp.abs(sw - exact).max() < tol


def periodic_spaces():
    F = FunctionSpace(4, Fourier, name="F")
    VD = TensorProduct(F, F, name="VD")
    return VD, TensorProduct(F, F, name="Q", system=VD.system)


def test_algebraic_fields_need_stiffly_accurate_tableau() -> None:
    VD, Q = periodic_spaces()
    equations, *_ = navier_stokes(VD, Q, 0.1)
    with pytest.raises(ValueError, match="algebraic"):
        CoupledIMEXRungeKutta(equations, tableau=IMEX_SSP2_222, initial=(0, None))


def test_time_dependent_source_is_refused() -> None:
    VD, Q = periodic_spaces()
    t = VD.system.base_time()
    equations, *_ = navier_stokes(VD, Q, 0.1, extra=sp.sin(t))
    with pytest.raises(NotImplementedError, match="time-dependent"):
        CoupledIMEXRungeKutta(equations, tableau=ARS443, initial=(0, None))
