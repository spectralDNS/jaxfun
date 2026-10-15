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
    IMEXRungeKutta,
)
from jaxfun.integrators.base import _advance
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import ulp

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

    # Solved for from the velocity, it is as accurate as the velocity, and
    # asking for it leaves the compiled step alone.
    compiled = _advance._cache_size()  # ty: ignore[unresolved-attribute]
    _, _, p_consistent = tg.consistent_algebraic((ux, uy, p))
    assert jnp.abs(p_consistent - pe).max() < 1e-5 * jnp.abs(pe).max()
    tg.solve(0.02, 1, state0=(ux, uy, p), progress=False)
    assert _advance._cache_size() == compiled  # ty: ignore[unresolved-attribute]


def test_consistent_pressure_at_rest_is_hydrostatic() -> None:
    """At rest in the conducting profile the pressure balances buoyancy alone.

    Wall-normal only: velocity v, pressure p and temperature T between walls.
    The profile is the temperature's boundary lifting, so it enters through the
    forcing, and the buoyancy through a linear term between transient fields.
    """
    N = 8
    poly = Legendre.Legendre
    D = FunctionSpace(N, poly, bcs=HOM, name="D")
    P = FunctionSpace(N, poly, name="P")
    Tb = FunctionSpace(N, poly, bcs={"left": {"D": 1}, "right": {"D": 0}}, name="Tb")
    W = CartesianProduct(D, P, Tb, name="W")
    v, p, T = TrialFunction(W, name="vpT", transient=True)
    w, q, s = TestFunction(W, name="wqs")
    t = D.system.base_time()
    y = D.system.base_scalars()[0]
    equations = (
        (v.diff(t) + p.diff(y) - T) * w,
        v.diff(y) * q,
        (T.diff(t) - T.diff(y, 2)) * s,
    )
    rb = CoupledIMEXRungeKutta(
        equations,
        tableau=ARS443,
        initial=(0, None, 0),
        pin={p: [(0,), (N - 1,)]},
        sparse=True,
    )
    # dp/dy = T = (1 - y)/2, with the constant pinned away.
    _, p_rest, _ = rb.consistent_algebraic(rb.initial_coefficients())
    pe = project(y / 2 - y**2 / 4, P).at[0].set(0)
    assert jnp.allclose(p_rest, pe, atol=float(1000 * ulp(1.0)))


@pytest.mark.parametrize("family", ["legendre", "fourier"])
def test_one_dimensional_system_matches_the_single_field_integrator(family) -> None:
    """Burgers for u, and a scalar w that u drives but that does not act back.

    u is then exactly what `IMEXRungeKutta` computes for Burgers alone: the same
    tableau, with the same implicit and explicit split. On Legendre the state is
    real, and must stay so.
    """
    nu, kappa = 0.05, 0.1
    if family == "legendre":
        U = FunctionSpace(16, Legendre.Legendre, bcs=HOM, name="U")
        S = FunctionSpace(16, Legendre.Legendre, bcs=HOM, name="S")
        x = U.system.base_scalars()[0]
        u0 = -sp.sin(sp.pi * x)
    else:
        U = FunctionSpace(16, Fourier, name="U")
        S = FunctionSpace(16, Fourier, name="S")
        x = U.system.base_scalars()[0]
        u0 = sp.sin(x)
    t = U.system.base_time()
    u, w = TrialFunction(CartesianProduct(U, S, name="C"), name="uw", transient=True)
    v, s = TestFunction(CartesianProduct(U, S, name="C"), name="vs")
    coupled = CoupledIMEXRungeKutta(
        (
            (u.diff(t) + u * u.diff(x) - nu * u.diff(x, 2)) * v,
            (w.diff(t) - kappa * w.diff(x, 2) - u) * s,
        ),
        tableau=ARS443,
        initial=(u0, 0),
        sparse=True,
    )
    ub, vb = TrialFunction(U, name="ub", transient=True), TestFunction(U, name="vb")
    burgers = IMEXRungeKutta(
        (ub.diff(t) + ub * ub.diff(x) - nu * ub.diff(x, 2)) * vb,
        tableau=ARS443,
        initial=u0,
        sparse=True,
    )
    u_coupled, w_coupled = coupled.solve(0.01, 20, progress=False)
    u_single = burgers.solve(0.01, 20, progress=False)
    assert jnp.allclose(u_coupled, u_single, atol=float(100 * ulp(1.0)))
    assert jnp.abs(w_coupled).max() > 0


def test_initial_condition_as_coefficients() -> None:
    """Coefficient arrays work as initial values, held as data on the stepper."""
    F = FunctionSpace(8, Fourier, name="F")
    VD = TensorProduct(F, F, name="VD")
    Q = TensorProduct(F, F, name="Q", system=VD.system)
    equations, (x, y), (i, j), _ = navier_stokes(VD, Q, 0.1)
    u0 = sp.sin(x) * sp.cos(y) * i - sp.cos(x) * sp.sin(y) * j
    symbolic = CoupledIMEXRungeKutta(
        equations, tableau=ARS443, initial=(u0, None), sparse=True
    )
    ux, uy, _ = symbolic.initial_coefficients()
    arrays = CoupledIMEXRungeKutta(
        equations, tableau=ARS443, initial=((ux, uy), None), sparse=True
    )
    for a, b in zip(
        arrays.solve(0.01, 2, progress=False),
        symbolic.solve(0.01, 2, progress=False),
        strict=True,
    ):
        assert jnp.array_equal(a, b)


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


def test_vector_space_alone() -> None:
    """A vector space on its own is one field: `u = i + exp(-nu t) sin(x - t) j`.

    The unit x-velocity advects the y-velocity through `Dot(Grad(u), u)`, while
    the viscous term damps it.
    """
    nu = 0.1
    F = FunctionSpace(8, Fourier, name="F")
    VD = TensorProduct(F, F, name="VD")
    W = CartesianProduct(VD, VD, name="W", rank=1)
    u = TrialFunction(W, name="u", transient=True)
    v = TestFunction(W, name="v")
    t = W.system.base_time()
    x, _ = W.system.base_scalars()
    i, j = W.system.base_vectors()
    equation = Dot(u.diff(t), v) + Dot(Dot(Grad(u), u), v) - nu * Dot(Div(Grad(u)), v)
    solver = CoupledIMEXRungeKutta(
        equation, tableau=ARS443, initial=(i + sp.sin(x) * j,), sparse=True
    )
    assert solver.algebraic == ()
    ux, uy = solver.solve(0.01, 20, progress=False)
    assert jnp.allclose(ux, project(sp.S.One, VD), atol=1e-5)
    exact = project(sp.sin(x - 0.2) * sp.exp(-nu * 0.2), VD)
    assert jnp.abs(uy - exact).max() < 1e-5


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
