"""Time integration on spaces with inhomogeneous boundary conditions.

Every problem here is built the same way: an exact solution is split into a
steady harmonic part, which carries the boundary data, and a transient part
that vanishes on every boundary. That makes the boundary lifting `B` a genuine
part of the solution while keeping a closed-form reference to compare against.

The lifting enters the assembled weak form twice. Under the time derivative it
contributes `inner(v, B)`, which is constant and must drop out. In the
stiffness term it contributes the load vector `inner(v, nu*Laplace(B))`, which
must be carried on the right-hand side of every stage. Getting the second one
wrong (or keeping the first) shows up immediately as a wrong steady state.
"""

import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as Cheb
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.integrators import ARK4_3_6L2SA, BackwardEuler, IMEXRungeKutta
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import lambdify, n

pytestmark = pytest.mark.integration

xs, ys, ts = sp.symbols("x,y,t", real=True)


def _rel_error(got, expected) -> float:
    return float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected))


def test_imex_rk_1d_inhomogeneous_dirichlet() -> None:
    """u_t = nu*u_xx on [-1, 1] with u(-1) = 0, u(1) = 1.

    Exact: u = (1 + x)/2 + cos(pi*x/2)*exp(-nu*pi^2*t/4). The linear steady part
    is the boundary lifting; the transient part vanishes at both ends.
    """
    N = 24
    nu_val = 0.5
    T = 0.5
    steps = 50
    dt = T / steps

    steady = (1 + xs) / 2
    ue = steady + sp.cos(sp.pi * xs / 2) * sp.exp(-nu_val * sp.pi**2 * ts / 4)

    bcs = {"left": {"D": steady.subs(xs, -1)}, "right": {"D": steady.subs(xs, 1)}}
    V = FunctionSpace(N, Cheb, bcs=bcs, name="V", fun_str="psi")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    nu = Constant("nu", nu_val)

    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))
    integrator = IMEXRungeKutta(
        weak_form,
        tableau=ARK4_3_6L2SA,
        time=(0.0, T),
        initial=V.system.expr_psi_to_base_scalar(ue.subs(ts, 0)),
        sparse=True,
        sparse_tol=1000,
    )

    uhat_T = integrator.solve(dt=dt, steps=steps, progress=False)
    xj = V.mesh()
    u_num = V.backward(uhat_T).real
    u_exact = lambdify((x,), V.system.expr_psi_to_base_scalar(ue.subs(ts, T)))(xj)
    assert _rel_error(u_num, u_exact) < 1e-4

    # The boundary values themselves are held, not just the interior shape. The
    # quadrature mesh has no points on the boundary, so evaluate on a uniform
    # one, which does.
    u_edges = V.evaluate_mesh(uhat_T, kind="uniform", N=64).real
    assert float(abs(u_edges[0])) < 1e-4
    assert float(abs(u_edges[-1] - 1.0)) < 1e-4


def test_backward_euler_1d_reaches_inhomogeneous_steady_state() -> None:
    """The boundary lifting must survive to t -> infinity.

    Dropping `inner(v, nu*Laplace(B))` from the right-hand side would leave the
    homogeneous steady state instead, so this pins the load-vector sign as much
    as its presence.
    """
    N = 24
    nu_val = 1.0
    T = 6.0
    steps = 240
    dt = T / steps

    steady = (1 + xs) / 2
    bcs = {"left": {"D": steady.subs(xs, -1)}, "right": {"D": steady.subs(xs, 1)}}
    V = FunctionSpace(N, Cheb, bcs=bcs, name="V", fun_str="psi")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    nu = Constant("nu", nu_val)

    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))
    integrator = BackwardEuler(
        weak_form,
        time=(0.0, T),
        initial=V.system.expr_psi_to_base_scalar(steady + sp.cos(sp.pi * xs / 2)),
        sparse=True,
        sparse_tol=1000,
    )

    xj = V.mesh()
    u_num = V.backward(integrator.solve(dt=dt, steps=steps, progress=False)).real
    assert _rel_error(u_num, (1 + xj) / 2) < 1e-3


def test_imex_rk_2d_boundary_values_vary_along_the_wall() -> None:
    """u_t = nu*Laplace(u), periodic in x, Dirichlet in y with x-dependent data.

    Exact: u = cos(x)*sinh(y) + cos(x)*cos(pi*y/2)*exp(-nu*(1 + pi^2/4)*t), so
    u(x, -+1) = -+sinh(1)*cos(x).
    """
    M = 24
    nu_val = 0.5
    T = 0.5
    steps = 50
    dt = T / steps

    steady = sp.cos(xs) * sp.sinh(ys)
    transient = sp.cos(xs) * sp.cos(sp.pi * ys / 2)
    ue = steady + transient * sp.exp(-nu_val * (1 + sp.pi**2 / 4) * ts)

    bcs = {"left": {"D": steady.subs(ys, -1)}, "right": {"D": steady.subs(ys, 1)}}
    F = FunctionSpace(M, FourierSpace, name="F", fun_str="E")
    D = FunctionSpace(M, Legendre, bcs=bcs, scaling=n + 1, name="D", fun_str="psi")
    V = TensorProduct(F, D, name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    x, y = V.system.base_scalars()
    t = V.system.base_time()
    nu = Constant("nu", nu_val)

    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))
    integrator = IMEXRungeKutta(
        weak_form,
        tableau=ARK4_3_6L2SA,
        time=(0.0, T),
        initial=V.system.expr_psi_to_base_scalar(ue.subs(ts, 0)),
        sparse=True,
        sparse_tol=1000,
    )
    uhat_T = integrator.solve(dt=dt, steps=steps, progress=False)

    Np = 40
    xj = V.mesh(kind="uniform", N=(Np, Np), broadcast=True)
    u_num = V.evaluate_mesh(uhat_T, kind="uniform", N=(Np, Np)).real
    u_exact = lambdify((x, y), V.system.expr_psi_to_base_scalar(ue.subs(ts, T)))(*xj)
    assert _rel_error(u_num, u_exact) < 1e-4


def test_imex_rk_2d_inhomogeneous_in_both_directions() -> None:
    """Both tensor factors carry inhomogeneous, spatially varying data.

    Exact: u = sinh(x)*cos(y) + cos(pi*x/2)*cos(pi*y/2)*exp(-nu*pi^2*t/2). The
    steady part is harmonic, so it is the boundary lifting on all four walls,
    and the transient part vanishes on each of them.
    """
    M = 20
    nu_val = 0.5
    T = 0.5
    steps = 50
    dt = T / steps

    steady = sp.sinh(xs) * sp.cos(ys)
    transient = sp.cos(sp.pi * xs / 2) * sp.cos(sp.pi * ys / 2)
    ue = steady + transient * sp.exp(-nu_val * sp.pi**2 * ts / 2)

    bcsx = {"left": {"D": steady.subs(xs, -1)}, "right": {"D": steady.subs(xs, 1)}}
    bcsy = {"left": {"D": steady.subs(ys, -1)}, "right": {"D": steady.subs(ys, 1)}}
    Dx = FunctionSpace(M, Legendre, bcs=bcsx, scaling=n + 1, name="Dx", fun_str="phi")
    Dy = FunctionSpace(M, Legendre, bcs=bcsy, scaling=n + 1, name="Dy", fun_str="psi")
    V = TensorProduct(Dx, Dy, name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    x, y = V.system.base_scalars()
    t = V.system.base_time()
    nu = Constant("nu", nu_val)

    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))
    integrator = IMEXRungeKutta(
        weak_form,
        tableau=ARK4_3_6L2SA,
        time=(0.0, T),
        initial=V.system.expr_psi_to_base_scalar(ue.subs(ts, 0)),
        sparse=True,
        sparse_tol=1000,
    )
    uhat_T = integrator.solve(dt=dt, steps=steps, progress=False)

    Np = 40
    xj = V.mesh(kind="uniform", N=(Np, Np), broadcast=True)
    u_num = V.evaluate_mesh(uhat_T, kind="uniform", N=(Np, Np)).real
    u_exact = lambdify((x, y), V.system.expr_psi_to_base_scalar(ue.subs(ts, T)))(*xj)
    assert _rel_error(u_num, u_exact) < 1e-4
