import jax.numpy as jnp
import pytest
import sympy as sp

import jaxfun.integrators.nonlinear as integrator_nonlinear
from jaxfun import Domain
from jaxfun.galerkin import TensorProductSpace
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as Cheb
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ETDRK4, RK4, BackwardEuler
from jaxfun.operators import Constant, Div, Grad
from jaxfun.typing import MeshKind
from jaxfun.utils.common import lambdify, n


def test_backward_euler_linear_advection_fourier() -> None:
    N = 32
    c = Constant("c", 1.0)
    T = 0.2
    steps = 200
    dt = T / steps

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(x)
    weak_form = v * (u.diff(t) + c * u.diff(x))

    integrator = BackwardEuler(
        V,
        weak_form,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    u_num = V.backward(uhat_t).real
    u_ex = lambdify(x, sp.sin(x - c.val * T))(xj)
    rel_error = jnp.linalg.norm(u_num - u_ex) / jnp.linalg.norm(u_ex)
    assert float(rel_error) < 3e-2


def test_rk4_solve_uses_initial_and_time_api() -> None:
    M = 16
    nu = Constant("nu", 0.1)
    T = 0.1
    steps = 200
    dt = T / steps

    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    V = FunctionSpace(M, Cheb, bcs=bcs, name="V", fun_str="psi", scaling=n + 1)

    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(sp.pi * (x + 1) / 2)
    weak_form = v * (u.diff(t) - nu * Div(Grad(u)))

    integrator = RK4(
        V,
        weak_form,
        time=(0.0, T),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )
    uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)

    xj = V.mesh()
    u_num = V.backward(uhat_t).real
    k = sp.pi / 2
    u_ex = lambdify(x, sp.sin(k * (x + 1)) * sp.exp(-nu.val * float(k**2) * T))(xj)
    error = jnp.linalg.norm(u_num - u_ex)
    assert float(error) < 2e-1


def test_solve_n_batches_and_return_batch_snapshots() -> None:
    N = 16
    dt = 0.01
    steps = 23

    V = FunctionSpace(N, FourierSpace, name="V", fun_str="E")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(x)
    weak_form = v * (u.diff(t) + u.diff(x))
    integrator = RK4(V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True)

    states = integrator.solve(
        dt=dt,
        steps=steps,
        n_batches=5,
        return_batch_snapshots=True,
        progress=False,
    )
    integrator_final = RK4(
        V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True
    )
    final = integrator_final.solve(dt=dt, steps=steps, n_batches=5, progress=False)
    integrator_restart = RK4(
        V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True
    )
    restarted = integrator_restart.solve(
        dt=dt,
        steps=steps,
        n_batches=5,
        state0=integrator_final.initial_coefficients(),
        progress=False,
    )

    # 23 steps in 5 batches => 1 initial + 5 snapshots + 1 remainder snapshot.
    assert states.shape == (7, V.num_dofs)
    assert jnp.allclose(states[-1], final)
    assert jnp.allclose(restarted, final)

    with pytest.raises(ValueError, match="n_batches must be a positive integer"):
        _ = integrator.solve(dt=dt, steps=steps, n_batches=0, progress=False)


def test_rk4_nonlinear_rhs_uses_direct_jaxfunction_evaluation() -> None:
    N = 32
    V = FunctionSpace(
        N,
        FourierSpace,
        domain=Domain(0.0, 2.0 * float(sp.pi)),
        name="V",
        fun_str="E",
    )
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    u0 = sp.sin(x) + sp.cos(2 * x)
    weak_form = v * (u.diff(t) + u * u.diff(x))
    integrator = RK4(V, weak_form, time=(0.0, 0.01), initial=u0, sparse=True)

    uhat = integrator.initial_coefficients()
    nonlinear = integrator.nonlinear_rhs(uhat)

    xj = V.mesh()
    u_phys = V.backward(uhat)
    ux_phys = V.evaluate_derivative(xj, uhat, k=1)
    expected = V.forward(-u_phys * ux_phys)

    assert jnp.allclose(nonlinear, expected, atol=1e-6, rtol=1e-6)


def test_rk4_nonlinear_rhs_caches_repeated_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    N = 32
    V = FunctionSpace(
        N, FourierSpace, domain=Domain(0.0, 2.0 * float(sp.pi)), name="V", fun_str="E"
    )
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    # Contains repeated subterms of both u and u_x after expansion.
    weak_form = v * (u.diff(t) + (u + u.diff(x)) ** 2)
    integrator = RK4(V, weak_form, time=(0.0, 0.01), initial=sp.sin(x), sparse=True)
    uhat = integrator.initial_coefficients()
    u_phys = V.backward(uhat)
    ux_phys = V.evaluate_derivative(V.mesh(), uhat, k=1)
    expected = V.forward(-((u_phys + ux_phys) ** 2))

    calls: list[int] = []
    original_backward = integrator.functionspace.backward
    original_eval = integrator.functionspace.backward_primitive

    def count_backward(
        c,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: int | None = None,
    ):
        calls.append(0)
        return original_backward(c, kind=kind, N=N)

    def count_eval(
        c,
        k: int = 0,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: int | None = None,
    ):
        calls.append(int(k))
        return original_eval(c, k=k, kind=kind, N=N)

    monkeypatch.setattr(integrator.functionspace, "backward", count_backward)
    monkeypatch.setattr(integrator.functionspace, "backward_primitive", count_eval)

    nonlinear = integrator.nonlinear_rhs(uhat)

    assert jnp.allclose(nonlinear, expected, atol=1e-6, rtol=1e-6)

    primitive_orders = {
        0
        if integrator_nonlinear._is_jaxfunction_leaf(node)
        else int(sp.sympify(node).derivative_count)  # ty:ignore[unresolved-attribute]
        for node in sp.core.traversal.preorder_traversal(integrator.nonlinear_expr)
        if integrator_nonlinear._is_jaxfunction_primitive(node)
    }

    # The derivative primitive should be evaluated once even though it appears
    # repeatedly after expansion; the base field now uses `backward(...)`.
    assert primitive_orders == {0, 1}
    assert calls.count(1) == 1
    assert calls.count(0) >= 1


def test_rk4_solve_passes_padding_to_nonlinear_backward_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    N = 32
    pad = 48
    dt = 1e-3
    steps = 2

    V = FunctionSpace(
        N, FourierSpace, domain=Domain(0.0, 2.0 * float(sp.pi)), name="V", fun_str="E"
    )
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()

    weak_form = v * (u.diff(t) + u * u.diff(x))
    integrator = RK4(V, weak_form, time=(0.0, dt * steps), initial=sp.sin(x))

    calls: list[int | None] = []
    original_eval = integrator.functionspace.backward_primitive

    def count_eval(
        c,
        k: int = 0,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: int | None = None,
    ):
        calls.append(N)
        return original_eval(c, k=k, kind=kind, N=N)

    monkeypatch.setattr(integrator.functionspace, "backward_primitive", count_eval)

    _ = integrator.solve(dt=dt, steps=steps, N=pad, progress=False)

    assert calls
    assert set(calls) == {pad}


def test_rk4_tensorproduct_projects_symbolic_initial_condition() -> None:
    N = 16
    dt = 1e-3
    steps = 4

    F = FourierSpace(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F), name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    x, y = V.system.base_scalars()

    u0 = sp.cos(sp.pi * x) * sp.cos(sp.pi * y)
    weak_form = v * (u.diff(t) + u.diff(x) + u.diff(y))
    integrator = RK4(V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True)

    uhat0 = integrator.initial_coefficients()
    assert uhat0.shape == V.num_dofs

    uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)
    assert uhat_t.shape == V.num_dofs
    assert bool(jnp.isfinite(V.backward(uhat_t)).all())


def test_rk4_tensorproduct_projects_x_only_initial_condition() -> None:
    N = 16
    dt = 1e-3
    steps = 2

    F = FourierSpace(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F), name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    x, y = V.system.base_scalars()

    u0 = sp.cos(sp.pi * x)
    weak_form = v * (u.diff(t) + u.diff(x) + u.diff(y))
    integrator = RK4(V, weak_form, time=(0.0, dt * steps), initial=u0, sparse=True)

    uhat0 = integrator.initial_coefficients()
    assert uhat0.shape == V.num_dofs

    u0_phys = V.backward(uhat0).real
    xj, _yj = V.mesh()
    expected = jnp.broadcast_to(jnp.cos(jnp.pi * xj), u0_phys.shape)
    assert jnp.allclose(u0_phys, expected, atol=1e-6, rtol=1e-6)

    uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)
    assert uhat_t.shape == V.num_dofs
    assert bool(jnp.isfinite(V.backward(uhat_t)).all())


def test_rk4_tensorproduct_nonlinear_rhs_uses_mixed_derivative_path() -> None:
    N = 16
    F = FourierSpace(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F), name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    x, y = V.system.base_scalars()

    u0 = sp.cos(sp.pi * x) * sp.cos(sp.pi * y)
    weak_form = v * (u.diff(t) + u * u.diff(x) + u * u.diff(y))
    integrator = RK4(V, weak_form, time=(0.0, 0.01), initial=u0, sparse=True)

    uhat = integrator.initial_coefficients()
    nonlinear = integrator.nonlinear_rhs(uhat)

    xj = list(V.mesh())
    u_phys = V.backward(uhat)
    ux_phys = V.evaluate_derivative(xj, uhat, k=(1, 0))
    uy_phys = V.evaluate_derivative(xj, uhat, k=(0, 1))
    expected = V.forward(-(u_phys * ux_phys + u_phys * uy_phys))

    assert jnp.allclose(nonlinear, expected, atol=2e-6, rtol=2e-6)


def test_etdrk4_tensorproduct_nonlinear_short_run_is_finite() -> None:
    N = 16
    mu = Constant("mu", sp.Rational(3, 20))
    dt = 2e-4
    steps = 8

    F = FourierSpace(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F), name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    x, y = V.system.base_scalars()

    u0 = sp.cos(sp.pi * x) * sp.cos(sp.pi * y)
    laplace_u = u.diff(x, 2) + u.diff(y, 2)
    weak_form = v * (u.diff(t) + (u * u).diff(x) / 2 + mu**2 * laplace_u.diff(x))
    integrator = ETDRK4(
        V,
        weak_form,
        time=(0.0, dt * steps),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
    )

    states = integrator.solve(
        dt=dt,
        steps=steps,
        n_batches=4,
        return_batch_snapshots=True,
        progress=False,
    )
    assert states.shape == (5,) + V.num_dofs

    u0_phys = V.backward(states[0]).real
    uT_phys = V.backward(states[-1]).real
    assert bool(jnp.isfinite(uT_phys).all())
    assert float(jnp.linalg.norm(uT_phys - u0_phys)) > 1e-8
