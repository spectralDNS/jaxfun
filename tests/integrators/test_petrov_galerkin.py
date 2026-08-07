import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as Cheb
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import ARK4_3_6L2SA, IMEXRungeKutta
from jaxfun.operators import Constant, Div, Grad

pytestmark = pytest.mark.integration


def test_imex_burgers_petrov_galerkin_matches_galerkin() -> None:
    """A Petrov-Galerkin (different test/trial spaces) and a plain Galerkin
    (test space == trial space) IMEX-RK discretization of the same nonlinear
    Burgers problem should converge to the same physical solution.

    Both are distinct, independently-consistent discretizations of the same
    continuous PDE, so their coefficients differ, but at a shared resolution
    and small enough `dt` their physical-space solutions (evaluated on the
    same grid via the shared trial space `V`) should agree closely. This is
    the nonlinear/PG counterpart to the linear diffusion accuracy check in
    `test_imex_rk_empirical_convergence_order` -- Burgers has no simple
    closed-form solution to compare against directly, so cross-checking two
    independent discretizations against each other is the correctness signal
    (mirroring `test_ark4_kdv_soliton_matches_etdrk4_reference`'s use of a
    second integrator as the reference rather than an analytic solution).

    """
    M = 24
    nu = Constant("nu", 0.1)
    dt = 1e-3
    steps = 100
    T = dt * steps

    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    V = FunctionSpace(M, Cheb, bcs=bcs, name="V", fun_str="psi")
    P = V.get_testspace("PG", name="P")
    assert P is not V

    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    u0 = sp.sin(sp.pi * (x + 1))

    def burgers_solution(test_space) -> tuple[IMEXRungeKutta, jnp.ndarray]:
        v = TestFunction(test_space, name="v")
        weak_form = v * (u.diff(t) + u * u.diff(x) - nu * Div(Grad(u)))
        integrator = IMEXRungeKutta(
            weak_form,
            tableau=ARK4_3_6L2SA,
            time=(0.0, T),
            initial=u0,
            sparse=True,
            sparse_tol=1000,
        )
        uhat_t = integrator.solve(dt=dt, steps=steps, progress=False)
        return integrator, V.backward(uhat_t).real

    integrator_g, u_g = burgers_solution(V)
    integrator_pg, u_pg = burgers_solution(P)

    assert integrator_g.testspace is integrator_g.trialspace
    assert integrator_pg.testspace is P
    assert integrator_pg.trialspace is V

    assert bool(jnp.isfinite(u_g).all())
    assert bool(jnp.isfinite(u_pg).all())

    rel_diff = jnp.linalg.norm(u_pg - u_g) / jnp.linalg.norm(u_g)
    assert float(rel_diff) < 1e-3
