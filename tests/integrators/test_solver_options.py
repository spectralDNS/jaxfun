"""Forwarding of linear-solver options through the integrators.

The operator an integrator solves against is assembled internally, so the only
way to reach its `solve` -- to widen `auto_threshold` before a sparse
tensor-product operator gives up and goes dense, say -- is through the
integrator. These tests pin that the options arrive, that each operator is
offered only what it declares, and that a misspelled option is refused instead
of being quietly dropped.
"""

import warnings

import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev as Cheb
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.integrators import ARK4_3_6L2SA, BackwardEuler, IMEXRungeKutta
from jaxfun.integrators.base import (
    _accepted_solve_options,
    _validate_solver_options,
    known_solve_options,
)
from jaxfun.la import IdentityMatrix, Matrix
from jaxfun.la.tpmatrix import TPMatrices
from jaxfun.operators import Constant, Div, Grad
from jaxfun.utils.common import n

pytestmark = pytest.mark.integration

xs, ys, ts = sp.symbols("x,y,t", real=True)


def _diffusion_1d(**params):
    """Build a 1D diffusion integrator, forwarding `params` to the constructor."""
    V = FunctionSpace(24, Cheb, name="V", fun_str="psi")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    weak_form = v * (u.diff(t) - Constant("nu", 0.5) * Div(Grad(u)))
    return V, IMEXRungeKutta(
        weak_form,
        tableau=ARK4_3_6L2SA,
        time=(0.0, 0.1),
        initial=sp.cos(sp.pi * x / 2),
        sparse=True,
        sparse_tol=1000,
        **params,
    )


def _diffusion_2d(**params):
    """Build a 2D integrator whose stage operator is a `TPMatrices`."""
    steady = sp.sinh(xs) * sp.cos(ys)
    bcsx = {"left": {"D": steady.subs(xs, -1)}, "right": {"D": steady.subs(xs, 1)}}
    bcsy = {"left": {"D": steady.subs(ys, -1)}, "right": {"D": steady.subs(ys, 1)}}
    Dx = FunctionSpace(16, Legendre, bcs=bcsx, scaling=n + 1, name="Dx", fun_str="phi")
    Dy = FunctionSpace(16, Legendre, bcs=bcsy, scaling=n + 1, name="Dy", fun_str="psi")
    V = TensorProduct(Dx, Dy, name="V")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    t = V.system.base_time()
    weak_form = v * (u.diff(t) - Constant("nu", 0.5) * Div(Grad(u)))
    u0 = steady + sp.cos(sp.pi * xs / 2) * sp.cos(sp.pi * ys / 2)
    return V, IMEXRungeKutta(
        weak_form,
        tableau=ARK4_3_6L2SA,
        time=(0.0, 0.1),
        initial=V.system.expr_psi_to_base_scalar(u0),
        sparse=True,
        sparse_tol=1000,
        **params,
    )


def test_only_declared_options_are_offered_to_an_operator() -> None:
    """Each operator sees the keyword-only options its own `solve` names."""
    assert _accepted_solve_options(TPMatrices) == {
        "method",
        "kron_method",
        "auto_threshold",
    }
    # A dense solve has nothing to choose, so it is offered nothing -- passing
    # `auto_threshold` to an integrator with a dense mass matrix is not an error.
    assert _accepted_solve_options(Matrix) == frozenset()
    assert _accepted_solve_options(IdentityMatrix) == frozenset()
    assert known_solve_options() >= {"method", "kron_method", "auto_threshold"}


def test_options_are_normalized_to_sorted_pairs() -> None:
    """Options are stored order-independently so equal configs compare equal."""
    assert _validate_solver_options(None) == ()
    assert _validate_solver_options({}) == ()
    assert _validate_solver_options({"auto_threshold": 400, "method": "lu"}) == (
        ("auto_threshold", 400),
        ("method", "lu"),
    )
    assert _validate_solver_options({"method": "lu", "auto_threshold": 400}) == (
        _validate_solver_options({"auto_threshold": 400, "method": "lu"})
    )


def test_unknown_option_is_rejected() -> None:
    """A misspelled option would otherwise be filtered out at every operator."""
    with pytest.raises(ValueError, match="auto_threshhold"):
        _diffusion_1d(solver_options={"auto_threshhold": 1000})
    with pytest.raises(ValueError, match="Accepted options are"):
        _diffusion_2d(solver_options={"not_an_option": 1})


def test_options_reach_the_stage_solve() -> None:
    """The option is forwarded, not swallowed: a bad value must be rejected."""
    _, integrator = _diffusion_2d(solver_options={"method": "no-such-method"})
    with pytest.raises(ValueError):
        integrator.solve(dt=1e-2, steps=1, progress=False)


# The bandwidth diagnostic fires alongside the fallback warning asserted below;
# `pytest.warns` re-emits whatever its pattern does not match, so drop it here.
@pytest.mark.filterwarnings(r"ignore:DiaMatrix\.lu_solve:UserWarning")
def test_auto_threshold_switches_the_solver_path() -> None:
    """`auto_threshold` must change which solver runs, not just be accepted.

    The 2D operator has bandwidth p*(q+1)=182. Under the default threshold of
    100 that exceeds the banded budget and `lu_solve` falls back to a dense
    solve, announcing it with a warning; widening the threshold past 182 keeps
    it on the banded path and the warning must disappear. The warning is the
    only externally visible evidence of which branch was taken, so it is worth
    asserting rather than filtering away.
    """
    _, default = _diffusion_2d()
    with pytest.warns(UserWarning, match="Falling back to dense solver"):
        default.solve(dt=1e-2, steps=1, progress=False)

    _, tuned = _diffusion_2d(solver_options={"auto_threshold": 10_000})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tuned.solve(dt=1e-2, steps=1, progress=False)
    assert not [w for w in caught if "Falling back to dense" in str(w.message)]


# The reference below is deliberately built without options, so it takes the
# dense fallback that `test_auto_threshold_switches_the_solver_path` asserts on
# -- that is the point of the comparison. Its warnings are expected here.
@pytest.mark.filterwarnings("ignore:Falling back to dense solver:UserWarning")
@pytest.mark.filterwarnings(r"ignore:DiaMatrix\.lu_solve:UserWarning")
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"auto_threshold": 10_000},
        {"kron_method": "banded", "auto_threshold": 10_000},
        {"method": "kron", "auto_threshold": 10_000},
    ],
)
def test_solver_options_do_not_change_the_answer_2d(options: dict) -> None:
    """Every solver path must agree: the options select speed, not semantics.

    Banded and dense substitution accumulate rounding differently, so this is a
    norm comparison rather than an elementwise one.
    """
    dt, steps = 1e-2, 10
    _, reference = _diffusion_2d()
    _, integrator = _diffusion_2d(solver_options=options)

    expected = reference.solve(dt=dt, steps=steps, progress=False)
    got = integrator.solve(dt=dt, steps=steps, progress=False)
    assert float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)) < 1e-4


def test_solver_options_reach_backward_euler() -> None:
    """The plumbing is on `BaseIntegrator`, so every integrator inherits it."""
    V = FunctionSpace(24, Cheb, name="V", fun_str="psi")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    weak_form = v * (u.diff(t) - Constant("nu", 0.5) * Div(Grad(u)))

    u0 = sp.cos(sp.pi * x / 2)
    reference = BackwardEuler(
        weak_form, time=(0.0, 0.1), initial=u0, sparse=True, sparse_tol=1000
    )
    tuned = BackwardEuler(
        weak_form,
        time=(0.0, 0.1),
        initial=u0,
        sparse=True,
        sparse_tol=1000,
        solver_options={"auto_threshold": 10_000},
    )

    expected = reference.solve(dt=1e-2, steps=10, progress=False)
    got = tuned.solve(dt=1e-2, steps=10, progress=False)
    assert float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)) < 1e-4
