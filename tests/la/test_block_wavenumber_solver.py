"""Tests for factored solves of block (multi-field) tensor-product systems.

Most systems are a small Stokes-type saddle point, `(u, v) + nu (grad u, grad v)
- (p, div v) + (div u, q)`, whose pressure is fixed only up to a constant. The
right-hand sides are made from a known solution with that constant removed, so
the solve has to recover it exactly.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.galerkin import (
    CartesianProduct,
    Chebyshev,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import inner
from jaxfun.la import BlockArray, BlockMatrix
from jaxfun.la.blockwavenumber import (
    BlockBandedWavenumberSolver,
    BlockGlobalSolver,
    BlockWavenumberSolver,
    block_lu_factor,
)
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import ulp

HOM = {"left": {"D": 0}, "right": {"D": 0}}


# The Chebyshev operator's two undetermined k = 0 pressure modes.
CHEB_PINS = [(2, (0, 0)), (2, (0, 7))]


@functools.cache
def stokes_operator(kind: str, N: int = 8, M: int = 6):
    """Return the terms and spaces of a Stokes-type block operator.

    `kind` picks the axes: "fourier_poly" (Fourier x Legendre), "fourier_cheb"
    (Fourier x Chebyshev, tested with GR spaces), "fourier" (Fourier x Fourier)
    or "poly" (Legendre x Legendre). `N` is the number of wall-normal modes and
    `M` the number of Fourier modes along the first axis.
    """
    if kind == "fourier_cheb":
        return _chebyshev_stokes(N, M)
    if kind == "fourier_poly":
        X, Y = (
            FunctionSpace(M, Fourier, name="F"),
            FunctionSpace(N, Legendre.Legendre, bcs=HOM, name="D"),
        )
        XP, YP = X, FunctionSpace(N - 2, Legendre.Legendre, name="P")
    elif kind == "fourier":
        X, Y = FunctionSpace(M, Fourier, name="F"), FunctionSpace(4, Fourier, name="G")
        XP, YP = X, Y
    else:
        X, Y = (FunctionSpace(8, Legendre.Legendre, bcs=HOM, name=n) for n in "XY")
        XP, YP = (FunctionSpace(6, Legendre.Legendre, name=n) for n in ("XP", "YP"))
    VD = TensorProduct(X, Y, name="VD")
    Q = TensorProduct(XP, YP, name="Q", system=VD.system)
    W = CartesianProduct(CartesianProduct(VD, VD, name="V", rank=1), Q, name="W")
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")
    A = inner(Dot(u, v) - 0.1 * Dot(Div(Grad(u)), v) - p * Div(v), sparse=True)
    B = inner(Div(u) * q, sparse=True)
    assert isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix)
    return list(A.mats) + list(B.mats), A.test_space, A.trial_space


def _chebyshev_stokes(N: int, M: int):
    """Fourier x Chebyshev Stokes with GR test spaces, the pressure's of order 0."""
    F = FunctionSpace(M, Fourier, name="F")
    D = FunctionSpace(N, Chebyshev.Chebyshev, bcs=HOM, name="D")
    P = FunctionSpace(N, Chebyshev.Chebyshev, name="P")
    VD = TensorProduct(F, D, name="VD")
    Q = TensorProduct(F, P, name="Q", system=VD.system)
    W = CartesianProduct(CartesianProduct(VD, VD, name="V", rank=1), Q, name="W")
    DG = TensorProduct(F, D.get_testspace("GR"), name="DG", system=VD.system)
    QG = TensorProduct(F, P.get_testspace("GR"), name="QG", system=VD.system)
    WT = CartesianProduct(CartesianProduct(DG, DG, rank=1), QG, name="WT")
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(WT, name="vq")
    A = inner(Dot(u, v) - 0.1 * Dot(Div(Grad(u)), v) + Dot(Grad(p), v), sparse=True)
    B = inner(Div(u) * q, sparse=True)
    assert isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix)
    return list(A.mats) + list(B.mats), A.test_space, A.trial_space


def advection_diffusion_operator():
    """Two fields whose (u, u) block mixes real (k^2) and imaginary (ik) weights."""
    F = FunctionSpace(6, Fourier, name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs=HOM, name="D")
    VD = TensorProduct(F, D, name="VD")
    W = CartesianProduct(VD, VD, name="W2")
    u, w = TrialFunction(W, name="uw")
    v, s = TestFunction(W, name="vs")
    x = VD.system.base_scalars()[0]
    A = inner((u + u.diff(x) - 0.1 * Div(Grad(u)) + w) * v, sparse=True)
    B = inner((w - 0.1 * Div(Grad(w)) + u) * s, sparse=True)
    assert isinstance(A, BlockMatrix) and isinstance(B, BlockMatrix)
    return list(A.mats) + list(B.mats), A.test_space, A.trial_space


def manufactured(mats, test_space, trial_space, complex_valued: bool, pins=()):
    """Return `(x, b)` with `b = A x` for a random `x` whose pressure mean is 0.

    The unknowns that `pins` fix are zero too.
    """
    keys = jax.random.split(jax.random.PRNGKey(3), 2 * len(trial_space.num_dofs))
    x = []
    for c, nd in enumerate(trial_space.num_dofs):
        xc = jax.random.normal(keys[2 * c], nd)
        if complex_valued:
            xc = xc + 1j * jax.random.normal(keys[2 * c + 1], nd)
        x.append(xc)
    x[-1] = x[-1].at[(0,) * x[-1].ndim].set(0)
    for c, index in pins:
        x[c] = x[c].at[index].set(0)
    A = BlockMatrix(mats, test_space, trial_space)
    b = (A @ BlockArray(trial_space, tuple_array=tuple(x))).array
    return tuple(x), b


WAVENUMBER = {"dense": BlockWavenumberSolver, "rcm": BlockBandedWavenumberSolver}


@pytest.mark.parametrize("method", ["dense", "rcm"])
@pytest.mark.parametrize("kind", ["fourier_poly", "fourier_cheb", "fourier", "poly"])
def test_block_solve_recovers_solution(kind, method) -> None:
    mats, test_space, trial_space = stokes_operator(kind)
    pins = CHEB_PINS if kind == "fourier_cheb" else []
    solver = block_lu_factor(mats, test_space, trial_space, pins, method=method)
    assert isinstance(
        solver, BlockGlobalSolver if kind == "poly" else WAVENUMBER[method]
    )
    x, b = manufactured(mats, test_space, trial_space, kind != "poly", pins)
    for xc, sc in zip(x, solver.solve(b), strict=True):
        assert sc.shape == xc.shape
        assert jnp.allclose(sc, xc, atol=float(1000 * ulp(1.0)))


def test_dense_factors_are_real_through_phases() -> None:
    """Stokes is complex only through ik, which phases on u and its row remove."""
    mats, test_space, trial_space = stokes_operator("fourier_poly")
    solver = block_lu_factor(mats, test_space, trial_space, method="dense")
    assert isinstance(solver, BlockWavenumberSolver)
    assert not jnp.iscomplexobj(solver.lu)


@pytest.mark.parametrize("method", ["dense", "rcm"])
def test_mixed_block_stays_complex(method) -> None:
    """A block with both k^2 and ik has no real phases; the solve is still exact."""
    mats, test_space, trial_space = advection_diffusion_operator()
    solver = block_lu_factor(mats, test_space, trial_space, method=method)
    if isinstance(solver, BlockWavenumberSolver):
        assert jnp.iscomplexobj(solver.lu)
    keys = jax.random.split(jax.random.PRNGKey(5), 4)
    x = tuple(
        jax.random.normal(keys[2 * c], nd) + 1j * jax.random.normal(keys[2 * c + 1], nd)
        for c, nd in enumerate(trial_space.num_dofs)
    )
    A = BlockMatrix(mats, test_space, trial_space)
    b = (A @ BlockArray(trial_space, tuple_array=x)).array
    for xc, sc in zip(x, solver.solve(b), strict=True):
        assert jnp.allclose(sc, xc, atol=float(1000 * ulp(1.0)))


def test_auto_picks_banded_for_a_narrow_band() -> None:
    small = stokes_operator("fourier_poly")
    large = stokes_operator("fourier_poly", 40)
    assert isinstance(block_lu_factor(*small), BlockWavenumberSolver)
    assert isinstance(block_lu_factor(*large), BlockBandedWavenumberSolver)


@pytest.mark.parametrize("method", ["dense", "rcm"])
def test_constant_pressure_is_pinned_automatically(method) -> None:
    """The k = 0 continuity row and pressure column are both structurally zero."""
    mats, test_space, trial_space = stokes_operator("fourier_poly")
    solver = block_lu_factor(mats, test_space, trial_space, method=method)
    assert isinstance(solver, WAVENUMBER[method])
    first_pressure_row = sum(nd[1] for nd in test_space.num_dofs[:2])
    assert (0, first_pressure_row) in solver.pinned_rows


@pytest.mark.parametrize("method", ["dense", "rcm"])
def test_explicit_pin_matches_automatic(method) -> None:
    mats, test_space, trial_space = stokes_operator("fourier_poly")
    auto = block_lu_factor(mats, test_space, trial_space, method=method)
    explicit = block_lu_factor(
        mats, test_space, trial_space, pins=[(2, (0, 0))], method=method
    )
    assert isinstance(auto, WAVENUMBER[method])
    assert isinstance(explicit, WAVENUMBER[method])
    assert explicit.pinned_rows == auto.pinned_rows
    _, b = manufactured(mats, test_space, trial_space, True)
    for a, e in zip(auto.solve(b), explicit.solve(b), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(e))


def test_pin_out_of_range_raises() -> None:
    mats, test_space, trial_space = stokes_operator("fourier_poly")
    with pytest.raises(IndexError):
        block_lu_factor(mats, test_space, trial_space, pins=[(2, (0, 99))])
