"""`get_testspace("GR")`: Petrov-Galerkin test functions cropped to the trial space.

The cropped space spans the trial space, so it is the Galerkin method, while its
operator matrices keep the Petrov-Galerkin band.
"""

from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as npcheb, legendre as nplegendre

import jaxfun.typing as jt
from jaxfun.galerkin import (
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.composite import GalerkinRecombined
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.la.tpmatrix import TPMatrices, TPMatricesWavenumberSolver

DIRICHLET = {"left": {"D": 0}, "right": {"D": 0}}
BIHARMONIC = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
CASES = [
    (DIRICHLET, {0: [-2, 0, 2, 4], 1: [-1, 1, 3], 2: [0, 2]}),
    (BIHARMONIC, {0: [-4, -2, 0, 2, 4, 6, 8], 2: [-2, 0, 2, 4, 6], 4: [0, 2, 4]}),
]


def _tol() -> float:
    return 1e3 * float(jnp.finfo(jnp.result_type(float)).eps)


def _exact(test_rows, trial_rows, j: int, family) -> np.ndarray:
    """Return (trial^{(j)}, test)_w exactly for the given (float) stencil rows.

    Quadrature would not do as a reference: it loses about N^(2j-1) to
    cancellation, already ~1e-12 at N = 24 for j = 4.
    """
    der = npcheb.chebder if family is Chebyshev else nplegendre.legder
    N = test_rows.shape[1]
    if family is Chebyshev:  # (T_n, T_n)_w / pi
        norms = [Fraction(1)] + [Fraction(1, 2)] * (N - 1)
        factor = np.pi
    else:
        norms = [Fraction(2, 2 * k + 1) for k in range(N)]
        factor = 1.0
    test = [[Fraction(float(v)) for v in r] for r in test_rows]
    trial = []
    for r in trial_rows:
        c = np.array([Fraction(float(v)) for v in r], dtype=object)
        for _ in range(j):
            c = der(c)
        trial.append(list(c) + [Fraction(0)] * (N - len(c)))
    out = np.zeros((len(test), len(trial)))
    for k, v in enumerate(test):
        for l, u in enumerate(trial):
            terms = (a * b * w for a, b, w in zip(v, u, norms, strict=True) if a and b)
            out[k, l] = factor * float(sum(terms, Fraction(0)))
    return out


def _offsets(A) -> list[int]:
    X = np.asarray(A.todense())
    # Row by row: the PG rows are scaled ~k^-q against the replaced trial rows.
    X = X / np.maximum(np.abs(X).max(axis=1, keepdims=True), 1e-300)
    nz = np.argwhere(np.abs(X) > 1e-6)
    return sorted(set((nz[:, 1] - nz[:, 0]).tolist()))


def test_coerce():
    kind = jt.TestSpaceKind
    assert kind.coerce("GR") is kind.GALERKIN_RECOMBINED
    assert kind.coerce("Galerkin-recombined") is kind.GR


@pytest.mark.parametrize("family", [Chebyshev, Legendre])
@pytest.mark.parametrize("bcs,bands", CASES)
def test_spans_trial_space_with_pg_band(family, bcs, bands):
    N = 24
    B = FunctionSpace(N, family, bcs=bcs)
    G = B.get_testspace("GR")
    assert isinstance(G, GalerkinRecombined)
    assert G.dim == B.dim and G.N == B.N
    St, Sg = np.asarray(B.S.todense()), np.asarray(G.S.todense())
    assert np.abs(Sg - (Sg @ np.linalg.pinv(St)) @ St).max() < _tol()
    rows = Sg / np.abs(Sg).max(axis=1, keepdims=True)  # PG rows are scaled ~k^-q
    assert np.linalg.matrix_rank(rows.astype(float)) == B.dim
    x = B.system.x
    u, v = TrialFunction(B), TestFunction(G)
    # The rows that stay PG must be PG's own (exact, recurrence-built) rows; the
    # replaced ones are checked against exact rational arithmetic.
    pg = B.get_testspace("PG")
    replaced = np.any(np.asarray(pg.S.todense())[:, N:] != 0, axis=1)
    assert 0 < replaced.sum() < B.dim and replaced[-1]
    for j, offsets in bands.items():
        form = u.diff(x, j) if j else u
        A = np.asarray(inner(form * v, sparse=True, kind="bilinear").todense())
        ref = np.array(
            inner(form * TestFunction(pg), sparse=True, kind="bilinear").todense()
        )
        ref[replaced] = _exact(Sg[replaced], St, j, family)
        err = np.abs(A - ref).max(axis=1)
        assert np.all(err <= _tol() * np.abs(ref).max(axis=1)), (j, err.max())
        assert _offsets(inner(form * v, sparse=True, kind="bilinear")) == offsets


@pytest.mark.parametrize("family", [Chebyshev, Legendre])
@pytest.mark.parametrize("bcs", [DIRICHLET, BIHARMONIC])
def test_solution_is_the_galerkin_solution(family, bcs):
    N = 30
    B = FunctionSpace(N, family, bcs=bcs)
    x = B.system.x
    u = TrialFunction(B)
    ue = (1 - x**2) ** 2 * (x**3 + x + 1)
    if bcs is BIHARMONIC:
        f = ue.diff(x, 4) - 3 * ue.diff(x, 2) + 5 * ue
    else:
        f = ue.diff(x, 2) - 5 * ue
    sols, conds = [], []
    for V in (B, B.get_testspace("GR")):
        v = TestFunction(V)
        if bcs is BIHARMONIC:
            form = u.diff(x, 4) * v - 3 * u.diff(x, 2) * v + 5 * u * v
        else:
            form = u.diff(x, 2) * v - 5 * u * v
        A = np.asarray(inner(form, kind="bilinear").todense())
        sols.append(np.linalg.solve(A, np.asarray(inner(f * v))))
        conds.append(np.linalg.cond(A.astype(float)))
    # Both are the Galerkin solution; they differ only by each solve's round-off.
    tol = _tol() * max(conds)
    assert np.abs(sols[0] - sols[1]).max() < tol * np.abs(sols[0]).max()


def test_stage_operator_takes_the_banded_wavenumber_solver():
    B = FunctionSpace(20, Chebyshev, bcs=BIHARMONIC)
    F0, F1 = FunctionSpace(6, Fourier), FunctionSpace(6, Fourier)
    VB = TensorProduct(F0, F1, B)
    PB = TensorProduct(F0, F1, B.get_testspace("GR"))
    x, y, z = VB.system.base_scalars()
    u, v = TrialFunction(VB), TestFunction(PB)
    lap = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
    A = inner((lap - 0.01 * lap.diff(x, 2) - 0.01 * lap.diff(z, 2)) * v, sparse=True)
    assert isinstance(A, TPMatrices)
    assert isinstance(A.lu_factor(), TPMatricesWavenumberSolver)
    dense = np.asarray(A.todense())
    ref = np.linalg.solve(dense, np.ones(dense.shape[0])).reshape(VB.num_dofs)
    got = np.asarray(A.solve(jnp.ones(VB.num_dofs)))
    assert np.abs(got - ref).max() < 10 * _tol() * np.abs(ref).max()


@pytest.mark.parametrize("family", [Chebyshev, Legendre])
def test_inhomogeneous_dirichlet_matches_galerkin(family):
    """The boundary lifting, a trial space narrower than the test space, too."""
    N = 20
    D = FunctionSpace(N, family, bcs={"left": {"D": 1}, "right": {"D": -2}})
    x = D.system.x
    u = TrialFunction(D)
    ue = (x**5 - x**2 + 3 * x) * (1 + x / 4) / 2 - x / 2 - 1
    f = ue.diff(x, 2) - 4 * ue
    sols = []
    for V in (D, D.get_testspace("GR")):
        v = TestFunction(V)
        A, b = inner((u.diff(x, 2) - 4 * u) * v - f * v, kind="system")
        sols.append(np.linalg.solve(np.asarray(A.todense()), np.asarray(b)))
    cond = np.linalg.cond(np.asarray(A.todense()).astype(float))
    assert np.abs(sols[0] - sols[1]).max() < _tol() * cond * np.abs(sols[0]).max()
