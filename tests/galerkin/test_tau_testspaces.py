"""`get_testspace("tau", tau_basis=...)`: the test space of a tau method.

The tau method with tau polynomials X_1, ..., X_m is testing with the
polynomials of degree less than N that are orthogonal to all of them.
"""

import functools
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from numpy.polynomial import chebyshev as npcheb

import jaxfun.typing as jt
from jaxfun.galerkin import FunctionSpace, TestFunction, TrialFunction, inner
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.composite import Composite, TauRecombined
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.Ultraspherical import Ultraspherical
from jaxfun.utils.common import ulp

DIRICHLET = {"left": {"D": 0}, "right": {"D": 0}}
NEUMANN = {"left": {"N": 0}, "right": {"N": 0}}
BIHARMONIC = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
THREE = {"left": {"D": 0, "N": 0}, "right": {"D": 0}}
METHODS = ("tau-T", "tau-C2", "tau-U", "tau-U2", "tau-U3")


def _tol() -> float:
    return 1e3 * float(jnp.finfo(jnp.result_type(float)).eps)


@functools.cache
def _families(N: int) -> tuple[Chebyshev, ChebyshevU, Ultraspherical]:
    return Chebyshev(N), ChebyshevU(N), Ultraspherical(N, lambda_=2)


def _tau_basis(method: str, N: int) -> list:
    """Return the `tau_basis` of the Chebyshev tau method `method`."""
    T, U, C2 = _families(N)
    return {
        "tau-T": [(T, -1), (T, -2)],
        "tau-C2": [(C2, -1), (C2, -2)],
        "tau-U": [(U, -1), (C2, -2)],  # U'_{N-1} = 2 C2_{N-2}
        "tau-U2": [(U, -1), (U, -2)],
        "tau-U3": [(U, -2), (C2, -1)],
    }[method]


def _T(n: int, N: int) -> np.ndarray:
    return np.eye(N)[n]


def _U(n: int, N: int) -> np.ndarray:  # T'_{n+1} = (n + 1) U_n
    return np.pad(npcheb.chebder(np.eye(n + 2)[n + 1]) / (n + 1), (0, N - n - 1))


def _C2(n: int, N: int) -> np.ndarray:  # C2_n = U'_{n+1} / 2
    return np.pad(npcheb.chebder(_U(n + 1, n + 2)) / 2, (0, N - n - 1))


def _tau_polynomials(method: str, N: int) -> np.ndarray:
    """Return the Chebyshev coefficients of the tau polynomials of `method`."""
    X = {
        "tau-T": (_T(N - 1, N), _T(N - 2, N)),
        "tau-C2": (_C2(N - 1, N), _C2(N - 2, N)),
        "tau-U": (_U(N - 1, N), _C2(N - 2, N)),
        "tau-U2": (_U(N - 1, N), _U(N - 2, N)),
        "tau-U3": (_U(N - 2, N), _C2(N - 1, N)),
    }[method]
    return np.array(X)


def _assert_orthogonal(S: np.ndarray, X: np.ndarray) -> None:
    """Assert that the rows of `S` are independent and Chebyshev-orthogonal to `X`."""
    N = S.shape[1]
    rows = S / np.abs(S).max(axis=1, keepdims=True)
    assert np.linalg.matrix_rank(rows) == len(S) == N - len(X)
    h = np.full(N, np.pi / 2)
    h[0] = np.pi
    X = X / np.abs(X).max(axis=1, keepdims=True)
    assert np.abs((rows * h) @ X.T).max() < _tol()


def _exact(test_rows: np.ndarray, trial_rows: np.ndarray, j: int, q: int):
    """Return (x^q trial^{(j)}, test)_w from the Chebyshev coefficients."""
    N = test_rows.shape[1]
    h = np.full(N, np.pi / 2)
    h[0] = np.pi
    C = []
    for c in trial_rows:
        c = npcheb.chebder(c, j)
        for _ in range(q):
            c = npcheb.chebmulx(c)
        C.append(np.pad(c, (0, max(N - len(c), 0)))[:N])
    return (test_rows * h) @ np.array(C).T


def _offsets(A) -> set[int]:
    X = np.asarray(A.todense())
    m = np.abs(X).max(axis=1, keepdims=True)
    nz = np.argwhere(np.abs(X / np.where(m > 0, m, 1)) > 1e-6)
    return set((nz[:, 1] - nz[:, 0]).tolist())


def test_coerce():
    assert jt.TestSpaceKind.coerce("tau") is jt.TestSpaceKind.TAU


@pytest.mark.parametrize("N", [12, 13])
@pytest.mark.parametrize("bcs", [DIRICHLET, NEUMANN], ids=["Dirichlet", "Neumann"])
@pytest.mark.parametrize("method", METHODS)
def test_orthogonal_to_tau_polynomials(method, bcs, N):
    B = FunctionSpace(N, Chebyshev, bcs=bcs)
    V = B.get_testspace("tau", tau_basis=_tau_basis(method, N))
    assert isinstance(V, TauRecombined)
    _assert_orthogonal(np.asarray(V.S.todense()), _tau_polynomials(method, N))


def test_petrov_galerkin_and_galerkin_recombined():
    N = 12
    B = FunctionSpace(N, Chebyshev, bcs=DIRICHLET)

    def stencil(V):
        return np.asarray(V.S.todense())

    tau_C2 = B.get_testspace("tau", tau_basis=_tau_basis("tau-C2", N))
    assert np.array_equal(stencil(tau_C2), stencil(B.get_testspace("PG")))
    tau_U2 = B.get_testspace("tau", tau_basis=_tau_basis("tau-U2", N))
    assert np.allclose(stencil(tau_U2), stencil(B.get_testspace("GR")))


@pytest.mark.parametrize("j,q", [(0, 0), (2, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
@pytest.mark.parametrize("method", ["tau-T", "tau-U", "tau-U2", "tau-U3"])
def test_matrices(method, j, q):
    N = 12
    B = cast(Composite, FunctionSpace(N, Chebyshev, bcs=DIRICHLET))
    V = B.get_testspace("tau", tau_basis=_tau_basis(method, N))
    x = B.system.x
    u = TrialFunction(B)
    form = x**q * u.diff(x, j) if j else x**q * u
    A = inner(form * TestFunction(V), sparse=True, kind="bilinear")
    ref = _exact(np.asarray(V.S.todense()), np.asarray(B.S.todense()), j, q)
    err = np.abs(np.asarray(A.todense()) - ref).max(axis=1)
    assert np.all(err <= _tol() * np.abs(ref).max(axis=1))
    pg = inner(form * TestFunction(B.get_testspace("PG")), sparse=True)
    assert _offsets(A) <= _offsets(pg)


@pytest.mark.parametrize("method", METHODS)
def test_solution_is_the_tau_solution(method):
    N, a, b = 12, 1.0, -2.0
    D = FunctionSpace(N, Chebyshev, bcs={"left": {"D": a}, "right": {"D": b}})
    x = D.system.x
    # f of degree N - 1, so that it is its own projection f_N.
    fh = 1.0 / (1 + np.arange(N))
    f = sum(sp.Rational(1, n + 1) * sp.chebyshevt(n, x) for n in range(N))
    V = D.get_testspace("tau", tau_basis=_tau_basis(method, N))
    u, v = TrialFunction(D), TestFunction(V)
    A, L = inner((u.diff(x, 2) + u) * v - f * v, kind="system")
    uh = np.asarray(D.to_orthogonal(A.solve(L)))
    # u'' + u = f + tau_a X_a + tau_b X_b with the boundary conditions, for the
    # Chebyshev coefficients of u and the two taus.
    M = np.zeros((N + 2, N + 2))
    M[:N, :N] = np.eye(N)
    M[:N, :N] += np.array([np.pad(npcheb.chebder(e, 2), (0, 2)) for e in np.eye(N)]).T
    M[:N, N:] = -_tau_polynomials(method, N).T
    M[N] = np.r_[(-1.0) ** np.arange(N), 0, 0]
    M[N + 1] = np.r_[np.ones(N), 0, 0]
    ref = np.linalg.solve(M, np.r_[fh, a, b])[:N]
    assert np.abs(uh - ref).max() < ulp(100) * np.abs(ref).max()


def test_biharmonic_lanczos():
    """Four conditions: tau_basis T_{N-1}, ..., T_{N-4} is Lanczos' method."""
    N = 14
    B = FunctionSpace(N, Chebyshev, bcs=BIHARMONIC)
    T = _families(N)[0]
    V = B.get_testspace("tau", tau_basis=[(T, -1), (T, -2), (T, -3), (T, -4)])
    _assert_orthogonal(np.asarray(V.S.todense()), np.eye(N)[N - 4 :])
    x = B.system.x
    fh = 1.0 / (1 + np.arange(N))
    f = sum(sp.Rational(1, n + 1) * sp.chebyshevt(n, x) for n in range(N))
    u, v = TrialFunction(B), TestFunction(V)
    A = inner((u.diff(x, 4) + u) * v, kind="bilinear")
    uh = np.asarray(B.to_orthogonal(A.solve(inner(f * v, kind="linear"))))
    # The Chebyshev coefficients 0, ..., N - 5 of u'''' + u = f, and the four
    # boundary conditions in place of the last four.
    n = np.arange(N)
    M = (
        np.eye(N)
        + np.array([np.pad(npcheb.chebder(e, 4), (0, 4)) for e in np.eye(N)]).T
    )
    M[N - 4 :] = [(-1.0) ** n, np.ones(N), (-1.0) ** (n + 1) * n**2, n**2]
    ref = np.linalg.solve(M, np.r_[fh[: N - 4], 0, 0, 0, 0])
    assert np.abs(uh - ref).max() < ulp(1000) * np.abs(ref).max()


@pytest.mark.parametrize(
    "family,lam", [(Chebyshev, 3), (Legendre, sp.Rational(7, 2))], ids=["C", "L"]
)
def test_three_conditions_petrov_galerkin(family, lam):
    """The order-3 PG functions read off the coefficients of C^(lam), so PG is
    the tau method with C^(lam)_{N-1}, C^(lam)_{N-2} and C^(lam)_{N-3}."""
    N = 12
    B = FunctionSpace(N, family, bcs=THREE)
    C = Ultraspherical(N, lambda_=lam)
    V = B.get_testspace("tau", tau_basis=[(C, -1), (C, -2), (C, -3)])
    pg = B.get_testspace("PG")
    S, P = np.asarray(V.S.todense()), np.asarray(pg.S.todense())
    rows = np.vstack([S, P])
    rows = rows / np.abs(rows).max(axis=1, keepdims=True)
    assert np.linalg.matrix_rank(rows, tol=_tol()) == len(S) == len(P)
    if family is Chebyshev:  # recombined from the PG functions, row for row
        assert np.array_equal(S, P)


def test_three_conditions_lanczos():
    N = 12
    B = FunctionSpace(N, Chebyshev, bcs=THREE)
    T = _families(N)[0]
    V = B.get_testspace("tau", tau_basis=[(T, -1), (T, -2), (T, -3)])
    _assert_orthogonal(np.asarray(V.S.todense()), np.eye(N)[N - 3 :])


def test_tau_polynomials_of_the_same_degree():
    N = 14
    B = FunctionSpace(N, Chebyshev, bcs=BIHARMONIC)
    T, U, _ = _families(N)
    tau_basis = [(T, -1), (U, -1), (T, -2), (U, -2)]
    V = B.get_testspace("tau", tau_basis=tau_basis)
    X = np.array([_T(N - 1, N), _U(N - 1, N), _T(N - 2, N), _U(N - 2, N)])
    _assert_orthogonal(np.asarray(V.S.todense()), X)


def test_legendre():
    """Legendre polynomials as tau basis: the test space is P_{N-3}."""
    N = 12
    B = FunctionSpace(N, Legendre, bcs=DIRICHLET)
    P = Legendre(N)
    S = np.asarray(B.get_testspace("tau", tau_basis=[(P, -1), (P, -2)]).S.todense())
    rows = S / np.abs(S).max(axis=1, keepdims=True)
    assert np.linalg.matrix_rank(rows) == N - 2
    assert np.abs(rows[:, N - 2 :]).max() < _tol()


@pytest.mark.parametrize("bcs", [DIRICHLET, BIHARMONIC], ids=["2", "4"])
def test_legendre_has_galerkin_sparsity(bcs):
    """Recombined from the Galerkin functions, Lanczos' method for Legendre has
    the matrices of the Galerkin method in band, and correct to quadrature."""
    N = 12
    B = FunctionSpace(N, Legendre, bcs=bcs)
    V = B.get_testspace("tau")
    x = B.system.x
    u = TrialFunction(B)
    for form in (u.diff(x, B.bcs.num_bcs()), u):
        A = inner(form * TestFunction(V), kind="bilinear")
        ref = inner(
            form * TestFunction(V), kind="bilinear", use_precomputed_matrices=False
        )
        assert _offsets(A) == _offsets(inner(form * TestFunction(B), kind="bilinear"))
        err = np.abs(np.asarray(A.todense()) - np.asarray(ref.todense())).max()
        assert err < _tol() * np.abs(np.asarray(ref.todense())).max()


@pytest.mark.parametrize("family", [Chebyshev, Legendre])
@pytest.mark.parametrize("bcs", [DIRICHLET, THREE, BIHARMONIC], ids=["2", "3", "4"])
def test_default_is_lanczos(family, bcs):
    """Without `tau_basis` the tau polynomials are the last orthogonal modes."""
    N = 12
    B = FunctionSpace(N, family, bcs=bcs)
    m = B.bcs.num_bcs()
    S = np.asarray(B.get_testspace("tau").S.todense())
    P = family(N)
    lanczos = B.get_testspace("tau", tau_basis=[(P, -k) for k in range(1, m + 1)])
    assert np.array_equal(S, np.asarray(lanczos.S.todense()))
    # Testing with the polynomials of degree less than N - m.
    rows = S / np.abs(S).max(axis=1, keepdims=True)
    assert np.linalg.matrix_rank(rows) == N - m
    assert np.abs(rows[:, N - m :]).max() < _tol()


def test_errors():
    N = 12
    B = FunctionSpace(N, Chebyshev, bcs=DIRICHLET)
    T = Chebyshev(N)
    with pytest.raises(ValueError, match="one per condition"):
        B.get_testspace("tau", tau_basis=[(T, -1)])
    with pytest.raises(ValueError, match="undetermined"):
        B.get_testspace("tau", tau_basis=[(T, -1), (T, -3)])
    with pytest.raises(ValueError, match="degree less than"):
        B.get_testspace("tau", tau_basis=[(Chebyshev(N + 2), -1), (T, -2)])
    with pytest.raises(NotImplementedError, match="boundary conditions"):
        T.get_testspace("tau")
