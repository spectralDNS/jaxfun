import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import (
    BlockTPMatrix,
    TPLUFactors,
    TPMatrices,
    TPMatricesDenseLUFactors,
    TPMatricesLUFactors,
    TPMatricesWavenumberSolver,
    TPMatrix,
    tpmats_dense_lu_factor,
    tpmats_lu_factor,
    tpmats_to_kron,
    tpmats_wavenumber_factor,
)
from jaxfun.la.diamatrix import DiaMatrix
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import lambdify, ulp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BCS = {"left": {"D": 0}, "right": {"D": 0}}

POLY_SPACES = pytest.mark.parametrize(
    "poly",
    [Legendre.Legendre, Chebyshev.Chebyshev],
    ids=["legendre", "chebyshev"],
)


def _poisson_poly2d(N: int, poly, sparse: bool = True):
    """Return (T, A, b, ue) for poly x poly Poisson with manufactured solution."""
    F0 = FunctionSpace(N, poly, BCS)
    F1 = FunctionSpace(N, poly, BCS)
    T = TensorProduct(F0, F1)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = (1 - x**2) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=sparse)
    return T, A, b, ue


def _poisson_fourier_poly_2d(N: int, poly, sparse: bool = True):
    """Return (T, A, b, ue) for Fourier x poly Poisson with manufactured solution."""
    F = FunctionSpace(N, Fourier)
    D = FunctionSpace(N, poly, BCS)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=sparse)
    return T, A, b, ue


# ---------------------------------------------------------------------------
# tpmats_lu_factor (diagonalization path)
# ---------------------------------------------------------------------------


def test_tpmats_lu_factor_returns_correct_type():
    _, A, b, _ = _poisson_poly2d(8, Legendre.Legendre)
    lu = tpmats_lu_factor(A)
    assert isinstance(lu, TPMatricesLUFactors)


@POLY_SPACES
def test_tpmats_lu_factor_solve_poly2d(poly):
    T, A, b, ue = _poisson_poly2d(16, poly)
    lu = tpmats_lu_factor(A)
    assert isinstance(lu, TPMatricesLUFactors)
    uh = lu.solve(b)
    assert uh.shape == b.shape
    x, y = T.system.base_scalars()
    N = 40
    uj = T.backward(uh, N=(N, N))
    xj = T.mesh(N=(N, N), broadcast=True)
    uej = lambdify((x, y), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / N
    assert l2 < float(ulp(100)), f"L2 error {l2:.2e} too large"


def test_tpmats_lu_factor_accepts_tpmatrix_single():
    """tpmats_lu_factor accepts a single TPMatrix as well as a list."""
    _, A, b, _ = _poisson_poly2d(8, Legendre.Legendre)
    lu = tpmats_lu_factor([A[0]])
    assert isinstance(lu, TPMatricesLUFactors)


# ---------------------------------------------------------------------------
# tpmats_wavenumber_factor
# ---------------------------------------------------------------------------


def test_tpmats_wavenumber_factor_returns_correct_type():
    _, A, _, _ = _poisson_fourier_poly_2d(8, Legendre.Legendre)
    wn = tpmats_wavenumber_factor(A)
    assert isinstance(wn, TPMatricesWavenumberSolver)


@POLY_SPACES
def test_tpmats_wavenumber_factor_solve_agrees_with_kron(poly):
    _, A, b, _ = _poisson_fourier_poly_2d(16, poly)
    wn = tpmats_wavenumber_factor(A)
    ref = tpmats_to_kron(A).solve(b.flatten()).reshape(b.shape)
    uh = wn.solve(b)
    assert uh.shape == b.shape
    assert float(jnp.max(jnp.abs(uh - ref))) < ulp(10)


@POLY_SPACES
def test_tpmats_wavenumber_factor_solve_fourier_poly_l2(poly):
    T, A, b, ue = _poisson_fourier_poly_2d(16, poly)
    wn = tpmats_wavenumber_factor(A)
    uh = wn.solve(b)
    x, y = T.system.base_scalars()
    N = 40
    uj = T.backward(uh, N=(N, N))
    xj = T.mesh(N=(N, N), broadcast=True)
    uej = lambdify((x, y), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / N
    assert l2 < jnp.sqrt(ulp(10)), f"L2 error {l2:.2e} too large"


def test_tpmats_wavenumber_factor_accepts_tpmatrices():
    """tpmats_wavenumber_factor accepts a TPMatrices object as well as a list."""
    _, A, b, _ = _poisson_fourier_poly_2d(8, Legendre.Legendre)
    wn = tpmats_wavenumber_factor(TPMatrices(A))
    assert isinstance(wn, TPMatricesWavenumberSolver)
    uh = wn.solve(b)
    assert uh.shape == b.shape


def test_tpmats_wavenumber_factor_type_error():
    with pytest.raises(TypeError):
        tpmats_wavenumber_factor("not a valid input")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TPMatricesWavenumberSolver.solve2
# ---------------------------------------------------------------------------


def test_wavenumber_solver_solve2_agrees_with_solve():
    _, A, b, _ = _poisson_fourier_poly_2d(16, Legendre.Legendre)
    wn = tpmats_wavenumber_factor(A)
    assert float(jnp.max(jnp.abs(wn.solve(b) - wn.solve2(b)))) < float(ulp(100))


# ---------------------------------------------------------------------------
# tpmats_dense_lu_factor / TPMatricesDenseLUFactors
# ---------------------------------------------------------------------------


@POLY_SPACES
def test_tpmats_dense_lu_factor_returns_correct_type(poly):
    _, A, _, _ = _poisson_poly2d(8, poly, sparse=False)
    lu = tpmats_dense_lu_factor(A)
    assert isinstance(lu, TPMatricesDenseLUFactors)


@POLY_SPACES
def test_tpmats_dense_lu_factor_solve_poly2d(poly):
    T, A, b, ue = _poisson_poly2d(16, poly, sparse=False)
    lu = tpmats_dense_lu_factor(A)
    uh = lu.solve(b)
    assert uh.shape == b.shape
    x, y = T.system.base_scalars()
    N = 40
    uj = T.backward(uh, N=(N, N))
    xj = T.mesh(N=(N, N), broadcast=True)
    uej = lambdify((x, y), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / N
    assert l2 < float(ulp(100)), f"L2 error {l2:.2e} too large"


@POLY_SPACES
def test_tpmatrices_solve_dispatches_dense_for_matrix(poly):
    """TPMatrices.lu_factor() dispatches to TPMatricesDenseLUFactors for dense matrices."""  # noqa: E501
    _, A, b, _ = _poisson_poly2d(12, poly, sparse=False)
    mats = TPMatrices(A)
    lu = mats.lu_factor()
    assert isinstance(lu, TPMatricesDenseLUFactors)
    uh = mats.solve(b)
    assert uh.shape == b.shape


@POLY_SPACES
def test_tpmats_dense_agrees_with_sparse(poly):
    """Dense and sparse solvers produce the same solution."""
    _, A_sp, b_sp, _ = _poisson_poly2d(16, poly, sparse=True)
    _, A_de, b_de, _ = _poisson_poly2d(16, poly, sparse=False)
    uh_sp = tpmats_lu_factor(A_sp).solve(b_sp)
    uh_de = tpmats_dense_lu_factor(A_de).solve(b_de)
    assert float(jnp.max(jnp.abs(uh_sp - uh_de))) < float(ulp(100))


def test_tpmats_dense_lu_factor_type_error():
    _, A, _, _ = _poisson_poly2d(8, Legendre.Legendre, sparse=True)
    with pytest.raises(TypeError):
        tpmats_dense_lu_factor(A)


# ---------------------------------------------------------------------------
# TPMatrix.solve / TPLUFactors  (single Kronecker-product term)
# ---------------------------------------------------------------------------


@POLY_SPACES
def test_tpmatrix_lu_factor_returns_tplufactors(poly):
    """TPMatrix.lu_factor() returns a TPLUFactors instance."""
    _, A, _, _ = _poisson_poly2d(8, poly)
    assert isinstance(A[0], TPMatrix)
    lu = A[0].lu_factor()
    assert isinstance(lu, TPLUFactors)


@POLY_SPACES
def test_tpmatrix_solve_single_term(poly):
    """TPMatrix.solve solves a single-term Kronecker system correctly."""
    T, A, b, ue = _poisson_poly2d(16, poly)
    # Use the first (and for a pure Laplacian, dominant) term directly
    tp = A[0]
    rhs = tp(tp.lu_factor().solve(b))  # round-trip: A*(A^{-1}*b) ≈ b
    assert float(jnp.max(jnp.abs(rhs - b))) < float(ulp(100))


# ---------------------------------------------------------------------------
# TPMatrices.solve auto-dispatch
# ---------------------------------------------------------------------------


def test_tpmatrices_solve_dispatches_wavenumber_for_fourier():
    """TPMatrices.solve should dispatch to TPMatricesWavenumberSolver for Fourier x poly."""  # noqa: E501
    _, A, b, _ = _poisson_fourier_poly_2d(12, Legendre.Legendre)
    mats = TPMatrices(A)
    lu = mats.lu_factor()
    assert isinstance(lu, TPMatricesWavenumberSolver)
    uh = mats.solve(b)
    assert uh.shape == b.shape


@POLY_SPACES
def test_tpmatrices_solve_dispatches_lu_for_poly(poly):
    """TPMatrices.solve should dispatch to TPMatricesLUFactors for all-polynomial."""
    _, A, b, _ = _poisson_poly2d(12, poly)
    mats = TPMatrices(A)
    lu = mats.lu_factor()
    assert isinstance(lu, TPMatricesLUFactors)
    uh = mats.solve(b)
    assert uh.shape == b.shape


def test_tpmatrices_lu_factor_caching():
    """lu_factor called twice returns the same cached object."""
    _, A, _, _ = _poisson_fourier_poly_2d(8, Legendre.Legendre)
    mats = TPMatrices(A)
    lu1 = mats.lu_factor()
    lu2 = mats.lu_factor()
    assert lu1 is lu2


@POLY_SPACES
def test_tpmatrices_solve_poly2d_l2(poly):
    T, A, b, ue = _poisson_poly2d(16, poly)
    uh = TPMatrices(A).solve(b)
    assert uh.shape == b.shape
    x, y = T.system.base_scalars()
    N = 40
    uj = T.backward(uh, N=(N, N))
    xj = T.mesh(N=(N, N), broadcast=True)
    uej = lambdify((x, y), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / N
    assert l2 < float(ulp(100)), f"L2 error {l2:.2e} too large"


@POLY_SPACES
def test_tpmatrices_solve_fourier_poly2d_l2(poly):
    T, A, b, ue = _poisson_fourier_poly_2d(16, poly)
    uh = TPMatrices(A).solve(b)
    assert uh.shape == b.shape
    x, y = T.system.base_scalars()
    N = 40
    uj = T.backward(uh, N=(N, N))
    xj = T.mesh(N=(N, N), broadcast=True)
    uej = lambdify((x, y), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / N
    assert l2 < jnp.sqrt(ulp(10)), f"L2 error {l2:.2e} too large"


# ---------------------------------------------------------------------------
# 3D: Fourier x Fourier x Legendre
# ---------------------------------------------------------------------------


def test_tpmatrices_solve_fourier_fourier_legendre_3d():
    """3D wavenumber solver: Fourier x Fourier x Legendre Poisson."""
    N = 8
    F0 = FunctionSpace(N, Fourier)
    F1 = FunctionSpace(N, Fourier)
    D = FunctionSpace(N, Legendre.Legendre, BCS)
    T = TensorProduct(F0, F1, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y, z = T.system.base_scalars()
    ue = sp.cos(2 * x) * sp.cos(2 * y) * (1 - z**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)

    mats = TPMatrices(A)
    lu = mats.lu_factor()
    assert isinstance(lu, TPMatricesWavenumberSolver)

    uh = mats.solve(b)
    assert uh.shape == b.shape

    M = 20
    uj = T.backward(uh, N=(M, M, M))
    xj = T.mesh(N=(M, M, M), broadcast=True)
    uej = lambdify((x, y, z), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / M
    assert l2 < jnp.sqrt(ulp(1000)), f"3D L2 error {l2:.2e} too large"


# ---------------------------------------------------------------------------
# BlockTPMatrix
# ---------------------------------------------------------------------------

BCS_VEC = {"left": {"D": 0}, "right": {"D": 0}}


def _vector_block_system(N: int, poly):
    """Two-component vector mass system in 2D.

    Uses inner(Dot(v, u)) to assemble a block-diagonal mass matrix
    (two decoupled identical blocks).  Returns (A, b, x_true) where b is
    a manufactured RHS consistent with the dense system.
    """
    F0 = FunctionSpace(N, poly, BCS_VEC)
    F1 = FunctionSpace(N, poly, BCS_VEC)
    T = TensorProduct(F0, F1)
    V = VectorTensorProductSpace(T, name="V")
    u = TrialFunction(V)
    v = TestFunction(V)
    M = inner(Dot(v, u), sparse=True)
    A = BlockTPMatrix(M, V, V)
    rng = np.random.default_rng(0)
    x_true = jnp.array(rng.standard_normal(A.shape[1]), dtype=jnp.float32)
    b = A.to_Matrix().matvec(x_true)
    return A, b, x_true


@POLY_SPACES
def test_blocktpmatrix_tosparse_returns_diamatrix(poly):
    A, b, _ = _vector_block_system(8, poly)
    sparse = A.tosparse()
    assert isinstance(sparse, DiaMatrix)
    assert sparse.shape == A.shape


@POLY_SPACES
def test_blocktpmatrix_solve_sparse_matches_dense(poly):
    A, b, x_true = _vector_block_system(8, poly)
    # Dense reference
    x_dense = A.to_Matrix().solve(b.ravel()).reshape(b.shape)
    # Sparse / RCM path
    x_sparse = A.solve(b)
    assert x_sparse.shape == b.shape
    assert jnp.allclose(x_sparse.ravel(), x_dense.ravel(), atol=ulp(1000))


@POLY_SPACES
def test_blocktpmatrix_rcm_reduces_bandwidth(poly):
    A, _, _ = _vector_block_system(8, poly)
    sparse = A.tosparse()
    A_perm, _, _ = sparse.rcm()
    bw_before = max(abs(k) for k in sparse.offsets)
    bw_after = max(abs(k) for k in A_perm.offsets)
    assert bw_after <= bw_before


@POLY_SPACES
def test_blocktpmatrix_call_matches_dense_matvec(poly):
    A, b, x_true = _vector_block_system(8, poly)
    # Warm the RCM cache via solve
    _ = A.solve(b)
    y_block = A(x_true)
    y_dense = A.to_Matrix().matvec(x_true.ravel()).reshape(x_true.shape)
    assert jnp.allclose(y_block.ravel(), y_dense.ravel(), atol=ulp(1000))


@POLY_SPACES
def test_blocktpmatrix_solve_cached_rcm(poly):
    """Second solve reuses cached RCM without reassembly."""
    A, b, _ = _vector_block_system(8, poly)
    x1 = A.solve(b)
    x2 = A.solve(b)
    assert jnp.allclose(x1.ravel(), x2.ravel(), atol=ulp(10))
