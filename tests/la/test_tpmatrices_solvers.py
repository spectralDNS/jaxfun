from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jax import Array

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
from jaxfun.la import (
    BaseMatrix,
    BlockArray,
    BlockMatrix,
    TPMatrices,
    TPMatrix,
    tpmats_to_kron,
)
from jaxfun.la.diamatrix import _PREFIX_BAND_SLACK, DiaMatrix
from jaxfun.la.tpmatrix import (
    TPLUFactors,
    TPMatricesDenseLUFactors,
    TPMatricesLUFactors,
    TPMatricesWavenumberSolver,
    _parity_halves_offsets,
    tpmats_dense_lu_factor,
    tpmats_lu_factor,
    tpmats_wavenumber_factor,
)
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import lambdify, ulp

pytestmark = pytest.mark.integration

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
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=sparse, kind="system")
    return T, cast(TPMatrices, A), b, ue


def _poisson_fourier_poly_2d(N: int, poly, sparse: bool = True):
    """Return (T, A, b, ue) for Fourier x poly Poisson with manufactured solution."""
    F = FunctionSpace(N, Fourier)
    D = FunctionSpace(N, poly, BCS)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=sparse, kind="system")
    return T, cast(TPMatrices, A), b, ue


# ---------------------------------------------------------------------------
# tpmats_lu_factor (diagonalization path)
# ---------------------------------------------------------------------------


def test_tpmats_lu_factor_returns_correct_type():
    _, A, b, _ = _poisson_poly2d(8, Legendre.Legendre)
    lu = tpmats_lu_factor(A.tpmats)
    assert isinstance(lu, TPMatricesLUFactors)


@POLY_SPACES
def test_tpmats_lu_factor_solve_poly2d(poly):
    T, A, b, ue = _poisson_poly2d(16, poly)
    lu = tpmats_lu_factor(A.tpmats)
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
    lu = tpmats_lu_factor(A.tpmats[0])
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
    ref = tpmats_to_kron(A.tpmats).solve(b.flatten()).reshape(b.shape)
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
    wn = tpmats_wavenumber_factor(A)
    assert isinstance(wn, TPMatricesWavenumberSolver)
    uh = wn.solve(b)
    assert uh.shape == b.shape


def test_tpmats_wavenumber_factor_type_error():
    with pytest.raises(TypeError):
        tpmats_wavenumber_factor("not a valid input")  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# tpmats_dense_lu_factor / TPMatricesDenseLUFactors
# ---------------------------------------------------------------------------


@POLY_SPACES
def test_tpmats_dense_lu_factor_returns_correct_type(poly):
    _, A, _, _ = _poisson_poly2d(8, poly, sparse=False)
    lu = tpmats_dense_lu_factor(A.tpmats)
    assert isinstance(lu, TPMatricesDenseLUFactors)


@POLY_SPACES
def test_tpmats_dense_lu_factor_solve_poly2d(poly):
    T, A, b, ue = _poisson_poly2d(16, poly, sparse=False)
    lu = tpmats_dense_lu_factor(A.tpmats)
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
    lu = A.lu_factor()
    assert isinstance(lu, TPMatricesDenseLUFactors)
    uh = A.solve(b)
    assert uh.shape == b.shape


@POLY_SPACES
def test_tpmats_dense_agrees_with_sparse(poly):
    """Dense and sparse solvers produce the same solution."""
    _, A_sp, b_sp, _ = _poisson_poly2d(16, poly, sparse=True)
    _, A_de, b_de, _ = _poisson_poly2d(16, poly, sparse=False)
    uh_sp = tpmats_lu_factor(A_sp.tpmats).solve(b_sp)
    uh_de = tpmats_dense_lu_factor(A_de.tpmats).solve(b_de)
    assert float(jnp.max(jnp.abs(uh_sp - uh_de))) < float(ulp(100))


def test_tpmats_dense_lu_factor_type_error():
    _, A, _, _ = _poisson_poly2d(8, Legendre.Legendre, sparse=True)
    with pytest.raises(TypeError):
        tpmats_dense_lu_factor(A.tpmats)


# ---------------------------------------------------------------------------
# TPMatrix.solve / TPLUFactors  (single Kronecker-product term)
# ---------------------------------------------------------------------------


@POLY_SPACES
def test_tpmatrix_lu_factor_returns_tplufactors(poly):
    """TPMatrix.lu_factor() returns a TPLUFactors instance."""
    _, A, _, _ = _poisson_poly2d(8, poly)
    assert isinstance(A.tpmats[0], TPMatrix)
    lu = A.tpmats[0].lu_factor()
    assert isinstance(lu, TPLUFactors)


@POLY_SPACES
def test_tpmatrix_solve_single_term(poly):
    """TPMatrix.solve solves a single-term Kronecker system correctly."""
    T, A, b, ue = _poisson_poly2d(16, poly)
    # Use the first (and for a pure Laplacian, dominant) term directly
    tp = A.tpmats[0]
    rhs = tp(tp.lu_factor().solve(b))  # round-trip: A*(A^{-1}*b) ≈ b
    assert float(jnp.max(jnp.abs(rhs - b))) < float(ulp(100))


# ---------------------------------------------------------------------------
# TPMatrices.solve auto-dispatch
# ---------------------------------------------------------------------------


def test_tpmatrices_solve_dispatches_wavenumber_for_fourier():
    """TPMatrices.solve should dispatch to TPMatricesWavenumberSolver for Fourier x poly."""  # noqa: E501
    _, A, b, _ = _poisson_fourier_poly_2d(12, Legendre.Legendre)
    lu = A.lu_factor()
    assert isinstance(lu, TPMatricesWavenumberSolver)
    uh = A.solve(b)
    assert uh.shape == b.shape


@POLY_SPACES
def test_tpmatrices_solve_dispatches_lu_for_poly(poly):
    """TPMatrices.solve should dispatch to TPMatricesLUFactors for all-polynomial."""
    _, A, b, _ = _poisson_poly2d(12, poly)
    lu = A.lu_factor()
    assert isinstance(lu, TPMatricesLUFactors)
    uh = A.solve(b)
    assert uh.shape == b.shape


def test_tpmatrices_lu_factor_caching():
    """lu_factor called twice returns the same cached object."""
    _, A, _, _ = _poisson_fourier_poly_2d(8, Legendre.Legendre)
    lu1 = A.lu_factor()
    lu2 = A.lu_factor()
    assert lu1 is lu2


@POLY_SPACES
def test_tpmatrices_solve_poly2d_l2(poly):
    T, A, b, ue = _poisson_poly2d(16, poly)
    uh = A.solve(b)
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
    uh = A.solve(b)
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
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True, kind="system")

    lu = A.lu_factor()
    assert isinstance(lu, TPMatricesWavenumberSolver)

    uh = A.solve(b)
    assert uh.shape == cast(Array, b).shape

    M = 20
    uj = T.backward(uh, N=(M, M, M))
    xj = T.mesh(N=(M, M, M), broadcast=True)
    uej = lambdify((x, y, z), ue)(*xj)
    l2 = float(jnp.linalg.norm(uj - uej)) / M
    assert l2 < jnp.sqrt(ulp(1000)), f"3D L2 error {l2:.2e} too large"


# ---------------------------------------------------------------------------
# BlockMatrix
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
    V = CartesianProduct(T, T, name="V", rank=1)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = inner(Dot(v, u), sparse=True)
    assert isinstance(A, BlockMatrix)
    assert isinstance(A, BaseMatrix)
    rng = np.random.default_rng(0)
    x_true = jnp.array(rng.standard_normal(A.shape[1]))
    b = A.to_matrix().matvec(x_true)
    return A, BlockArray(V, flat_array=b), x_true


@POLY_SPACES
def test_blockmatrix_tosparse_returns_diamatrix(poly):
    A, b, _ = _vector_block_system(8, poly)
    sparse = A.tosparse()
    assert A.ndim == 2
    assert isinstance(sparse, DiaMatrix)
    assert sparse.shape == A.shape


@POLY_SPACES
def test_blockmatrix_solve_sparse_matches_dense(poly):
    A, b, x_true = _vector_block_system(8, poly)
    # Dense reference
    x_dense = A.to_matrix().solve(b.flatten())
    # Sparse / RCM path
    x_sparse = A.solve(b)
    assert x_sparse.shape == b.shape
    assert jnp.allclose(x_sparse.flatten(), x_dense.ravel(), atol=ulp(1000))


@POLY_SPACES
def test_blockmatrix_rcm_reduces_bandwidth(poly):
    A, _, _ = _vector_block_system(8, poly)
    sparse = A.tosparse()
    A_perm, _, _ = sparse.rcm()
    bw_before = max(abs(k) for k in sparse.offsets)
    bw_after = max(abs(k) for k in A_perm.offsets)
    assert bw_after <= bw_before


@POLY_SPACES
def test_blockmatrix_call_matches_dense_matvec(poly):
    A, b, x_true = _vector_block_system(8, poly)
    # Warm the RCM cache via solve
    _ = A.solve(b)
    y_block = A(BlockArray(A.test_space, flat_array=x_true))
    y_dense = A.to_matrix().matvec(x_true.ravel()).reshape(x_true.shape)
    assert jnp.allclose(y_block.flatten(), y_dense.ravel(), atol=ulp(1000))


@POLY_SPACES
def test_blockmatrix_solve_cached_rcm(poly):
    """Second solve reuses cached RCM without reassembly."""
    A, b, _ = _vector_block_system(8, poly)
    x1 = A.solve(b)
    x2 = A.solve(b)
    assert jnp.allclose(x1.flatten(), x2.flatten(), atol=ulp(10))


# ---------------------------------------------------------------------------
# Banded substitution: sequential scan vs parallel prefix
# ---------------------------------------------------------------------------


def _biharmonic_fourier_poly_2d(n: int, poly):
    """Fourier x poly biharmonic, for the wide-band end of the solver."""
    F = FunctionSpace(n, Fourier)
    D = FunctionSpace(n, poly, {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}})
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2) ** 2
    A, b = inner(
        v * Div(Grad(Div(Grad(u)))) - v * Div(Grad(Div(Grad(ue)))),
        sparse=True,
        kind="system",
    )
    return T, cast(TPMatrices, A), b, ue


@pytest.mark.parametrize(
    "build", [_poisson_fourier_poly_2d, _biharmonic_fourier_poly_2d], ids=["d2", "d4"]
)
@POLY_SPACES
def test_prefix_substitution_agrees_with_scan(build, poly, monkeypatch) -> None:
    """The two substitutions are the same solve, reached at different depths.

    `_affine_prefix` resolves the recurrence by composing affine maps rather
    than stepping through it, which is a different order of operations and so a
    different rounding path -- but not a different answer.
    """
    _, A, b, _ = build(16, poly)

    monkeypatch.setenv("JAXFUN_WAVENUMBER_SUBSTITUTION", "scan")
    seq = tpmats_wavenumber_factor(A).solve(b)
    monkeypatch.setenv("JAXFUN_WAVENUMBER_SUBSTITUTION", "prefix")
    par = tpmats_wavenumber_factor(A).solve(b)

    scale = float(jnp.max(jnp.abs(seq)))
    assert float(jnp.max(jnp.abs(par - seq))) < scale * jnp.sqrt(ulp(10))


def _band_is_narrow_enough(wn) -> bool:
    """The `auto` test in `_make_wavenumber_solve`, over an assembled solver."""
    r = max(
        max((-o for o in wn.L_offsets if o < 0), default=0),
        max((o for o in wn.U_offsets if o > 0), default=0),
    )
    return r * r <= _PREFIX_BAND_SLACK * (len(wn.L_offsets) + len(wn.U_offsets))


def test_band_width_follows_the_formulation_not_the_basis() -> None:
    """What `auto` keys off, and why it cannot key off the basis instead.

    A Chebyshev stiffness matrix assembled Galerkin-style is nearly dense
    upper-triangular, and the prefix substitution -- whose companion stack grows
    as `r**2` against the factors' `r` -- must decline it. The same operator in
    a Petrov-Galerkin formulation is banded, and should be taken. Same basis,
    opposite decision, so the offsets are the only honest thing to read.
    """
    galerkin = tpmats_wavenumber_factor(
        _poisson_fourier_poly_2d(16, Chebyshev.Chebyshev)[1]
    )
    assert not _band_is_narrow_enough(galerkin), (
        f"Galerkin Chebyshev should be too wide, got U_offsets={galerkin.U_offsets}"
    )

    F = FunctionSpace(16, Fourier)
    D = FunctionSpace(16, Chebyshev.Chebyshev, BCS, name="D")
    T = TensorProduct(F, D)
    Tt = TensorProduct(F, D.get_testspace("PG", name="Pt"))
    u, v = TrialFunction(T), TestFunction(Tt)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    A_pg, _ = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True, kind="system")
    petrov = tpmats_wavenumber_factor(cast(TPMatrices, A_pg))
    assert _band_is_narrow_enough(petrov), (
        f"Petrov-Galerkin Chebyshev should be narrow, got U_offsets={petrov.U_offsets}"
    )


def test_unknown_substitution_is_rejected(monkeypatch) -> None:
    """A typo in the override must not silently fall back to a default."""
    monkeypatch.setenv("JAXFUN_WAVENUMBER_SUBSTITUTION", "parallel")
    _, A, _, _ = _poisson_fourier_poly_2d(8, Legendre.Legendre)
    with pytest.raises(ValueError, match="must be 'scan', 'prefix' or 'auto'"):
        tpmats_wavenumber_factor(A)


# ---------------------------------------------------------------------------
# Odd-even (parity) decoupling
# ---------------------------------------------------------------------------


def _poisson_fourier_poly_sizes(n_fourier: int, n_poly: int, poly):
    """Poisson with the two axes sized independently.

    The shared-`n` helpers cannot reach an odd polynomial extent: Fourier
    requires an even mode count, and that is the number they pass to both.
    """
    F = FunctionSpace(n_fourier, Fourier)
    D = FunctionSpace(n_poly, poly, BCS)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True, kind="system")
    return T, cast(TPMatrices, A), b, ue


@POLY_SPACES
@pytest.mark.parametrize("n_poly", [16, 17], ids=["even-extent", "odd-extent"])
def test_parity_decoupling_leaves_the_answer_alone(poly, n_poly, monkeypatch) -> None:
    """Decoupled or not, it is the same operator and the same solution.

    `n_poly=17` gives an odd polynomial extent, where the odd-index block is one
    shorter than the even one and is padded with an identity row. That row's
    solution is zero and is dropped on the way out; if the padding leaked into
    the system the answer would move.
    """
    _, A, b, _ = _poisson_fourier_poly_sizes(16, n_poly, poly)

    monkeypatch.setenv("JAXFUN_WAVENUMBER_PARITY", "off")
    plain = tpmats_wavenumber_factor(A).solve(b)
    monkeypatch.setenv("JAXFUN_WAVENUMBER_PARITY", "on")
    split = tpmats_wavenumber_factor(A).solve(b)

    scale = float(jnp.max(jnp.abs(plain)))
    assert float(jnp.max(jnp.abs(split - plain))) < scale * jnp.sqrt(ulp(10))


@POLY_SPACES
def test_parity_decoupling_halves_the_band_and_the_extent(poly, monkeypatch) -> None:
    """What the decoupling is *for*, stated on the factors it produces.

    Every operator here has even offsets only, so the polynomial axis splits into
    two independent halves. The factors should come back over half the extent
    with the offsets halved -- and contiguous, because the odd slots the window
    used to carry as structural zeros are gone.
    """
    _, A, _, _ = _poisson_fourier_poly_2d(16, poly)

    monkeypatch.setenv("JAXFUN_WAVENUMBER_PARITY", "off")
    plain = tpmats_wavenumber_factor(A)
    monkeypatch.setenv("JAXFUN_WAVENUMBER_PARITY", "on")
    split = tpmats_wavenumber_factor(A)

    assert plain._parity is False and split._parity is True
    # Half the sequential extent, twice the batch: depth traded for width.
    assert split.L.shape[-1] == (plain.L.shape[-1] + 1) // 2
    assert split.L.shape[0] == plain.L.shape[0] * 2
    # Offsets halved, and now adjacent rather than every other one.
    assert split.L_offsets == tuple(o // 2 for o in plain.L_offsets)
    assert split.U_offsets == tuple(o // 2 for o in plain.U_offsets)


def test_parity_decoupling_is_declined_for_odd_offsets(monkeypatch) -> None:
    """An operator that couples the parities must be left alone.

    Nothing in the suite assembles one, so build the predicate's input directly:
    a first-derivative term would put odd offsets in the band, and the two halves
    stop being independent.
    """
    monkeypatch.setenv("JAXFUN_WAVENUMBER_PARITY", "on")
    assert _parity_halves_offsets((-2, 0, 2))
    assert _parity_halves_offsets((-4, -2, 0, 2, 4))
    assert not _parity_halves_offsets((-1, 0, 1))
    assert not _parity_halves_offsets((-2, -1, 0, 2))
    assert not _parity_halves_offsets((0,))


def test_wavenumber_factor_keeps_a_complex_fourier_diagonal() -> None:
    """An odd derivative along the periodic axis makes that diagonal imaginary.

    The working dtype used to come from the polynomial axis alone, so a complex
    Fourier diagonal was cast to real on the way in. For an odd derivative the
    diagonal is *purely* imaginary, so that cast zeroed every k != 0 block and
    the solver factorised a singular matrix and returned nan. Every operator
    previously routed here was Helmholtz-like, whose diagonal is real, which is
    why nothing noticed.
    """
    F = FunctionSpace(16, Fourier)
    D = FunctionSpace(16, Legendre.Legendre, BCS)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2)
    # First derivative in x (imaginary diagonal) plus a mass term, so that the
    # k = 0 block stays non-singular and the system is solvable.
    form = v * u.diff(x, 1).diff(y, 2) - v * u
    rhs_form = v * ue.diff(x, 1).diff(y, 2) - v * ue
    A_raw, b_raw = inner(form - rhs_form, sparse=True, kind="system")
    A, b = cast(TPMatrices, A_raw), cast(Array, b_raw)

    fourier_diag = A.tpmats[0].mats[0].data[0]
    assert bool(jnp.any(jnp.abs(jnp.imag(fourier_diag)) > 0)), (
        "this test is pointless unless the Fourier diagonal really is complex"
    )

    uh = tpmats_wavenumber_factor(A).solve(b)
    assert bool(jnp.all(jnp.isfinite(uh))), "a truncated diagonal returns nan"

    ref = tpmats_to_kron(list(A.tpmats)).solve(b.flatten()).reshape(b.shape)
    scale = float(jnp.max(jnp.abs(ref)))
    assert float(jnp.max(jnp.abs(uh - ref))) < scale * jnp.sqrt(ulp(10))
