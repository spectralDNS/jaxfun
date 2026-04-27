"""Tests for diakron, tpmats_to_kron, TPMatrix matvec, and solvers.

Covers:
- diakron correctness vs jnp.kron on square/non-square DIA inputs
- tpmats_to_kron correctness (dense Matrix path and DIA path) vs numpy kron
- tpmats_to_kron scale is applied exactly once
- TPMatrix._matmul_array  (A @ u) and _rmatmul_array (u @ A) for 2-D and 3-D
- scale propagation through __call__ / __matmul__ / __rmatmul__
- LU and dense solvers on matrices built via tpmats_to_kron
- Poisson-like and biharmonic-like 2D problems solved end-to-end
"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import numpy as np

from jaxfun.galerkin import (
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TPMatrix, tpmats_to_kron
from jaxfun.la import DiaMatrix, Matrix, diags
from jaxfun.la.diamatrix import diakron
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import n as sym_n, ulp


def _diag3(n: int) -> DiaMatrix:
    """Symmetric tridiagonal n×n DiaMatrix."""
    return diags(
        [jnp.ones(n - 1), -2 * jnp.ones(n), jnp.ones(n - 1)],
        offsets=(-1, 0, 1),
    )


def _diag5(n: int) -> DiaMatrix:
    """Symmetric pentadiagonal n×n DiaMatrix."""
    return diags(
        [
            0.5 * jnp.ones(n - 2),
            jnp.ones(n - 1),
            -3 * jnp.ones(n),
            jnp.ones(n - 1),
            0.5 * jnp.ones(n - 2),
        ],
        offsets=(-2, -1, 0, 1, 2),
    )


def _poisson_tpmats(N: int = 8) -> tuple[list[TPMatrix], jnp.ndarray]:
    """Return (A, b) for a 2-D Poisson problem on a Legendre composite space."""
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, Legendre.Legendre, bcs=bcs)
    T = TensorProduct(D, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = (1 - x**2) * (1 - y**2)
    A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)
    return A, b


def _biharmonic_tpmats(N: int = 8) -> tuple[list[TPMatrix], jnp.ndarray]:
    """Return (A, b) for a 2-D biharmonic problem on a Chebyshev composite space."""
    from jaxfun.coordinates import x, y
    from jaxfun.galerkin import Chebyshev

    bcs_x = {
        "left": {"D": 0, "N": 0},
        "right": {"D": 0, "N": 0},
    }
    Dx = FunctionSpace(N, Chebyshev.Chebyshev, bcs=bcs_x, scaling=sym_n + 1)
    T = TensorProduct(Dx, Dx)
    v, u = TestFunction(T), TrialFunction(T)
    ue = (1 - x**2) ** 2 * (1 - y**2) ** 2
    ue_sym = T.system.expr_psi_to_base_scalar(ue)
    A, b = inner(
        Div(Grad(Div(Grad(u)))) * v - Div(Grad(Div(Grad(ue_sym)))) * v,
        sparse=True,
    )
    return A, b


class TestDiakron:
    def test_identity_kron_tridiag(self):
        """I_m ⊗ T_n should equal jnp.kron(I_m, T_n)."""
        m, n = 3, 4
        I = diags([jnp.ones(m)], offsets=(0,))
        T = _diag3(n)
        K = diakron(I, T)
        expected = jnp.kron(jnp.eye(m), np.array(T.todense()))
        assert jnp.allclose(K.todense(), expected, ulp(100))

    def test_tridiag_kron_identity(self):
        """T_m ⊗ I_n should equal jnp.kron(T_m, I_n)."""
        m, n = 4, 3
        T = _diag3(m)
        I = diags([jnp.ones(n)], offsets=(0,))
        K = diakron(T, I)
        expected = jnp.kron(np.array(T.todense()), jnp.eye(n))
        assert jnp.allclose(K.todense(), expected, ulp(100))

    def test_tridiag_kron_tridiag(self):
        """T_m ⊗ T_n vs jnp.kron."""
        m, n = 4, 5
        A = _diag3(m)
        B = _diag3(n)
        K = diakron(A, B)
        expected = jnp.kron(np.array(A.todense()), np.array(B.todense()))
        assert jnp.allclose(K.todense(), expected, ulp(100))

    def test_penta_kron_tridiag(self):
        """Pentadiagonal ⊗ tridiagonal vs jnp.kron."""
        A = _diag5(5)
        B = _diag3(4)
        K = diakron(A, B)
        expected = jnp.kron(np.array(A.todense()), np.array(B.todense()))
        assert jnp.allclose(K.todense(), expected, ulp(100))

    def test_result_shape(self):
        m, n = 3, 5
        A = _diag3(m)
        B = _diag3(n)
        K = diakron(A, B)
        assert K.shape == (m * n, m * n)

    def test_result_is_diamatrix(self):
        K = diakron(_diag3(4), _diag3(3))
        assert isinstance(K, DiaMatrix)

    def test_nonsquare_fallback(self):
        """Non-square B triggers the dense fallback; result must still be correct."""
        A = _diag3(3)
        B = diags(
            [jnp.ones(2), -2 * jnp.ones(3), jnp.ones(3)],
            offsets=(-1, 0, 1),
            shape=(3, 4),
        )
        K = diakron(A, B)
        expected = jnp.kron(np.array(A.todense()), np.array(B.todense()))
        assert jnp.allclose(K.todense(), expected, ulp(100))

    def test_sparse_structure_preserved(self):
        """The number of stored diagonals should equal |A.offsets| * |B.offsets|
        when all Kronecker offset combinations are distinct."""
        A = _diag3(4)  # 3 offsets
        B = _diag3(5)  # 3 offsets
        K = diakron(A, B)
        # offsets = {k_a*5 + k_b} for k_a in {-1,0,1}, k_b in {-1,0,1} → 9 distinct
        assert len(K.offsets) == 9


class TestTpmatsToKron:
    def test_poisson_matches_numpy_kron(self):
        """tpmats_to_kron must agree with np.kron applied to raw factor matrices."""
        A, _ = _poisson_tpmats(N=8)
        C = tpmats_to_kron(A)

        result_np = None
        for tpm in A:
            a0 = np.array(
                tpm.mats[0].todense()
                if isinstance(tpm.mats[0], DiaMatrix)
                else cast(Matrix, tpm.mats[0]).data
            )
            for m in tpm.mats[1:]:
                a1 = np.array(
                    m.todense() if isinstance(m, DiaMatrix) else cast(Matrix, m).data
                )
                a0 = np.kron(a0, a1)
            a0 = a0 * float(tpm.scale)  # ty:ignore[invalid-argument-type]
            result_np = a0 if result_np is None else result_np + a0

        assert np.max(np.abs(np.array(C.todense()) - result_np)) < 1e-5

    def test_poisson_returns_diamatrix(self):
        A, _ = _poisson_tpmats(N=8)
        C = tpmats_to_kron(A)
        assert isinstance(C, DiaMatrix)

    def test_scale_applied_once(self):
        """Doubling tpm.scale and halving the factor matrices must give same result."""
        A, _ = _poisson_tpmats(N=6)
        C_ref = tpmats_to_kron(A)

        # Build a modified list where scale is doubled but first factor halved
        modified = []
        for tpm in A:
            from jaxfun.galerkin.tensorproductspace import TPMatrix
            from jaxfun.la import Matrix

            mats_new = list(tpm.mats)
            first = mats_new[0]
            if isinstance(first, DiaMatrix):
                mats_new[0] = first * 0.5
            else:
                mats_new[0] = Matrix(first.data * 0.5)
            modified.append(
                TPMatrix(
                    mats_new,
                    float(tpm.scale) * 2.0,  # ty:ignore[invalid-argument-type]
                    global_indices=tpm.global_indices,
                )
            )
        C_mod = tpmats_to_kron(modified)
        assert jnp.allclose(C_ref.todense(), C_mod.todense(), atol=1e-5)

    def test_biharmonic_matches_numpy_kron(self):
        """Same correctness check for higher-order (biharmonic) problem."""
        import jax

        jax.config.update("jax_enable_x64", True)
        A, _ = _biharmonic_tpmats(N=10)
        C = tpmats_to_kron(A)

        result_np = None
        for tpm in A:
            a0 = np.array(tpm.mats[0].todense())
            for m in tpm.mats[1:]:
                a1 = np.array(m.todense())
                a0 = np.kron(a0, a1)
            a0 = a0 * float(tpm.scale)  # ty:ignore[invalid-argument-type]
            result_np = a0 if result_np is None else result_np + a0

        assert np.max(np.abs(np.array(C.todense()) - result_np)) < ulp(100)


class TestKronSolve:
    """LU and dense-fallback solvers on matrices assembled via tpmats_to_kron."""

    def test_poisson_lu_solve(self):
        """LU solver should recover the PDE solution to near machine precision."""
        A, b = _poisson_tpmats(N=10)
        C = tpmats_to_kron(A)
        b_flat = b.flatten()
        x = C.solve(b_flat)
        residual = float(jnp.linalg.norm(C.todense() @ x - b_flat))
        assert residual < ulp(1000)

    def test_poisson_lu_vs_dense(self):
        """DIA LU solve and jnp.linalg.solve must agree."""
        A, b = _poisson_tpmats(N=8)
        C = tpmats_to_kron(A)
        b_flat = b.flatten()
        x_lu = C.solve(b_flat)
        x_dense = jnp.linalg.solve(C.todense(), b_flat)
        assert jnp.allclose(x_lu, x_dense, atol=ulp(100))

    def test_poisson_dense_fallback(self):
        """Forcing the dense-fallback path (dense_threshold=0) must give same answer."""
        A, b = _poisson_tpmats(N=8)
        C = cast(DiaMatrix, tpmats_to_kron(A))
        b_flat = b.flatten()
        x_banded = C.solve(b_flat, dense_threshold=10_000)
        x_dense = C.solve(b_flat, dense_threshold=0)
        assert jnp.allclose(x_banded, x_dense, atol=ulp(1000))

    def test_kron_solve_multiple_rhs(self):
        """solve() on a kron matrix must work for 2-D right-hand sides."""
        A, b = _poisson_tpmats(N=6)
        C = cast(DiaMatrix, tpmats_to_kron(A))
        rng = np.random.default_rng(42)
        n = C.shape[0]
        X_true = jnp.array(rng.standard_normal((n, 3)))
        B = C.todense() @ X_true
        X_hat = C.solve(B)
        assert X_hat.shape == (n, 3)
        assert jnp.allclose(X_hat, X_true, atol=ulp(1000))

    def test_biharmonic_dense_fallback(self):
        """Biharmonic problem (wide band) must use the dense-fallback path and
        still produce a low-error solution."""
        import jax
        import pytest

        if not jax.config.x64_enabled:  # ty:ignore[unresolved-attribute]
            pytest.skip("requires float64 (run with jax_enable_x64=True)")

        A, b = _biharmonic_tpmats(N=12)
        C = tpmats_to_kron(A)
        b_flat = b.flatten()
        x = C.solve(b_flat)  # hits dense_threshold automatically
        residual = float(jnp.linalg.norm(C.todense() @ x - b_flat))
        assert residual < ulp(10000)


class TestManualKronSolve:
    """Verify LU solver on matrices built directly via diakron, without the
    Galerkin assembly layer."""

    def _build_laplacian(self, n: int) -> DiaMatrix:
        """Discrete 2-D Laplacian via Kronecker product: I⊗T + T⊗I."""
        T = _diag3(n)
        I = diags([jnp.ones(n)], offsets=(0,))
        return diakron(I, T) + diakron(T, I)

    def test_laplacian_shape(self):
        L = self._build_laplacian(5)
        assert L.shape == (25, 25)

    def test_laplacian_symmetric(self):
        L = self._build_laplacian(6)
        assert jnp.allclose(L.todense(), L.todense().T, atol=ulp(100))

    def test_laplacian_lu_solve(self):
        """LU solve on I⊗T + T⊗I must match dense solve."""
        n = 7
        L = self._build_laplacian(n)
        rng = np.random.default_rng(0)
        b = jnp.array(rng.standard_normal(n * n))
        x_lu = L.solve(b)
        x_dense = jnp.linalg.solve(L.todense(), b)
        assert jnp.allclose(x_lu, x_dense, atol=ulp(100))

    def test_kron_tridiag_tridiag_solve(self):
        """A⊗B x = b must be solved correctly."""
        A = _diag3(4)
        B = _diag3(5)
        K = diakron(A, B)
        rng = np.random.default_rng(1)
        b = jnp.array(rng.standard_normal(20))
        x = K.solve(b)
        assert float(jnp.linalg.norm(K.todense() @ x - b)) < ulp(1000)

    def test_kron_pentadiag_solve(self):
        """solve on A⊗B where A and B are pentadiagonal."""
        A = _diag5(6)
        B = _diag5(5)
        K = diakron(A, B)
        rng = np.random.default_rng(2)
        b = jnp.array(rng.standard_normal(30))
        x = K.solve(b)
        assert float(jnp.linalg.norm(K.todense() @ x - b)) < ulp(1000)

    def test_lu_vs_dense_on_kron(self):
        """LU and dense paths must agree for a manually built kron matrix."""
        L = self._build_laplacian(5)
        rng = np.random.default_rng(3)
        b = jnp.array(rng.standard_normal(25))
        x_banded = L.solve(b, dense_threshold=10_000)
        x_dense = L.solve(b, dense_threshold=0)
        assert jnp.allclose(x_banded, x_dense, atol=ulp(100))


# ---------------------------------------------------------------------------
# TPMatrix._matmul_array and _rmatmul_array
# ---------------------------------------------------------------------------


def _make_tpmatrix_2d(
    m0: int, n0: int, m1: int, n1: int, scale: float = 1.0
) -> tuple[TPMatrix, np.ndarray]:
    """Return a TPMatrix(A0, A1) and the corresponding dense Kronecker product."""
    rng = np.random.default_rng(7)
    a0 = rng.standard_normal((m0, n0)).astype(np.float32)
    a1 = rng.standard_normal((m1, n1)).astype(np.float32)
    tp = TPMatrix(
        [Matrix(jnp.array(a0)), Matrix(jnp.array(a1))],  # ty:ignore[invalid-argument-type]
        scale,
    )
    K = np.kron(a0, a1) * scale
    return tp, K


def _make_tpmatrix_3d(
    sizes: tuple[int, int, int], scale: float = 1.0, seed: int = 11
) -> tuple[TPMatrix, np.ndarray]:
    """Return a square TPMatrix(A0, A1, A2) and matching dense Kronecker product."""
    rng = np.random.default_rng(seed)
    mats_np = [rng.standard_normal((s, s)).astype(np.float32) for s in sizes]
    K = mats_np[0]
    for a in mats_np[1:]:
        K = np.kron(K, a)
    K = K * scale
    tp = TPMatrix([Matrix(jnp.array(a)) for a in mats_np], scale)  # ty:ignore[invalid-argument-type]
    return tp, K


class TestTPMatrixMatmul:
    """Tests for TPMatrix._matmul_array (A @ u) and _rmatmul_array (u @ A)."""

    # ------------------------------------------------------------------
    # 2-D: square factor matrices
    # ------------------------------------------------------------------

    def test_2d_matmul_vs_kron(self):
        """(A0⊗A1) @ vec(w) == vec(A0 @ w @ A1.T) for square factors."""
        n0, n1 = 5, 4
        tp, K = _make_tpmatrix_2d(n0, n0, n1, n1)
        rng = np.random.default_rng(0)
        w = jnp.array(rng.standard_normal((n0, n1)).astype(np.float32))
        result = tp(w)
        expected = (K @ np.array(w).ravel()).reshape(n0, n1)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_2d_rmatmul_vs_kron(self):
        """vec(w) @ (A0⊗A1) == vec(A0.T @ w @ A1)."""
        n0, n1 = 5, 4
        tp, K = _make_tpmatrix_2d(n0, n0, n1, n1)
        rng = np.random.default_rng(1)
        w = jnp.array(rng.standard_normal((n0, n1)).astype(np.float32))
        result = w @ tp  # calls __rmatmul__
        expected = (np.array(w).ravel() @ K).reshape(n0, n1)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_2d_matmul_rmatmul_consistent(self):
        """For a symmetric Kronecker product A@w and w@A must coincide."""
        n = 4
        T = _diag3(n)
        I = diags([jnp.ones(n)], offsets=(0,))
        K_dia = diakron(T, I)  # symmetric
        tp = TPMatrix([T, I], 1.0)  # ty:ignore[invalid-argument-type]
        rng = np.random.default_rng(2)
        w = jnp.array(rng.standard_normal((n, n)).astype(np.float32))
        # A @ w (flattened) and w @ A (flattened) differ only by interpretation:
        # check against kron dense product
        K_dense = np.array(K_dia.todense())
        fwd = np.array(tp(w)).ravel()
        rev = np.array(w @ tp).ravel()
        assert np.allclose(fwd, K_dense @ np.array(w).ravel(), atol=1e-5)
        assert np.allclose(rev, np.array(w).ravel() @ K_dense, atol=1e-5)

    # ------------------------------------------------------------------
    # 2-D: non-square factor matrices
    # ------------------------------------------------------------------

    def test_2d_matmul_nonsquare(self):
        """Works when A0 is m0×n0 with m0 ≠ n0."""
        tp, K = _make_tpmatrix_2d(3, 5, 4, 6)  # (12, 30) Kronecker product
        rng = np.random.default_rng(3)
        w = jnp.array(rng.standard_normal((5, 6)).astype(np.float32))
        result = tp(w)
        expected = (K @ np.array(w).ravel()).reshape(3, 4)
        assert result.shape == (3, 4)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_2d_rmatmul_nonsquare(self):
        """vec(w) @ kron for non-square factors."""
        tp, K = _make_tpmatrix_2d(3, 5, 4, 6)
        rng = np.random.default_rng(4)
        w = jnp.array(rng.standard_normal((3, 4)).astype(np.float32))
        result = w @ tp
        expected = (np.array(w).ravel() @ K).reshape(5, 6)
        assert result.shape == (5, 6)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    # ------------------------------------------------------------------
    # 2-D: scale propagation
    # ------------------------------------------------------------------

    def test_2d_scale_applied(self):
        """scale=3 must multiply the output by 3."""
        n0, n1 = 4, 5
        tp1, _ = _make_tpmatrix_2d(n0, n0, n1, n1, scale=1.0)
        # same matrices, tripled scale
        tp3 = TPMatrix(list(tp1.mats), scale=3.0)
        rng = np.random.default_rng(5)
        w = jnp.array(rng.standard_normal((n0, n1)).astype(np.float32))
        assert jnp.allclose(tp3(w), 3.0 * tp1(w), atol=1e-5)

    def test_2d_rmatmul_scale_applied(self):
        n0, n1 = 4, 5
        tp1, _ = _make_tpmatrix_2d(n0, n0, n1, n1, scale=1.0)
        # same matrices, double scale
        tp2 = TPMatrix(list(tp1.mats), scale=2.0)
        rng = np.random.default_rng(6)
        w = jnp.array(rng.standard_normal((n0, n1)).astype(np.float32))
        assert jnp.allclose(w @ tp2, 2.0 * (w @ tp1), atol=1e-5)

    # ------------------------------------------------------------------
    # 3-D: square factor matrices
    # ------------------------------------------------------------------

    def test_3d_matmul_vs_kron(self):
        """(A0⊗A1⊗A2) @ vec(w) == result from sequential matvec."""
        # Use uniform sizes to avoid JIT pytree shape mismatch across tests
        tp, K = _make_tpmatrix_3d((4, 4, 4))
        rng = np.random.default_rng(8)
        w = jnp.array(rng.standard_normal((4, 4, 4)).astype(np.float32))
        result = tp(w)
        expected = (K @ np.array(w).ravel()).reshape(4, 4, 4)
        assert result.shape == (4, 4, 4)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-4)

    def test_3d_rmatmul_vs_kron(self):
        """vec(w) @ (A0⊗A1⊗A2) via __rmatmul__."""
        tp, K = _make_tpmatrix_3d((4, 4, 4), seed=99)
        rng = np.random.default_rng(9)
        w = jnp.array(rng.standard_normal((4, 4, 4)).astype(np.float32))
        result = w @ tp
        expected = (np.array(w).ravel() @ K).reshape(4, 4, 4)
        assert result.shape == (4, 4, 4)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-4)

    def test_3d_scale_applied(self):
        """scale propagates correctly in 3-D."""
        tp1, _ = _make_tpmatrix_3d((4, 4, 4), scale=1.0)
        tp4 = TPMatrix(list(tp1.mats), scale=4.0)
        rng = np.random.default_rng(10)
        w = jnp.array(rng.standard_normal((4, 4, 4)).astype(np.float32))
        assert jnp.allclose(tp4(w), 4.0 * tp1(w), atol=1e-4)

    # ------------------------------------------------------------------
    # Galerkin-assembled TPMatrix (DIA factor matrices)
    # ------------------------------------------------------------------

    def test_galerkin_2d_matmul_vs_kron(self):
        """TPMatrix from inner() with DIA factors: A@u matches tpmats_to_kron @ u."""
        A, _ = _poisson_tpmats(N=8)
        C = tpmats_to_kron(A)
        rng = np.random.default_rng(20)
        shape = tuple(m.shape[1] for m in A[0].mats)
        w = jnp.array(rng.standard_normal(shape).astype(np.float32))
        result = sum(tpm(w) for tpm in A)
        expected = (C.todense() @ np.array(w).ravel()).reshape(shape)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-4)

    def test_galerkin_2d_rmatmul_vs_kron(self):
        """w @ TPMatrix from inner() matches w @ tpmats_to_kron."""
        A, _ = _poisson_tpmats(N=8)
        C = tpmats_to_kron(A)
        rng = np.random.default_rng(21)
        shape = tuple(m.shape[0] for m in A[0].mats)
        w = jnp.array(rng.standard_normal(shape).astype(np.float32))
        result = sum(w @ tpm for tpm in A)
        expected = (np.array(w).ravel() @ C.todense()).reshape(shape)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-4)


class TestTpmatsToScipySparse:
    """Tests for tpmats_to_scipy_sparse and tpmats_to_scipy_kron."""

    def test_scipy_sparse_factors_match_dense(self):
        """Each factor in tpmats_to_scipy_sparse matches the TPMatrix dense factor."""
        from jaxfun.galerkin.tensorproductspace import tpmats_to_scipy_sparse

        A, _ = _poisson_tpmats(N=8)
        result = tpmats_to_scipy_sparse(A)
        assert len(result) == len(A)
        for tpm, (f0, f1) in zip(A, result):
            scale = complex(tpm.scale).real
            assert np.allclose(
                f0.toarray(), np.array(tpm.mats[0].todense()) * scale, atol=1e-6
            )
            assert np.allclose(f1.toarray(), np.array(tpm.mats[1].todense()), atol=1e-6)

    def test_scipy_sparse_scale_applied(self):
        """Scale is folded into first factor, not applied twice."""
        from jaxfun.galerkin.tensorproductspace import tpmats_to_scipy_sparse

        A, _ = _poisson_tpmats(N=8)
        # Build a scaled copy
        scaled = [TPMatrix(list(tpm.mats), scale=tpm.scale * 3) for tpm in A]
        orig = tpmats_to_scipy_sparse(A)
        scl = tpmats_to_scipy_sparse(scaled)
        for (o0, o1), (s0, s1) in zip(orig, scl):
            assert np.allclose(s0.toarray(), 3 * o0.toarray(), atol=1e-6)
            assert np.allclose(s1.toarray(), o1.toarray(), atol=1e-6)

    def test_scipy_kron_matches_tpmats_to_kron(self):
        """tpmats_to_scipy_kron assembles the same global matrix as tpmats_to_kron."""
        from jaxfun.galerkin.tensorproductspace import tpmats_to_scipy_kron

        A, _ = _poisson_tpmats(N=8)
        K_jax = np.array(tpmats_to_kron(A).todense())
        K_sp = tpmats_to_scipy_kron(A).toarray()
        assert np.allclose(K_jax, K_sp, atol=1e-5)

    def test_scipy_kron_scale_matches_tpmats_to_kron(self):
        """Scaled tpmats_to_scipy_kron matches scaled tpmats_to_kron."""
        from jaxfun.galerkin.tensorproductspace import tpmats_to_scipy_kron

        A, _ = _poisson_tpmats(N=8)
        scaled = [TPMatrix(list(tpm.mats), scale=tpm.scale * 2) for tpm in A]
        K_jax = np.array(tpmats_to_kron(scaled).todense())
        K_sp = tpmats_to_scipy_kron(scaled).toarray()
        assert np.allclose(K_jax, K_sp, atol=1e-5)

    def test_scipy_kron_matvec_matches_solve_rhs(self):
        """Scipy kron matrix applied to known vector matches JAX kron result."""
        from jaxfun.galerkin.tensorproductspace import tpmats_to_scipy_kron

        A, b = _poisson_tpmats(N=8)
        K_sp = tpmats_to_scipy_kron(A)
        K_jax = tpmats_to_kron(A)
        x = np.random.default_rng(42).standard_normal(K_sp.shape[1]).astype(np.float32)
        assert np.allclose(K_sp @ x, np.array(K_jax.todense()) @ x, atol=1e-4)
