"""Tests for diakron, tpmats_to_kron, and solvers on Kronecker-assembled matrices.

Covers:
- diakron correctness vs jnp.kron on square/non-square DIA inputs
- tpmats_to_kron correctness (dense Matrix path and DIA path) vs numpy kron
- tpmats_to_kron scale is applied exactly once
- LU and dense solvers on matrices built via tpmats_to_kron
- Poisson-like and biharmomic-like 2D problems solved end-to-end
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

        jax.config.update("jax_enable_x64", True)
        A, b = _biharmonic_tpmats(N=12)
        C = tpmats_to_kron(A)
        b_flat = b.flatten()
        x = C.solve(b_flat)  # hits dense_threshold automatically
        residual = float(jnp.linalg.norm(C.todense() @ x - b_flat))
        assert residual < ulp(100)


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
