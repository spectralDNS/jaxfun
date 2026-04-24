"""Tests for DiaMatrix and diags in sparsemat.py."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse

from jaxfun.la.diamatrix import DiaMatrix, LUFactors, diags


def _tridiag(n: int) -> tuple[np.ndarray, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    a = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
    )
    A = DiaMatrix.from_dense(jnp.array(a), offsets=(-1, 0, 1))
    return a, A


def _pentadiag(n: int) -> tuple[np.ndarray, DiaMatrix]:
    """Return a symmetric pentadiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    a = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
        + np.diag(-0.5 * np.ones(n - 2), 2)
        + np.diag(-0.5 * np.ones(n - 2), -2)
    )
    A = DiaMatrix.from_dense(jnp.array(a), offsets=(-2, -1, 0, 1, 2))
    return a, A


def _tridiagO(n: int) -> tuple[np.ndarray, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    a = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 2), 2)
        + np.diag(-np.ones(n - 2), -2)
    )
    A = DiaMatrix.from_dense(jnp.array(a), offsets=(-2, 0, 2))
    return a, A


def _pentadiagO(n: int) -> tuple[np.ndarray, DiaMatrix]:
    """Return a symmetric pentadiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    a = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 2), 2)
        + np.diag(-np.ones(n - 2), -2)
        + np.diag(-0.5 * np.ones(n - 4), 4)
        + np.diag(-0.5 * np.ones(n - 4), -4)
    )
    A = DiaMatrix.from_dense(jnp.array(a), offsets=(-4, -2, 0, 2, 4))
    return a, A


def _circdiag(n: int) -> tuple[np.ndarray, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (dense_numpy, DiaMatrix)."""
    a = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
        + np.diag(-np.ones(1), n - 1)  # wrap-around super-diagonal
        + np.diag(np.ones(1), -(n - 1))  # wrap-around sub-diagonal
    )
    A = DiaMatrix.from_dense(jnp.array(a), offsets=(-n + 1, -1, 0, 1, n - 1))
    return a, A


trimatrices = (_tridiag, _tridiagO)
allmatrices = (_tridiag, _pentadiag, _tridiagO, _pentadiagO, _circdiag)


class TestConstruction:
    @pytest.mark.parametrize("mat", trimatrices)
    def test_from_dense_shape(self, mat):
        a, A = mat(6)
        assert A.shape == (6, 6)
        assert A.data.shape == (3, 6)  # 3 diagonals, padded to n_cols=6

    @pytest.mark.parametrize("mat", allmatrices)
    def test_from_dense_default_offsets(self, mat):
        """from_dense with no offsets should detect all non-zero diagonals."""
        a, A = mat(8)
        A2 = DiaMatrix.from_dense(jnp.array(a))
        assert jnp.allclose(A2.todense(), jnp.array(a))

    def test_from_dense_nonsquare(self):
        a = jnp.zeros((3, 5))
        a = a.at[0, 0].set(1.0)
        a = a.at[1, 2].set(2.0)
        A = DiaMatrix.from_dense(a, offsets=(0, 2))
        assert A.shape == (3, 5)
        assert A.data.shape == (2, 5)

    def test_diags_square_inferred(self):
        """diags with no shape should infer a square matrix."""
        A = diags([jnp.ones(4), -2 * jnp.ones(5), jnp.ones(4)], offsets=(-1, 0, 1))
        assert A.shape == (5, 5)

    def test_diags_nonsquare_explicit(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        assert A.shape == (4, 6)
        assert A.data.shape == (3, 6)

    def test_diags_single_diagonal(self):
        A = diags([jnp.array([3.0, 3.0, 3.0])], offsets=(0,))
        assert A.shape == (3, 3)
        assert jnp.allclose(A.diagonal(), jnp.array([3.0, 3.0, 3.0]))

    def test_diags_scalar_broadcast(self):
        """A length-1 diagonal should be broadcast to the full diagonal length."""
        n = 5
        A = diags(
            [jnp.array([-1.0]), jnp.array([2.0]), jnp.array([-1.0])],
            offsets=(-1, 0, 1),
            shape=(n, n),
        )
        assert A.shape == (n, n)
        assert jnp.allclose(A.diagonal(0), 2.0 * jnp.ones(n))
        assert jnp.allclose(A.diagonal(1), -1.0 * jnp.ones(n - 1))
        assert jnp.allclose(A.diagonal(-1), -1.0 * jnp.ones(n - 1))
        # dense round-trip via todense
        import numpy as np

        dense = np.array(A.todense())
        expected = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        assert jnp.allclose(jnp.array(dense), jnp.array(expected))

    def test_diags_vs_scipy(self):
        """diags should match scipy.sparse.diags for square matrices."""
        d_sub = np.ones(4)
        d_main = -2 * np.ones(5)
        d_sup = np.ones(4)
        A = diags(
            [jnp.array(d_sub), jnp.array(d_main), jnp.array(d_sup)], offsets=(-1, 0, 1)
        )
        A_scipy = scipy.sparse.diags(
            [d_sub, d_main, d_sup], offsets=(-1, 0, 1)
        ).toarray()
        assert jnp.allclose(A.todense(), jnp.array(A_scipy))

    def test_init_stores_int32_offsets(self):
        _, A = _tridiag(3)
        assert len(A.offsets) == 3


class TestToDense:
    @pytest.mark.parametrize("mat", allmatrices)
    def test_round_trip_square(self, mat):
        a, A = mat(6)
        assert jnp.allclose(A.todense(), jnp.array(a))

    def test_round_trip_nonsquare(self):
        # Build a known sparse 4×7 matrix by hand via diags.
        A = diags(
            [
                jnp.array([1.0, 2.0, 3.0]),
                jnp.array([4.0, 5.0, 6.0, 7.0]),
                jnp.array([8.0, 9.0, 10.0, 11.0]),
            ],
            offsets=(-1, 0, 1),
            shape=(4, 7),
        )
        assert jnp.allclose(A.todense(), A.todense())  # idempotent
        A2 = DiaMatrix.from_dense(A.todense(), offsets=(-1, 0, 1))
        assert jnp.allclose(A2.todense(), A.todense())

    def test_zero_matrix(self):
        A = DiaMatrix.from_dense(jnp.zeros((3, 4)), offsets=(0,))
        assert jnp.allclose(A.todense(), jnp.zeros((3, 4)))


class TestMatvec:
    def test_matvec_identity(self):
        A = DiaMatrix.from_dense(jnp.eye(4), offsets=(0,))
        x = jnp.arange(4, dtype=float)
        assert jnp.allclose(A.matvec(x), x)

    def test_matvec_tridiag(self):
        a, A = _tridiag(4)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert jnp.allclose(A.matvec(x), jnp.array(a) @ x)

    def test_matvec_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        x = jnp.ones(6)
        expected = A.todense() @ x
        assert jnp.allclose(A.matvec(x), expected)

    def test_matmat_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        AT = A.T
        expected = A.todense() @ AT.todense()
        H = A @ AT
        assert jnp.allclose(H.todense(), expected)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmat_shape(self, mat):
        _, A = mat(8)
        X = jnp.ones((8, 3))
        Y = A.matmat(X)
        assert Y.shape == (8, 3)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmat_vs_dense(self, mat):
        a, A = mat(6)
        rng = np.random.default_rng(1)
        X = jnp.array(rng.standard_normal((6, 3)))
        assert jnp.allclose(A.matmat(X), jnp.array(a) @ X, atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_apply_dispatches(self, mat):
        _, A = mat(8)
        x = jnp.ones(8)
        X = jnp.ones((8, 2))
        assert A.apply(x).shape == (8,)
        assert A.apply(X).shape == (8, 2)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_vector(self, mat):
        a, A = mat(6)
        x = jnp.arange(6, dtype=float)
        assert jnp.allclose(A @ x, jnp.array(a) @ x)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_matrix(self, mat):
        a, A = mat(6)
        X = jnp.eye(6)
        assert jnp.allclose(A @ X, jnp.array(a), atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_diamatrix(self, mat):
        a, A = mat(6)
        result = A @ A
        expected = jnp.array(a) @ jnp.array(a)
        assert jnp.allclose(result.todense(), expected, atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_rmatmul_vector(self, mat):
        a, A = mat(6)
        x = jnp.arange(6, dtype=float)
        assert jnp.allclose(x @ A, x @ jnp.array(a))

    @pytest.mark.parametrize("mat", allmatrices)
    def test_rmatmul_matrix(self, mat):
        a, A = mat(6)
        X = jnp.eye(6)
        assert jnp.allclose(X @ A, jnp.array(a), atol=1e-6)


class TestTranspose:
    @pytest.mark.parametrize("mat", allmatrices)
    def test_T_shape_square(self, mat):
        _, A = mat(6)
        assert A.T.shape == (6, 6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_T_dense_square(self, mat):
        a, A = mat(6)
        assert jnp.allclose(A.T.todense(), jnp.array(a).T)

    def test_T_shape_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        AT = A.T
        assert AT.shape == (6, 4)

    def test_T_dense_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        assert jnp.allclose(A.T.todense(), A.todense().T)

    def test_T_offsets_negated(self):
        _, A = _tridiag(4)
        # A is symmetric so offsets of T should be negated [-1,0,1] -> [1,0,-1]
        assert set(A.T.offsets) == {1, 0, -1}

    @pytest.mark.parametrize("mat", allmatrices)
    def test_double_T_identity(self, mat):
        _, A = mat(6)
        assert jnp.allclose(A.T.T.todense(), A.todense())


class TestDiagonal:
    def test_main_diagonal(self):
        _, A = _tridiag(4)
        assert jnp.allclose(A.diagonal(0), jnp.array([2.0, 2.0, 2.0, 2.0]))

    def test_super_diagonal(self):
        _, A = _tridiag(4)
        assert jnp.allclose(A.diagonal(1), jnp.array([-1.0, -1.0, -1.0]))

    def test_sub_diagonal(self):
        _, A = _tridiag(4)
        assert jnp.allclose(A.diagonal(-1), jnp.array([-1.0, -1.0, -1.0]))

    def test_unstored_diagonal_zeros(self):
        _, A = _tridiag(4)
        # diagonal 2 is not stored; should return zeros of correct length
        d = A.diagonal(2)
        assert d.shape == (2,)
        assert jnp.allclose(d, jnp.zeros(2))

    def test_out_of_range_diagonal(self):
        _, A = _tridiag(3)
        d = A.diagonal(5)
        assert d.shape == (0,)

    def test_diagonal_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        np_dense = np.array(A.todense())
        assert jnp.allclose(A.diagonal(0), jnp.array(np.diag(np_dense, 0)))
        assert jnp.allclose(A.diagonal(1), jnp.array(np.diag(np_dense, 1)))
        assert jnp.allclose(A.diagonal(-1), jnp.array(np.diag(np_dense, -1)))


class TestScalarArithmetic:
    def test_scale(self):
        a, A = _tridiag(3)
        assert jnp.allclose((A.scale(3.0)).todense(), 3.0 * jnp.array(a))

    def test_mul(self):
        a, A = _tridiag(3)
        assert jnp.allclose((A * 2.0).todense(), 2.0 * jnp.array(a))

    def test_rmul(self):
        a, A = _tridiag(3)
        assert jnp.allclose((0.5 * A).todense(), 0.5 * jnp.array(a))

    def test_neg(self):
        a, A = _tridiag(3)
        assert jnp.allclose((-A).todense(), -jnp.array(a))


class TestMatvecAxis:
    """Test matvec / apply with the axis parameter on N-D arrays."""

    @pytest.mark.parametrize("mat", allmatrices)
    def test_axis0_2d_equals_matmat(self, mat):
        """matvec(X, axis=0) should equal the classic A @ X product."""
        a, A = mat(6)
        rng = np.random.default_rng(7)
        X = jnp.array(rng.standard_normal((6, 5)))
        expected = jnp.array(a) @ X
        assert jnp.allclose(A.matvec(X, axis=0), expected, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_axis1_2d(self, mat):
        """matvec(X, axis=1) contracts along columns: each row of X gets multiplied."""
        a, A = mat(6)
        rng = np.random.default_rng(8)
        X = jnp.array(rng.standard_normal((5, 6)))  # (batch, m)
        expected = (jnp.array(a) @ X.T).T  # (n, batch) → (batch, n)
        assert jnp.allclose(A.matvec(X, axis=1), expected, atol=1e-5)

    def test_axis_1d_ignores_axis(self):
        """For 1-D input the axis argument is irrelevant."""
        a, A = _tridiag(4)
        x = jnp.arange(4, dtype=float)
        result = A.matvec(x, axis=0)
        assert jnp.allclose(result, jnp.array(a) @ x)

    def test_axis0_3d(self):
        """matvec(T, axis=0): T shape (m, b1, b2) → (n, b1, b2)."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(9)
        T = jnp.array(rng.standard_normal((4, 3, 2)))
        # Expected: contract axis 0 with A
        # result[i, j, k] = sum_l A[i, l] * T[l, j, k]
        expected = jnp.einsum("il,ljk->ijk", jnp.array(a), T)
        assert jnp.allclose(A.matvec(T, axis=0), expected, atol=1e-5)

    def test_axis1_3d(self):
        """matvec(T, axis=1): T shape (b0, m, b2) → (b0, n, b2)."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(10)
        T = jnp.array(rng.standard_normal((3, 4, 2)))
        expected = jnp.einsum("il,jlk->jik", jnp.array(a), T)
        assert jnp.allclose(A.matvec(T, axis=1), expected, atol=1e-5)

    def test_axis2_3d(self):
        """matvec(T, axis=2): T shape (b0, b1, m) → (b0, b1, n)."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(11)
        T = jnp.array(rng.standard_normal((3, 2, 4)))
        expected = jnp.einsum("il,jkl->jki", jnp.array(a), T)
        assert jnp.allclose(A.matvec(T, axis=2), expected, atol=1e-5)

    def test_negative_axis(self):
        """Negative axis should work like numpy convention."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(12)
        T = jnp.array(rng.standard_normal((3, 2, 4)))
        assert jnp.allclose(A.matvec(T, axis=-1), A.matvec(T, axis=2), atol=1e-6)

    def test_output_shape_axis0(self):
        a, A = _tridiag(4)  # 4×4 square → n == m == 4
        T = jnp.ones((4, 5, 6))
        assert A.matvec(T, axis=0).shape == (4, 5, 6)

    def test_output_shape_nonsquare(self):
        """Non-square matrix changes the axis size from m to n."""
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        T = jnp.ones((6, 5))  # axis 0 has size m=6
        Y = A.matvec(T, axis=0)
        assert Y.shape == (4, 5)  # axis 0 now has size n=4

    def test_apply_axis(self):
        """apply() should forward the axis argument to matvec."""
        a, A = _tridiag(4)
        X = jnp.ones((4, 3))
        assert jnp.allclose(A.apply(X, axis=0), A.matvec(X, axis=0))
        assert jnp.allclose(A.apply(X.T, axis=1), A.matvec(X.T, axis=1))


class TestAddSub:
    def test_add(self):
        a, A = _tridiag(4)
        result = A + A
        assert jnp.allclose(result.todense(), 2.0 * jnp.array(a))

    def test_sub_zero(self):
        _, A = _tridiag(4)
        result = A - A
        assert jnp.allclose(result.todense(), jnp.zeros((4, 4)))

    def test_add_shape_mismatch(self):
        _, A = _tridiag(3)
        _, B = _tridiag(4)
        with pytest.raises(ValueError):
            _ = A + B


class TestProperties:
    def test_nnz_square(self):
        _, A = _tridiag(5)
        # main=5, sub=4, sup=4
        assert A.nnz == 13

    def test_nnz_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        # main=4, sub(-1)=3, sup(+1)=4
        assert A.nnz == 11

    def test_ndim(self):
        _, A = _tridiag(3)
        assert A.ndim == 2

    def test_dtype(self):
        _, A = _tridiag(3)
        assert A.dtype == jnp.float32

    def test_astype(self):
        # float16 is always available regardless of x64 mode.
        _, A = _tridiag(3)
        B = A.astype(jnp.float16)
        assert B.dtype == jnp.float16
        assert jnp.allclose(B.todense().astype(jnp.float32), A.todense(), atol=1e-2)

    def test_repr(self):
        _, A = _tridiag(3)
        r = repr(A)
        assert "DiaMatrix" in r
        assert "shape=(3, 3)" in r


class TestSolve:
    @pytest.mark.parametrize("mat", allmatrices)
    def test_solve_small(self, mat):
        a, A = mat(7)
        x_true = jnp.arange(1, 8, dtype=float)
        b = a @ x_true
        x_hat = A.solve(b)
        assert jnp.allclose(x_hat, x_true, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_solve_residual(self, mat):
        a, A = mat(8)
        rng = np.random.default_rng(42)
        b = jnp.array(rng.standard_normal(8))
        x = A.solve(b)
        assert float(jnp.linalg.norm(A.matvec(x) - b)) < 1e-5

    @pytest.mark.parametrize("mat", allmatrices)
    def test_solve_axis1(self, mat):
        """A.solve(B, axis=1): each row of B is a separate RHS."""
        a, A = mat(6)
        rng = np.random.default_rng(55)
        X_true = jnp.array(rng.standard_normal((4, 6)))
        B = (jnp.array(a) @ X_true.T).T  # (4, 6)
        X_hat = A.solve(B, axis=1)
        assert X_hat.shape == (4, 6)
        assert jnp.allclose(X_hat, X_true, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_solve_axis_matches_matvec(self, mat):
        """A.solve(A.matvec(X, axis=k), axis=k) == X for k in 0,1,2."""
        a, A = mat(7)
        rng = np.random.default_rng(66)
        for ax in (0, 1, 2):
            shape = [3, 7, 4]
            shape[ax] = 7
            X = jnp.array(rng.standard_normal(shape))
            B = A.matvec(X, axis=ax)
            X_hat = A.solve(B, axis=ax)
            assert X_hat.shape == tuple(shape)
            assert jnp.allclose(X_hat, X, atol=1e-5), f"failed for axis={ax}"


class TestLU:
    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_factor_returns_lu_factors(self, mat):
        _, A = mat(4)
        lu = A.lu_factor()
        assert isinstance(lu, LUFactors)
        assert isinstance(lu.L, DiaMatrix)
        assert isinstance(lu.U, DiaMatrix)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_L_shape(self, mat):
        _, A = mat(5)
        lu = A.lu_factor()
        assert lu.L.shape == (5, 5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_U_shape(self, mat):
        _, A = mat(5)
        lu = A.lu_factor()
        assert lu.U.shape == (5, 5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_L_unit_diagonal(self, mat):
        """L must have a unit main diagonal."""
        _, A = mat(5)
        lu = A.lu_factor()
        assert jnp.allclose(lu.L.diagonal(0), jnp.ones(5))

    def test_lu_L_offsets(self):
        """For a tridiagonal matrix (lower bandwidth 1), L has offsets (-1, 0)."""
        _, A = _pentadiag(5)
        lu = A.lu_factor()
        assert set(lu.L.offsets) == {-2, -1, 0}

    def test_lu_U_offsets(self):
        """For a tridiagonal (no pivoting), U has offsets (0, 1)."""
        _, A = _pentadiagO(5)
        lu = A.lu_factor()
        assert set(lu.U.offsets) == {0, 1, 2, 3, 4}

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_product_equals_original(self, mat):
        """L @ U should reproduce the permuted matrix P @ A."""
        a, A = mat(5)
        lu = A.lu_factor()
        LU = cast(DiaMatrix, lu.L @ lu.U)
        PA = jnp.array(a) if lu.perm is None else jnp.array(a)[lu.perm, :]
        assert jnp.allclose(LU.todense(), PA, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_exact(self, mat):
        a, A = mat(5)
        x_true = jnp.arange(1.0, 6.0)
        b = jnp.array(a) @ x_true
        lu = A.lu_factor()
        x_hat = lu.solve(b)
        assert jnp.allclose(x_hat, x_true, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_random_rhs(self, mat):
        a, A = mat(8)
        rng = np.random.default_rng(99)
        b = jnp.array(rng.standard_normal(8))
        lu = A.lu_factor()
        x = lu.solve(b)
        assert float(jnp.linalg.norm(A.matvec(x) - b)) < 1e-5

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_multiple_rhs(self, mat):
        """solve() should work for 2-D right-hand sides (n, k)."""
        a, A = mat(6)
        rng = np.random.default_rng(7)
        X_true = jnp.array(rng.standard_normal((6, 4)))
        B = jnp.array(a) @ X_true
        lu = A.lu_factor()
        X_hat = lu.solve(B)
        assert X_hat.shape == (6, 4)
        assert jnp.allclose(X_hat, X_true, atol=1e-5)

    def test_lu_nonsquare_raises(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        with pytest.raises(ValueError, match="square"):
            A.lu_factor()

    def test_lu_repr(self):
        _, A = _tridiag(3)
        lu = A.lu_factor()
        assert "LUFactors" in repr(lu)

    def test_lu_diagonal_only(self):
        """Pure diagonal matrix: L == I, U == A."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = diags([d], offsets=(0,))
        lu = A.lu_factor()
        assert jnp.allclose(lu.U.diagonal(0), d)
        assert jnp.allclose(lu.L.diagonal(0), jnp.ones(4))
        b = jnp.array([2.0, 4.0, 6.0, 8.0])
        x = lu.solve(b)
        assert jnp.allclose(x, b / d, atol=1e-6)

    def test_lu_zero_diagonal(self):
        """Matrix whose main diagonal starts with zero — requires pivoting."""
        # Construct a simple banded matrix with A[0,0] = 0:
        #   A = [[0, 1, 0],
        #        [1, 2, 1],
        #        [0, 1, 2]]
        a = jnp.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        A = DiaMatrix.from_dense(a, offsets=(-1, 0, 1))
        lu = A.lu_factor(pivot=True)
        # L @ U == P @ A
        PA = a if lu.perm is None else a[lu.perm, :]
        H = cast(DiaMatrix, lu.L @ lu.U)
        assert jnp.allclose(H.todense(), PA, atol=1e-5)
        # Solve A x = b
        x_true = jnp.array([1.0, 2.0, 3.0])
        b = a @ x_true
        x = lu.solve(b)
        assert jnp.allclose(x, x_true, atol=1e-5)

    def test_lu_perm_attribute(self):
        """perm is None when no pivoting occurred; a JAX int array when it did."""
        _, A = _tridiag(4)
        lu_no_swap = A.lu_factor()
        assert lu_no_swap.perm is None

        # zero diagonal forces a swap — must use pivot=True
        a = jnp.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
        lu_swap = DiaMatrix.from_dense(a, offsets=(-1, 0, 1)).lu_factor(pivot=True)
        assert lu_swap.perm is not None
        assert lu_swap.perm.shape == (3,)
        assert lu_swap.perm.dtype == jnp.int32

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_axis1(self, mat):
        """solve(B, axis=1) solves each row of B as a separate RHS."""
        a, A = mat(6)
        lu = A.lu_factor()
        rng = np.random.default_rng(11)
        X_true = jnp.array(rng.standard_normal((4, 6)))  # (batch, n)
        B = jnp.array(a) @ X_true.T  # (n, batch)
        B_T = B.T  # (batch, n)
        X_hat = lu.solve(B_T, axis=1)
        assert X_hat.shape == (4, 6)
        assert jnp.allclose(X_hat, X_true, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_3d_axis(self, mat):
        """solve on a 3-D array along a non-zero axis."""
        a, A = mat(5)
        lu = A.lu_factor()
        rng = np.random.default_rng(22)
        X_true = jnp.array(rng.standard_normal((3, 5, 4)))  # (a, n, b)
        # Build RHS: apply A along axis 1
        B = A.matvec(X_true, axis=1)
        X_hat = lu.solve(B, axis=1)
        assert X_hat.shape == (3, 5, 4)
        assert jnp.allclose(X_hat, X_true, atol=1e-5)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_axis_matches_matvec(self, mat):
        """lu.solve(A.matvec(X, axis=k), axis=k) == X for any axis."""
        a, A = mat(5)
        lu = A.lu_factor()
        rng = np.random.default_rng(33)
        for ax in (0, 1, 2):
            shape = [3, 5, 4]
            shape[ax] = 5
            X = jnp.array(rng.standard_normal(shape))
            B = A.matvec(X, axis=ax)
            X_hat = lu.solve(B, axis=ax)
            assert X_hat.shape == tuple(shape)
            assert jnp.allclose(X_hat, X, atol=1e-5), f"failed for axis={ax}"


class TestGetRow:
    def test_first_row(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_row(0), jnp.array(a[0]))

    def test_last_row(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_row(4), jnp.array(a[4]))

    def test_interior_row(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_row(2), jnp.array(a[2]))

    def test_all_rows_match_todense(self):
        a, A = _tridiag(6)
        dense = A.todense()
        for i in range(6):
            assert jnp.allclose(A.get_row(i), dense[i]), f"row {i} mismatch"

    def test_pentadiag(self):
        a, A = _pentadiag(7)
        dense = A.todense()
        for i in range(7):
            assert jnp.allclose(A.get_row(i), dense[i]), f"row {i} mismatch"

    def test_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        dense = A.todense()
        for i in range(4):
            assert jnp.allclose(A.get_row(i), dense[i]), f"row {i} mismatch"

    def test_unstored_diagonal_is_zero(self):
        # Only main diagonal stored; every row should be a one-hot vector.
        A = diags([jnp.arange(1, 5, dtype=jnp.float32)], offsets=(0,))
        for i in range(4):
            row = A.get_row(i)
            expected = jnp.zeros(4).at[i].set(float(i + 1))
            assert jnp.allclose(row, expected), f"row {i} mismatch"

    def test_jit_compatible(self):
        import jax

        _, A = _tridiag(5)
        row_jit = jax.jit(A.get_row)(2)
        assert jnp.allclose(row_jit, A.get_row(2))

    def test_vmap_reconstructs_matrix(self):
        import jax

        _, A = _tridiag(5)
        all_rows = jax.vmap(A.get_row)(jnp.arange(5))
        assert jnp.allclose(all_rows, A.todense())


class TestGetColumn:
    def test_first_column(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_column(0), jnp.array(a[:, 0]))

    def test_last_column(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_column(4), jnp.array(a[:, 4]))

    def test_interior_column(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_column(2), jnp.array(a[:, 2]))

    def test_all_columns_match_todense(self):
        a, A = _tridiag(6)
        dense = A.todense()
        for j in range(6):
            assert jnp.allclose(A.get_column(j), dense[:, j]), f"col {j} mismatch"

    def test_pentadiag(self):
        a, A = _pentadiag(7)
        dense = A.todense()
        for j in range(7):
            assert jnp.allclose(A.get_column(j), dense[:, j]), f"col {j} mismatch"

    def test_nonsquare(self):
        A = diags(
            [jnp.ones(3), -2 * jnp.ones(4), jnp.ones(4)],
            offsets=(-1, 0, 1),
            shape=(4, 6),
        )
        dense = A.todense()
        for j in range(6):
            assert jnp.allclose(A.get_column(j), dense[:, j]), f"col {j} mismatch"

    def test_unstored_diagonal_is_zero(self):
        # Only main diagonal stored; every column should be a one-hot vector.
        A = diags([jnp.arange(1, 5, dtype=jnp.float32)], offsets=(0,))
        for j in range(4):
            col = A.get_column(j)
            expected = jnp.zeros(4).at[j].set(float(j + 1))
            assert jnp.allclose(col, expected), f"col {j} mismatch"

    def test_jit_compatible(self):
        import jax

        _, A = _tridiag(5)
        col_jit = jax.jit(A.get_column)(2)
        assert jnp.allclose(col_jit, A.get_column(2))

    def test_vmap_reconstructs_matrix(self):
        import jax

        _, A = _tridiag(5)
        # vmap over columns produces (m, n); transpose to (n, m) for comparison
        all_cols = jax.vmap(A.get_column)(jnp.arange(5))
        assert jnp.allclose(all_cols.T, A.todense())
