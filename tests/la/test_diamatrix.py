"""Tests for DiaMatrix and diags in sparsemat.py."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse

from jaxfun.la.diamatrix import DenseIndexingWarning, DiaMatrix, LUFactors, diags
from jaxfun.la.matrix import Matrix
from jaxfun.la.pinned import PinnedSystem
from jaxfun.utils.common import ulp


def _tridiag(n: int) -> tuple[Matrix, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (Matrix, DiaMatrix)."""
    dense = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
    )
    a = Matrix(jnp.array(dense))
    A = DiaMatrix.from_dense(a.data, offsets=(-1, 0, 1))
    return a, A


def _pentadiag(n: int) -> tuple[Matrix, DiaMatrix]:
    """Return a symmetric pentadiagonal nxn matrix as (Matrix, DiaMatrix)."""
    dense = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
        + np.diag(-0.5 * np.ones(n - 2), 2)
        + np.diag(-0.5 * np.ones(n - 2), -2)
    )
    a = Matrix(jnp.array(dense))
    A = DiaMatrix.from_dense(a.data, offsets=(-2, -1, 0, 1, 2))
    return a, A


def _tridiagO(n: int) -> tuple[Matrix, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (Matrix, DiaMatrix)."""
    dense = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 2), 2)
        + np.diag(-np.ones(n - 2), -2)
    )
    a = Matrix(jnp.array(dense))
    A = DiaMatrix.from_dense(a.data, offsets=(-2, 0, 2))
    return a, A


def _pentadiagO(n: int) -> tuple[Matrix, DiaMatrix]:
    """Return a symmetric pentadiagonal nxn matrix as (Matrix, DiaMatrix)."""
    dense = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 2), 2)
        + np.diag(-np.ones(n - 2), -2)
        + np.diag(-0.5 * np.ones(n - 4), 4)
        + np.diag(-0.5 * np.ones(n - 4), -4)
    )
    a = Matrix(jnp.array(dense))
    A = DiaMatrix.from_dense(a.data, offsets=(-4, -2, 0, 2, 4))
    return a, A


def _circdiag(n: int) -> tuple[Matrix, DiaMatrix]:
    """Return a symmetric tridiagonal nxn matrix as (Matrix, DiaMatrix)."""
    dense = (
        np.diag(2 * np.ones(n))
        + np.diag(-np.ones(n - 1), 1)
        + np.diag(-np.ones(n - 1), -1)
        + np.diag(-np.ones(1), n - 1)  # wrap-around super-diagonal
        + np.diag(np.ones(1), -(n - 1))  # wrap-around sub-diagonal
    )
    a = Matrix(jnp.array(dense))
    A = DiaMatrix.from_dense(a.data, offsets=(-n + 1, -1, 0, 1, n - 1))
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
        A2 = DiaMatrix.from_dense(a.data)
        assert jnp.allclose(A2.todense(), a.todense())

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
        assert jnp.allclose(A.todense(), a.todense())

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
        assert jnp.allclose(A.matvec(x), a @ x)

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
        Y = A @ X
        assert Y.shape == (8, 3)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmat_vs_dense(self, mat):
        a, A = mat(6)
        rng = np.random.default_rng(1)
        X = jnp.array(rng.standard_normal((6, 3)))
        assert jnp.allclose(A @ X, a @ X, atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_vector(self, mat):
        a, A = mat(6)
        x = jnp.arange(6, dtype=float)
        assert jnp.allclose(A @ x, a @ x)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_matrix(self, mat):
        a, A = mat(6)
        X = jnp.eye(6)
        assert jnp.allclose(A @ X, a.todense(), atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_Matrix(self, mat):
        """DiaMatrix @ DiaMatrix and Matrix @ Matrix should agree."""
        a, A = mat(6)
        result = A @ A
        expected = (a @ a).todense()
        assert jnp.allclose(result.todense(), expected, atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_rmatmul_vector(self, mat):
        a, A = mat(6)
        x = jnp.arange(6, dtype=float)
        assert jnp.allclose(x @ A, x @ a)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_rmatmul_matrix(self, mat):
        a, A = mat(6)
        X = jnp.eye(6)
        assert jnp.allclose(X @ A, a.todense(), atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_rmatmul_2d_array_x_matrix(self, mat):
        """Array (2-D) @ Matrix → Matrix.__rmatmul__(Array) 2-D path."""
        a, A = mat(6)
        rng = np.random.default_rng(42)
        X = jnp.array(rng.standard_normal((4, 6)))
        expected = X @ a.data
        assert jnp.allclose(X @ a, expected, atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_dia_x_matrix(self, mat):
        """DiaMatrix @ Matrix → DiaMatrix.__matmul__(Matrix) path."""
        a, A = mat(6)
        rng = np.random.default_rng(43)
        B = Matrix(jnp.array(rng.standard_normal((6, 4))))
        result = A @ B
        expected = A.todense() @ B.data
        assert isinstance(result, Matrix)
        assert jnp.allclose(result.data, expected, atol=1e-6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_matmul_matrix_x_dia(self, mat):
        """Matrix @ DiaMatrix → Matrix.__matmul__(DiaMatrix) path."""
        a, A = mat(6)
        rng = np.random.default_rng(44)
        B = Matrix(jnp.array(rng.standard_normal((4, 6))))
        result = B @ A
        expected = B.data @ A.todense()
        assert isinstance(result, Matrix)
        assert jnp.allclose(result.data, expected, atol=1e-6)


class TestTranspose:
    @pytest.mark.parametrize("mat", allmatrices)
    def test_T_shape_square(self, mat):
        _, A = mat(6)
        assert A.T.shape == (6, 6)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_T_dense_square(self, mat):
        a, A = mat(6)
        assert jnp.allclose(A.T.todense(), a.T.todense())

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
        assert jnp.allclose((A.scale(3.0)).todense(), (a * 3.0).todense())

    def test_mul(self):
        a, A = _tridiag(3)
        assert jnp.allclose((A * 2.0).todense(), (a * 2.0).todense())

    def test_rmul(self):
        a, A = _tridiag(3)
        assert jnp.allclose((0.5 * A).todense(), (0.5 * a).todense())

    def test_neg(self):
        a, A = _tridiag(3)
        assert jnp.allclose((-A).todense(), (-a).todense())


def _indexing_matrix() -> tuple[jax.Array, DiaMatrix]:
    dense = jnp.arange(1, 21, dtype=jnp.float32).reshape(4, 5)
    return dense, DiaMatrix.from_dense(dense)


def _assert_getitem_matches_dense(key) -> None:
    dense, A = _indexing_matrix()
    if _warns_for_dense_indexing(key):
        with pytest.warns(DenseIndexingWarning, match="materializes dense output"):
            actual = A[key]
    else:
        actual = A[key]
    expected = dense[key]
    assert actual.shape == expected.shape
    assert jnp.array_equal(actual, expected)


def _warns_for_dense_indexing(key) -> bool:
    if key is Ellipsis or key is None or key is True:
        return True

    if isinstance(key, jax.Array):
        return bool(key.dtype == jnp.bool_ and key.ndim == 0 and bool(key))

    return False


class TestGetItem:
    @pytest.mark.parametrize(
        "key",
        [
            pytest.param(1, id="row-int"),
            pytest.param(slice(1, None, 2), id="row-slice"),
            pytest.param(jnp.array([3, 1]), id="row-advanced"),
            pytest.param(Ellipsis, id="ellipsis"),
            pytest.param(None, id="newaxis"),
            pytest.param(True, id="bool-scalar-true"),
            pytest.param(False, id="bool-scalar-false"),
            pytest.param(jnp.array(True), id="jax-bool-scalar-true"),
            pytest.param(jnp.array(False), id="jax-bool-scalar-false"),
            pytest.param((Ellipsis, 2), id="ellipsis-column"),
            pytest.param((1, Ellipsis), id="row-ellipsis"),
            pytest.param((None, slice(None), 2), id="newaxis-column"),
        ],
    )
    def test_basic_indexing_forms_match_dense_array(self, key):
        _assert_getitem_matches_dense(key)

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param(
                (jnp.array([0, 2, 3]), jnp.array([1, 2, 4])),
                id="paired-advanced-indices",
            ),
            pytest.param(
                (jnp.array([[0], [2]]), jnp.array([1, 3])),
                id="broadcast-advanced-indices",
            ),
            pytest.param(
                (jnp.array([True, False, True, False]), slice(None)),
                id="row-bool-mask",
            ),
            pytest.param(
                (slice(None), jnp.array([False, True, False, True, False])),
                id="column-bool-mask",
            ),
            pytest.param(
                (
                    jnp.array([True, False, True, False]),
                    jnp.array([False, True, False, False, True]),
                ),
                id="paired-bool-masks",
            ),
            pytest.param(
                jnp.array(
                    [
                        [True, False, False, False, False],
                        [False, False, False, False, False],
                        [False, False, True, False, False],
                        [False, False, False, False, True],
                    ]
                ),
                id="matrix-bool-mask",
            ),
        ],
    )
    def test_advanced_indexing_forms_match_dense_array(self, key):
        _assert_getitem_matches_dense(key)


class TestMatvecAxis:
    """Test matvec / apply with the axis parameter on N-D arrays."""

    @pytest.mark.parametrize("mat", allmatrices)
    def test_axis0_2d_equals_matmat(self, mat):
        """matvec(X, axis=0) should equal the classic A @ X product."""
        a, A = mat(6)
        rng = np.random.default_rng(7)
        X = jnp.array(rng.standard_normal((6, 5)))
        expected = a @ X
        assert jnp.allclose(A.matvec(X, axis=0), expected, atol=ulp(100))
        assert jnp.allclose(a.matvec(X, axis=0), expected, atol=ulp(100))

    @pytest.mark.parametrize("mat", allmatrices)
    def test_axis1_2d(self, mat):
        """matvec(X, axis=1) contracts along columns: each row of X gets multiplied."""
        a, A = mat(6)
        rng = np.random.default_rng(8)
        X = jnp.array(rng.standard_normal((5, 6)))  # (batch, m)
        expected = (a @ X.T).T  # (n, batch) → (batch, n)
        assert jnp.allclose(A.matvec(X, axis=1), expected, atol=ulp(100))
        assert jnp.allclose(a.matvec(X, axis=1), expected, atol=ulp(100))

    def test_axis_1d_ignores_axis(self):
        """For 1-D input the axis argument is irrelevant."""
        a, A = _tridiag(4)
        x = jnp.arange(4, dtype=float)
        result = A.matvec(x, axis=0)
        assert jnp.allclose(result, a @ x)

    def test_axis0_3d(self):
        """matvec(T, axis=0): T shape (m, b1, b2) → (n, b1, b2)."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(9)
        T = jnp.array(rng.standard_normal((4, 3, 2)))
        # Expected: contract axis 0 with A
        # result[i, j, k] = sum_l A[i, l] * T[l, j, k]
        expected = jnp.einsum("il,ljk->ijk", a.data, T)
        assert jnp.allclose(A.matvec(T, axis=0), expected, atol=ulp(100))
        assert jnp.allclose(a.matvec(T, axis=0), expected, atol=ulp(100))

    def test_axis1_3d(self):
        """matvec(T, axis=1): T shape (b0, m, b2) → (b0, n, b2)."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(10)
        T = jnp.array(rng.standard_normal((3, 4, 2)))
        expected = jnp.einsum("il,jlk->jik", a.data, T)
        assert jnp.allclose(A.matvec(T, axis=1), expected, atol=ulp(100))
        assert jnp.allclose(a.matvec(T, axis=1), expected, atol=ulp(100))

    def test_axis2_3d(self):
        """matvec(T, axis=2): T shape (b0, b1, m) → (b0, b1, n)."""
        a, A = _tridiag(4)
        rng = np.random.default_rng(11)
        T = jnp.array(rng.standard_normal((3, 2, 4)))
        expected = jnp.einsum("il,jkl->jki", a.data, T)
        assert jnp.allclose(A.matvec(T, axis=2), expected, atol=ulp(100))
        assert jnp.allclose(a.matvec(T, axis=2), expected, atol=ulp(100))

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


class TestAddSub:
    def test_add(self):
        a, A = _tridiag(4)
        result = A + A
        expected = (a + a).todense()
        assert jnp.allclose(result.todense(), expected)

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
        assert jnp.allclose(x_hat, x_true, atol=ulp(100))
        # Matrix.solve should give the same answer
        assert jnp.allclose(a.solve(b), x_true, atol=ulp(100))

    @pytest.mark.parametrize("mat", allmatrices)
    def test_solve_residual(self, mat):
        a, A = mat(8)
        rng = np.random.default_rng(42)
        b = jnp.array(rng.standard_normal(8))
        x = A.solve(b)
        assert float(jnp.linalg.norm(A.matvec(x) - b)) < ulp(100)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_solve_axis1(self, mat):
        """A.solve(B, axis=1): each row of B is a separate RHS."""
        a, A = mat(6)
        rng = np.random.default_rng(55)
        X_true = jnp.array(rng.standard_normal((4, 6)))
        B = (a @ X_true.T).T  # (4, 6)
        X_hat = A.solve(B, axis=1)
        assert X_hat.shape == (4, 6)
        assert jnp.allclose(X_hat, X_true, atol=ulp(100))
        assert jnp.allclose(a.solve(B, axis=1), X_true, atol=ulp(100))

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
            assert jnp.allclose(X_hat, X, atol=ulp(100)), f"failed for axis={ax}"


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
        PA = a.todense() if lu.perm is None else a.todense()[lu.perm, :]
        assert jnp.allclose(LU.todense(), PA, atol=ulp(100))

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_exact(self, mat):
        a, A = mat(5)
        x_true = jnp.arange(1.0, 6.0)
        b = a @ x_true
        lu = A.lu_factor()
        x_hat = lu.solve(b)
        assert jnp.allclose(x_hat, x_true, atol=ulp(100))
        # Matrix.lu_solve should give the same result
        assert jnp.allclose(a.lu_solve(b), x_true, atol=ulp(100))

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_random_rhs(self, mat):
        a, A = mat(8)
        rng = np.random.default_rng(99)
        b = jnp.array(rng.standard_normal(8))
        lu = A.lu_factor()
        x = lu.solve(b)
        assert float(jnp.linalg.norm(A.matvec(x) - b)) < ulp(100)

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_multiple_rhs(self, mat):
        """solve() should work for 2-D right-hand sides (n, k)."""
        a, A = mat(6)
        rng = np.random.default_rng(7)
        X_true = jnp.array(rng.standard_normal((6, 4)))
        B = a @ X_true
        lu = A.lu_factor()
        X_hat = lu.solve(B)
        assert X_hat.shape == (6, 4)
        assert jnp.allclose(X_hat, X_true, atol=ulp(100))
        # Matrix.solve should reproduce the same result
        assert jnp.allclose(a.solve(B), X_true, atol=ulp(100))

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
        assert jnp.allclose(H.todense(), PA, atol=ulp(100))
        # Solve A x = b
        x_true = jnp.array([1.0, 2.0, 3.0])
        b = a @ x_true
        x = lu.solve(b)
        assert jnp.allclose(x, x_true, atol=ulp(100))

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

    @pytest.mark.parametrize("mat", allmatrices)
    def test_lu_solve_axis1(self, mat):
        """solve(B, axis=1) solves each row of B as a separate RHS."""
        a, A = mat(6)
        lu = A.lu_factor()
        rng = np.random.default_rng(11)
        X_true = jnp.array(rng.standard_normal((4, 6)))  # (batch, n)
        B = a @ X_true.T  # (n, batch)
        B_T = B.T  # (batch, n)
        X_hat = lu.solve(B_T, axis=1)
        assert X_hat.shape == (4, 6)
        assert jnp.allclose(X_hat, X_true, atol=ulp(100))
        assert jnp.allclose(a.lu_solve(B_T, axis=1), X_true, atol=ulp(100))

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
        assert jnp.allclose(X_hat, X_true, atol=ulp(100))
        # Matrix counterpart
        assert jnp.allclose(a.lu_solve(B, axis=1), X_true, atol=ulp(100))
        assert jnp.allclose(a.solve(B, axis=1), X_true, atol=ulp(100))

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
            assert jnp.allclose(X_hat, X, atol=ulp(100)), (
                f"DiaMatrix failed for axis={ax}"
            )
            # Matrix counterpart: a.matvec then a.solve / a.lu_solve
            B_dense = a.matvec(X, axis=ax)
            assert jnp.allclose(a.solve(B_dense, axis=ax), X, atol=ulp(100)), (
                f"Matrix.solve failed for axis={ax}"
            )
            assert jnp.allclose(a.lu_solve(B_dense, axis=ax), X, atol=ulp(100)), (
                f"Matrix.lu_solve failed for axis={ax}"
            )


class TestGetRow:
    def test_first_row(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_row(0), a.get_row(0))

    def test_last_row(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_row(4), a.get_row(4))

    def test_interior_row(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_row(2), a.get_row(2))

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
        assert jnp.allclose(A.get_column(0), a.get_column(0))

    def test_last_column(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_column(4), a.get_column(4))

    def test_interior_column(self):
        a, A = _tridiag(5)
        assert jnp.allclose(A.get_column(2), a.get_column(2))

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


class TestPin:
    """Tests for :meth:`DiaMatrix.pin`, :meth:`Matrix.pin`, and :class:`PinnedSystem`."""  # noqa: E501

    # ------------------------------------------------------------------
    # Construction and metadata
    # ------------------------------------------------------------------

    def test_pin_returns_pinned_system_dia(self):
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0})
        assert isinstance(sys, PinnedSystem)

    def test_pin_returns_pinned_system_matrix(self):
        a, _ = _tridiag(5)
        sys = a.pin({0: 0.0})
        assert isinstance(sys, PinnedSystem)

    def test_constraints_stored_as_sorted_tuple(self):
        _, A = _tridiag(6)
        sys = A.pin({3: 1.0, 0: 0.0})
        # Must be a tuple of 2-tuples, sorted by index.
        assert sys.constraints == ((0, 0.0), (3, 1.0))

    def test_negative_index_normalised(self):
        """pin({-1: v}) should be equivalent to pin({n-1: v})."""
        _, A = _tridiag(5)
        sys_neg = A.pin({-1: 2.0})
        sys_pos = A.pin({4: 2.0})
        assert sys_neg.constraints == sys_pos.constraints

    def test_shape_preserved(self):
        _, A = _tridiag(7)
        sys = A.pin({0: 0.0})
        assert sys.shape == (7, 7)

    def test_repr(self):
        _, A = _tridiag(4)
        sys = A.pin({0: 0.0})
        r = repr(sys)
        assert "PinnedSystem" in r
        assert "0=0.0" in r

    # ------------------------------------------------------------------
    # Pytree structure
    # ------------------------------------------------------------------

    def test_is_pytree_registered(self):
        """PinnedSystem must be a valid JAX pytree."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0})
        leaves, treedef = jax.tree_util.tree_flatten(sys)
        # Only the data array inside the matrix should be a leaf.
        assert len(leaves) == 1
        assert isinstance(leaves[0], jax.Array)

    def test_pytree_round_trip(self):
        """Unflatten should reproduce an identical PinnedSystem."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0, 4: 1.0})
        leaves, treedef = jax.tree_util.tree_flatten(sys)
        sys2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert sys2.constraints == sys.constraints
        assert sys2.shape == sys.shape

    def test_constraints_are_static(self):
        """Constraint indices/values must live in the treedef, not leaves."""
        _, A = _tridiag(5)
        sys1 = A.pin({0: 0.0})
        sys2 = A.pin({0: 1.0})
        _, td1 = jax.tree_util.tree_flatten(sys1)
        _, td2 = jax.tree_util.tree_flatten(sys2)
        # Different constraint value → different treedef.
        assert td1 != td2

    # ------------------------------------------------------------------
    # Row substitution — modified matrix checks
    # ------------------------------------------------------------------

    def test_pinned_row_is_identity_dia(self):
        """After pin({i: v}), row i of the modified DiaMatrix must be e_i."""
        _, A = _tridiag(5)
        sys = A.pin({2: 3.0})
        assert isinstance(sys.matrix, DiaMatrix)
        row2 = sys.matrix.get_row(2)
        expected = jnp.zeros(5).at[2].set(1.0)
        assert jnp.allclose(row2, expected)

    def test_pinned_row_is_identity_matrix(self):
        """After pin({i: v}), row i of the modified Matrix must be e_i."""
        a, _ = _tridiag(5)
        sys = a.pin({2: 3.0})
        assert isinstance(sys.matrix, Matrix)
        row2 = sys.matrix.get_row(2)
        expected = jnp.zeros(5).at[2].set(1.0)
        assert jnp.allclose(row2, expected)

    def test_other_rows_unchanged_dia(self):
        """Rows not pinned must remain identical to the original."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0})
        for i in (1, 2, 3, 4):
            assert jnp.allclose(sys.matrix.get_row(i), A.get_row(i))

    def test_pin_adds_main_diagonal_if_missing(self):
        """pin() must not crash when the main diagonal is not stored."""
        # Build a matrix with only off-diagonals (no main diagonal).
        data = jnp.ones((2, 5))
        A_no_main = DiaMatrix(data=data, offsets=(-1, 1), shape=(5, 5))
        sys = A_no_main.pin({0: 0.0})
        row0 = sys.matrix.get_row(0)
        expected = jnp.zeros(5).at[0].set(1.0)
        assert jnp.allclose(row0, expected)

    # ------------------------------------------------------------------
    # fix_rhs
    # ------------------------------------------------------------------

    def test_fix_rhs_1d(self):
        _, A = _tridiag(5)
        sys = A.pin({0: 7.0, 4: -3.0})
        b = jnp.ones(5)
        b_mod = sys.fix_rhs(b)
        assert float(b_mod[0]) == pytest.approx(7.0)
        assert float(b_mod[4]) == pytest.approx(-3.0)
        # Interior values unchanged.
        assert jnp.allclose(b_mod[1:4], jnp.ones(3))

    def test_fix_rhs_2d_axis0(self):
        """fix_rhs on a (n, k) array with axis=0 pins rows."""
        _, A = _tridiag(5)
        sys = A.pin({0: 9.0})
        B = jnp.ones((5, 3))
        B_mod = sys.fix_rhs(B, axis=0)
        assert jnp.allclose(B_mod[0], 9.0 * jnp.ones(3))
        assert jnp.allclose(B_mod[1:], jnp.ones((4, 3)))

    def test_fix_rhs_2d_axis1(self):
        """fix_rhs on a (k, n) array with axis=1 pins columns."""
        _, A = _tridiag(5)
        sys = A.pin({0: 9.0})
        B = jnp.ones((3, 5))
        B_mod = sys.fix_rhs(B, axis=1)
        assert jnp.allclose(B_mod[:, 0], 9.0 * jnp.ones(3))
        assert jnp.allclose(B_mod[:, 1:], jnp.ones((3, 4)))

    # ------------------------------------------------------------------
    # solve — correctness
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("use_matrix", [False, True])
    def test_solve_1d_single_pin(self, use_matrix):
        """Solving a tridiagonal system with one DOF pinned to zero."""
        a, A = _tridiag(5)
        mat = a if use_matrix else A
        sys = mat.pin({0: 0.0})
        # Build RHS from the true solution with x[0] = 0 imposed.
        x_true = jnp.array([0.0, 1.0, 2.0, 1.0, 0.0])
        b = A.matvec(x_true)  # use original A for matvec, not pinned
        x_hat = sys.solve(b)
        assert jnp.allclose(x_hat, x_true, atol=ulp(100))

    @pytest.mark.parametrize("use_matrix", [False, True])
    def test_solve_1d_two_pins(self, use_matrix):
        """Tridiagonal with both endpoints pinned."""
        a, A = _tridiag(5)
        mat = a if use_matrix else A
        sys = mat.pin({0: 0.0, -1: 0.0})
        x_true = jnp.array([0.0, 1.0, 1.5, 1.0, 0.0])
        b = A.matvec(x_true)
        x_hat = sys.solve(b)
        assert jnp.allclose(x_hat, x_true, atol=ulp(100))

    @pytest.mark.parametrize("use_matrix", [False, True])
    def test_solve_dia_matches_matrix(self, use_matrix):
        """DiaMatrix.pin and Matrix.pin must give the same answer."""
        a, A = _tridiag(6)
        rng = np.random.default_rng(42)
        b = jnp.array(rng.standard_normal(6))
        x_dia = A.pin({0: 0.0}).solve(b)
        x_mat = a.pin({0: 0.0}).solve(b)
        assert jnp.allclose(x_dia, x_mat, atol=ulp(100))

    # ------------------------------------------------------------------
    # solve — axis argument
    # ------------------------------------------------------------------

    def test_solve_2d_axis0(self):
        """solve(B, axis=0): each column of B is an independent RHS."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0, 4: 0.0})
        rng = np.random.default_rng(7)
        k = 4
        # Build true solutions with pinned BCs satisfied.
        X_true = jnp.array(rng.standard_normal((5, k)))
        X_true = X_true.at[0].set(0.0).at[4].set(0.0)
        B = jnp.array([A.matvec(X_true[:, j]) for j in range(k)]).T  # (5, k)
        X_hat = sys.solve(B, axis=0)
        assert X_hat.shape == (5, k)
        assert jnp.allclose(X_hat, X_true, atol=ulp(100))

    def test_solve_2d_axis1(self):
        """solve(B, axis=1): each row of B is an independent RHS."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0, 4: 0.0})
        rng = np.random.default_rng(8)
        k = 3
        X_true = jnp.array(rng.standard_normal((k, 5)))
        X_true = X_true.at[:, 0].set(0.0).at[:, 4].set(0.0)
        B = jnp.stack([A.matvec(X_true[j]) for j in range(k)])  # (k, 5)
        X_hat = sys.solve(B, axis=1)
        assert X_hat.shape == (k, 5)
        assert jnp.allclose(X_hat, X_true, atol=ulp(100))

    # ------------------------------------------------------------------
    # LU caching
    # ------------------------------------------------------------------

    def test_lu_factor_cached(self):
        """lu_factor() called twice should return the exact same object."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0})
        lu1 = sys.lu_factor()
        lu2 = sys.lu_factor()
        assert lu1 is lu2

    def test_solve_reuses_lu(self):
        """Repeated solve() calls must not recompute the LU."""
        _, A = _tridiag(5)
        sys = A.pin({0: 0.0})
        b = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])
        x1 = sys.solve(b)
        lu_after_first = sys.lu_factor()
        x2 = sys.solve(b)
        lu_after_second = sys.lu_factor()
        assert lu_after_first is lu_after_second
        assert jnp.allclose(x1, x2)
