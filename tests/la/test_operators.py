import jax.numpy as jnp
import pytest

from jaxfun.la import (
    DiaMatrix,
    IdentityMatrix,
    Matrix,
    TensorMatrix,
    TPMatrices,
    TPMatrix,
    ZeroMatrix,
    diags,
)


def test_identity_matrix_preserves_state_shape() -> None:
    identity = IdentityMatrix((2, 3))
    x = jnp.arange(6.0).reshape((2, 3))

    assert identity.shape == (6, 6)
    assert identity.is_diagonal
    assert identity.diagonal_or_none().shape == x.shape
    assert jnp.allclose(identity @ x, x)
    assert jnp.allclose(identity.solve(x), x)


def test_zero_matrix_preserves_state_shape_and_rejects_solve() -> None:
    zero = ZeroMatrix((2, 3))
    x = jnp.arange(6.0).reshape((2, 3))

    assert zero.shape == (6, 6)
    assert zero.is_zero
    assert zero.is_diagonal
    assert zero.diagonal_or_none().shape == x.shape
    assert jnp.allclose(zero @ x, jnp.zeros_like(x))
    with pytest.raises(ValueError, match="zero operator"):
        zero.solve(x)


def test_identity_and_diamatrix_arithmetic_preserves_diamatrix_structure() -> None:
    identity = IdentityMatrix(3)
    laplace = diags(
        [jnp.ones(2), -2 * jnp.ones(3), jnp.ones(2)],
        offsets=(-1, 0, 1),
    )

    system = identity - 0.1 * laplace

    assert isinstance(system, DiaMatrix)
    expected = jnp.eye(3) - 0.1 * laplace.todense()
    assert jnp.allclose(system.todense(), expected)


def test_diamatrix_and_matrix_arithmetic_promotes_to_matrix() -> None:
    dense = Matrix(jnp.eye(3))
    sparse = diags([jnp.array([1.0, 2.0, 3.0])], offsets=(0,))

    system = sparse + dense

    assert isinstance(system, Matrix)
    assert jnp.allclose(system.todense(), sparse.todense() + dense.todense())


def test_tensor_product_arithmetic_preserves_tpmatrices_structure() -> None:
    d0 = diags([jnp.array([1.0, 2.0])], offsets=(0,))
    d1 = diags([jnp.array([3.0, 4.0, 5.0])], offsets=(0,))
    tp = TPMatrix([d0, d1], scale=2.0)

    system = 1.5 * tp - 0.5 * tp

    assert isinstance(system, TPMatrices)
    expected = tp.todense()
    assert jnp.allclose(system.todense(), expected)


def test_tensormatrix_arithmetic_preserves_tensormatrix_structure() -> None:
    data = jnp.arange(16.0).reshape((2, 2, 2, 2))
    tensor = TensorMatrix(data)

    system = 1.5 * tensor - 0.5 * tensor

    assert isinstance(system, TensorMatrix)
    assert jnp.allclose(system.data, data)


def test_matrix_arithmetic_with_identity_solves_diagonal_without_lu() -> None:
    identity = IdentityMatrix((2, 2))
    dense_diagonal = Matrix(jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0])))
    system = identity + 2.0 * dense_diagonal
    rhs = jnp.ones((2, 2))

    expected_diag = jnp.array([[3.0, 5.0], [7.0, 9.0]])
    assert isinstance(system, Matrix)
    assert jnp.allclose(system.diagonal_or_none().reshape((2, 2)), expected_diag)
    assert jnp.allclose(system.solve(rhs), rhs / expected_diag)
    assert not hasattr(system, "_lu_cache")


def test_matrix_solve_uses_cached_diagonal_without_lu() -> None:
    matrix = Matrix(jnp.diag(jnp.array([1.0, 2.0, 4.0])))
    rhs = jnp.array([2.0, 4.0, 8.0])

    assert matrix.diagonal_or_none() is not None
    assert jnp.allclose(matrix.solve(rhs), jnp.array([2.0, 2.0, 2.0]))
    assert not hasattr(matrix, "_lu_cache")


def test_diamatrix_solve_diagonal_without_lu() -> None:
    matrix = diags([jnp.array([1.0, 2.0, 4.0])], offsets=(0,))
    rhs = jnp.array([2.0, 4.0, 8.0])

    assert jnp.allclose(matrix.solve(rhs), jnp.array([2.0, 2.0, 2.0]))
    assert not hasattr(matrix, "_lu_cache")


def test_tpmatrix_solve_diagonal_without_factor_lu() -> None:
    d0 = diags([jnp.array([1.0, 2.0])], offsets=(0,))
    d1 = diags([jnp.array([3.0, 4.0])], offsets=(0,))
    system = TPMatrix([d0, d1], scale=2.0)
    rhs = jnp.ones((2, 2))

    assert jnp.allclose(system.solve(rhs), rhs / system.diagonal_or_none())
    assert not hasattr(d0, "_lu_cache")
    assert not hasattr(d1, "_lu_cache")
