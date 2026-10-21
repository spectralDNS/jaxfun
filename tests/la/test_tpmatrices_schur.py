"""Shared-factor merging, the eigenvector-conditioning guard of the
diagonalization solver, and the generalized Schur (QZ) solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.galerkin import CartesianProduct, FunctionSpace, Legendre, TensorProduct
from jaxfun.la import BlockMatrix, DiaMatrix, Matrix, TPMatrices, TPMatrix
from jaxfun.la.matrixprotocol import SolverNotApplicable
from jaxfun.la.tpmatrix import (
    TPMatricesLUFactors,
    TPMatricesSchurSolver,
    tpmats_lu_factor,
    tpmats_schur_factor,
)
from jaxfun.utils.common import ulp


def _dia(a: np.ndarray) -> DiaMatrix:
    return DiaMatrix.from_dense(jnp.asarray(a), tol=0)


def _stiffness(n: int) -> np.ndarray:
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def _mass(n: int) -> np.ndarray:
    return 4 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)


def _first_derivative(n: int) -> np.ndarray:
    # A shifted centred difference. Paired with `_mass` its eigenvectors are
    # nearly parallel, and no diagonal scaling makes them less so: a bidiagonal
    # or nonsymmetric tridiagonal would not do, being diagonally similar to a
    # well-conditioned matrix.
    return 4 * np.eye(n) + np.eye(n, k=1) - np.eye(n, k=-1)


def _advection_diffusion(n0: int, n1: int, dense_axis1: bool = False) -> TPMatrices:
    """``K⊗M + B⊗D + c B⊗M`` with ``c`` folded into a matrix, as `inner` does."""
    wrap = (lambda a: Matrix(jnp.asarray(a))) if dense_axis1 else _dia
    K, B = _dia(_stiffness(n0)), _dia(_mass(n0))
    M, D = wrap(_mass(n1)), wrap(_first_derivative(n1))
    return TPMatrices(
        [
            TPMatrix([K, M], 1),
            TPMatrix([B, D], 1),
            TPMatrix([_dia(0.5 * _mass(n0)), M], 1),
        ]
    )


def _residual(A: TPMatrices, uh, b) -> float:
    return float(jnp.linalg.norm(A(uh) - b) / jnp.linalg.norm(b))


def test_merge_folds_scaled_shared_factor():
    A = _advection_diffusion(6, 7)
    merged = A.merge()
    assert len(merged) == 2
    assert all(isinstance(m, DiaMatrix) for tp in merged.tpmats for m in tp.mats)
    u = jnp.asarray(np.random.default_rng(0).standard_normal(A.tpmats[0].shape[1]))
    u = u.reshape(6, 7)
    err = jnp.linalg.norm(merged(u) - A(u)) / jnp.linalg.norm(A(u))
    assert float(err) < float(ulp(100))


def test_merge_is_skipped_for_traced_values():
    n = 6
    K, M = _dia(_stiffness(n)), _dia(_mass(n))
    u = jnp.ones((n, n))

    @jax.jit
    def apply(c):
        A = TPMatrices([TPMatrix([K, M], c), TPMatrix([K, M.scale(2.0)], 1)])
        assert A.merge() is A
        return A(u)

    A = TPMatrices([TPMatrix([K, M], 1.0), TPMatrix([K, M.scale(2.0)], 1)])
    assert len(A.merge()) == 1
    assert jnp.allclose(apply(1.0), A.merge()(u))


def test_block_matrix_merges_within_blocks_only():
    n = 6
    D = FunctionSpace(
        n + 2, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}}
    )
    T = TensorProduct(D, D)
    W = CartesianProduct(T, T)
    K, M = _dia(_stiffness(n)), _dia(_mass(n))
    # The first two share M on axis 1 and merge; the third has the first's
    # factors, but in another block.
    terms = [
        TPMatrix([K, M], 1, (0, 0)),
        TPMatrix([M.scale(0.5), M], 1, (0, 0)),
        TPMatrix([K, M], 1, (1, 0)),
    ]
    B = BlockMatrix(terms, W, W)
    assert sorted(tp.global_indices for tp in B.mats) == [(0, 0), (1, 0)]
    K0, M0 = _stiffness(n), _mass(n)
    ref = np.zeros((2 * n * n, 2 * n * n))
    ref[: n * n, : n * n] = np.kron(K0, M0) + 0.5 * np.kron(M0, M0)
    ref[n * n :, : n * n] = np.kron(K0, M0)
    assert np.allclose(np.asarray(B.todense()), ref)


def test_merge_returns_the_operator_when_nothing_merges():
    n = 6
    K, M = _dia(_stiffness(n)), _dia(_mass(n))
    A = TPMatrices([TPMatrix([K, M], 1), TPMatrix([M, K], 1)])
    assert A.merge() is A


def test_lu_factor_merges_a_scaled_mass():
    # K⊗M + M⊗K + c M⊗M with c folded into the matrix has three distinct
    # matrices per axis until merged.
    n = 8
    K, M = _dia(_stiffness(n)), _dia(_mass(n))
    A = TPMatrices(
        [TPMatrix([K, M], 1), TPMatrix([M, K], 1), TPMatrix([M.scale(10.0), M], 1)]
    )
    lu = tpmats_lu_factor(list(A.tpmats))
    assert isinstance(lu, TPMatricesLUFactors)
    b = jnp.ones((n, n))
    assert _residual(A, lu.solve(b), b) < float(ulp(100))


def _anisotropic_laplace(n: int) -> TPMatrices:
    """-u_xx - 2 u_yy - 3 u_zz, with the constants folded into separately built
    mass matrices, as `inner` does. No two terms share all but one factor."""
    K0, M0 = _stiffness(n), _mass(n)
    return TPMatrices(
        [
            TPMatrix([_dia(K0), _dia(M0), _dia(M0)], 1),
            TPMatrix([_dia(2 * M0), _dia(K0), _dia(M0)], 1),
            TPMatrix([_dia(M0), _dia(3 * M0), _dia(K0)], 1),
        ]
    )


def test_distinct_factors_counts_matrices_up_to_a_scalar():
    A = _anisotropic_laplace(6)
    form = A.distinct_factors()
    assert len(form.matrices) == 2  # K and M, shared by all three axes
    assert form.indices == [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert np.allclose(np.asarray(form.scales, dtype=float), [1, 2, 3])
    rebuilt = TPMatrices(
        [
            TPMatrix([form.matrices[i] for i in idx], s)
            for idx, s in zip(form.indices, form.scales, strict=True)
        ]
    )
    u = jnp.asarray(np.random.default_rng(3).standard_normal((6, 6, 6)))
    err = jnp.linalg.norm(rebuilt(u) - A(u)) / jnp.linalg.norm(A(u))
    assert float(err) < float(ulp(100))


def test_lu_factor_counts_factors_up_to_a_scalar():
    # M, 2M and 3M on an axis count once; merging cannot help here.
    n = 6
    A = _anisotropic_laplace(n)
    assert A.merge() is A
    lu = tpmats_lu_factor(list(A.tpmats))
    b = jnp.asarray(np.random.default_rng(2).standard_normal((n, n, n)))
    uh = np.asarray(lu.solve(b)).ravel()
    K0, M0 = _stiffness(n), _mass(n)
    dense = (
        np.kron(np.kron(K0, M0), M0)
        + 2 * np.kron(np.kron(M0, K0), M0)
        + 3 * np.kron(np.kron(M0, M0), K0)
    )
    ref = np.linalg.solve(dense, np.asarray(b).ravel())
    assert np.linalg.norm(uh - ref) / np.linalg.norm(ref) < float(ulp(100))


def test_lu_factor_accepts_a_scaled_symmetric_definite_pencil():
    # A basis scaling, as `scaling=n+1` applies, spreads the norms of the
    # M-orthonormal eigenvectors and inflates cond(V); the pencil stays normal
    # in the M inner product and diagonalizes accurately.
    n = 12
    s = 1.0 / np.arange(1, n + 1) ** 3
    S = np.diag(s)
    K0, M0 = _stiffness(n), _mass(n)
    A = TPMatrices(
        [
            TPMatrix([_dia(S @ K0 @ S), _dia(S @ M0 @ S)], 1),
            TPMatrix([_dia(S @ M0 @ S), _dia(S @ K0 @ S)], 1),
        ]
    )
    lu = tpmats_lu_factor(list(A.tpmats))
    assert np.linalg.cond(np.asarray(lu.eigvecs[0])) > 1e3
    # The scaled solution, scaled back, solves the unscaled system.
    c = np.ones((n, n))
    u = s[:, None] * np.asarray(lu.solve(jnp.asarray(s[:, None] * c * s))) * s
    u0 = np.linalg.solve(np.kron(K0, M0) + np.kron(M0, K0), c.ravel()).reshape(n, n)
    assert np.linalg.norm(u - u0) / np.linalg.norm(u0) < float(ulp(1000))


def test_lu_factor_rejects_ill_conditioned_eigenvectors():
    A = _advection_diffusion(8, 10)
    with pytest.raises(SolverNotApplicable, match="condition number"):
        tpmats_lu_factor(list(A.tpmats))


@pytest.mark.parametrize(
    "shape, dense_axis1, schur_axis",
    [((12, 8), False, 1), ((8, 12), False, 0), ((8, 12), True, 1)],
    ids=["schur-on-shorter-axis-1", "schur-on-shorter-axis-0", "schur-on-dense-axis"],
)
def test_schur_solver(shape, dense_axis1, schur_axis):
    A = _advection_diffusion(*shape, dense_axis1=dense_axis1)
    solver = tpmats_schur_factor(A)
    assert solver.schur_axis == schur_axis
    b = jnp.asarray(np.random.default_rng(1).standard_normal(shape))
    uh = solver.solve(b)
    assert uh.dtype == b.dtype
    assert _residual(A, uh, b) < float(ulp(100))


def test_lu_factor_falls_back_to_schur():
    A = _advection_diffusion(8, 10)
    assert isinstance(A.lu_factor(), TPMatricesSchurSolver)
    b = jnp.ones((8, 10))
    assert _residual(A, A.solve(b, method="lu"), b) < float(ulp(100))


def test_schur_rejects_more_than_two_terms():
    # Biharmonic-like: no two terms share a factor on either axis.
    n = 6
    B, K, F = _dia(_mass(n)), _dia(_stiffness(n)), _dia(_stiffness(n) @ _stiffness(n))
    A = TPMatrices([TPMatrix([B, F], 1), TPMatrix([K, K], 2), TPMatrix([F, B], 1)])
    with pytest.raises(SolverNotApplicable, match="reduces to 3"):
        tpmats_schur_factor(A)
