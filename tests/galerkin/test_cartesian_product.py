"""Tests for CartesianProductSpace and related infrastructure.

Covers the heterogeneous (rank=-1) Cartesian product space, BlockArray
arithmetic, and the nested mixed-system assembly path used by Stokes-like
problems.

API notes
---------
* When component spaces use different polynomial degrees (e.g. Stokes
  velocity N vs. pressure N-2), explicit num_quad_points must be passed to
  inner(), evaluate_mesh, and backward so all components use the same mesh.
"""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.cartesianproductspace import (
    CartesianProduct,
    CartesianProductSpace,
    CartesianTensorProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.la import BlockArray, BlockMatrix, GlobalArray
from jaxfun.utils.common import ulp

pytestmark = pytest.mark.integration

N = 8  # polynomial degree used throughout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_component_space(N: int):
    """W = CartesianProduct(T_cheb, T_leg) — two scalar TPS with different N."""
    T0 = TensorProduct(Chebyshev.Chebyshev(N), Chebyshev.Chebyshev(N))
    T1 = TensorProduct(Legendre.Legendre(N - 2), Legendre.Legendre(N - 2))
    return CartesianProduct(T0, T1, name="W"), T0, T1


def _same_n_two_component_space(N: int):
    """W = CartesianProduct(T_cheb, T_leg) — both components use the same N.

    Use this when calling evaluate_mesh without explicit N, so that all
    components share the same quadrature grid.
    """
    T0 = TensorProduct(Chebyshev.Chebyshev(N), Chebyshev.Chebyshev(N))
    T1 = TensorProduct(Legendre.Legendre(N), Legendre.Legendre(N))
    return CartesianProduct(T0, T1, name="W"), T0, T1


def _stokes_like_space(N: int):
    """W = CartesianProduct(V, Q) — rank-1 velocity block V plus scalar pressure Q."""
    T = TensorProduct(Chebyshev.Chebyshev(N), Chebyshev.Chebyshev(N))
    Q = TensorProduct(Legendre.Legendre(N - 2), Legendre.Legendre(N - 2))
    V = CartesianProduct(T, T, name="V", rank=1)
    W = CartesianProduct(V, Q, name="W")
    return W, V, Q


# ---------------------------------------------------------------------------
# 1. Space structure
# ---------------------------------------------------------------------------


def test_heterogeneous_space_properties():
    W, T0, T1 = _two_component_space(N)
    assert isinstance(W, CartesianTensorProductSpace)
    assert not isinstance(W, VectorTensorProductSpace)
    assert W.rank == -1
    assert W.dims == 2
    assert W.num_components == 2
    assert W.dim == T0.dim + T1.dim
    assert W.block_sizes == (T0.dim, T1.dim)
    assert W.num_dofs == (T0.num_dofs, T1.num_dofs)


def test_heterogeneous_space_flatten():
    W, T0, T1 = _two_component_space(N)
    flat = W.flatten()
    assert len(flat) == 2
    assert flat[0].dim == T0.dim
    assert flat[1].dim == T1.dim


def test_nested_cartesian_product_structure():
    """CartesianProduct(V, Q) where V is itself a CartesianProductSpace."""
    W, V, Q = _stokes_like_space(N)
    assert W.num_components == 3  # two velocity + one pressure
    assert W.dim == V.dim + Q.dim
    flat = W.flatten()
    assert len(flat) == 3
    for i, space in enumerate(flat):
        assert space.global_index == i


def test_rank_one_is_vector_tps():
    T = TensorProduct(Chebyshev.Chebyshev(N), Chebyshev.Chebyshev(N))
    V = CartesianProduct(T, T, name="V", rank=1)
    assert isinstance(V, VectorTensorProductSpace)


def test_rank_minus_one_is_not_vector_tps():
    W, _, _ = _two_component_space(N)
    assert not isinstance(W, VectorTensorProductSpace)


# ---------------------------------------------------------------------------
# 2. BlockArray
# ---------------------------------------------------------------------------


def test_block_array_from_tuple():
    W, T0, T1 = _two_component_space(N)
    a0 = jnp.ones(T0.num_dofs)
    a1 = jnp.ones(T1.num_dofs) * 2.0
    ba = BlockArray(W, tuple_array=(a0, a1))
    assert jnp.allclose(ba[0], a0)
    assert jnp.allclose(ba[1], a1)
    assert ba.shape == (W.dim,)


def test_block_array_flat_roundtrip_two_components():
    """Flat → BlockArray → flat preserves data for two-component space."""
    W, T0, T1 = _two_component_space(N)
    a0 = jnp.arange(T0.dim, dtype=float).reshape(T0.num_dofs)
    a1 = jnp.arange(T1.dim, dtype=float).reshape(T1.num_dofs) + T0.dim
    flat = BlockArray(W, tuple_array=(a0, a1)).flatten()
    assert flat.shape == (W.dim,)
    ba_rt = BlockArray(W, flat_array=flat)
    assert jnp.allclose(ba_rt[0], a0)
    assert jnp.allclose(ba_rt[1], a1)


def test_block_array_flat_roundtrip_three_components():
    """Three-component roundtrip verifies the s0 += s1 cumulative-offset fix."""
    W, V, Q = _stokes_like_space(N)
    flat_spaces = W.flatten()
    blocks = [
        jnp.arange(s.dim, dtype=float).reshape(s.num_dofs) * (i + 1)
        for i, s in enumerate(flat_spaces)
    ]
    flat = BlockArray(W, tuple_array=tuple(blocks)).flatten()
    assert flat.shape == (W.dim,)
    ba_rt = BlockArray(W, flat_array=flat)
    for i, b in enumerate(blocks):
        assert jnp.allclose(ba_rt[i], b), f"Block {i} mismatch after flat roundtrip"


def test_block_array_arithmetic():
    W, T0, T1 = _two_component_space(N)
    a0, a1 = jnp.ones(T0.num_dofs), jnp.ones(T1.num_dofs) * 2.0
    b0, b1 = jnp.ones(T0.num_dofs) * 3.0, jnp.ones(T1.num_dofs) * 4.0
    A = BlockArray(W, tuple_array=(a0, a1))
    B = BlockArray(W, tuple_array=(b0, b1))

    C = A + B
    assert jnp.allclose(C[0], a0 + b0) and jnp.allclose(C[1], a1 + b1)
    D = A - B
    assert jnp.allclose(D[0], a0 - b0) and jnp.allclose(D[1], a1 - b1)
    assert jnp.allclose((-A)[0], -a0) and jnp.allclose((-A)[1], -a1)
    assert jnp.allclose((3.0 * A)[0], 3 * a0) and jnp.allclose((A * 3.0)[1], 3 * a1)


def test_block_array_accumulate():
    W, T0, T1 = _two_component_space(N)
    ba = BlockArray(W)
    ba.accumulate(GlobalArray(0, jnp.ones(T0.num_dofs)))
    ba.accumulate(GlobalArray(0, jnp.ones(T0.num_dofs)))
    ba.accumulate(GlobalArray(1, jnp.ones(T1.num_dofs) * 5.0))
    assert jnp.allclose(ba[0], 2.0)
    assert jnp.allclose(ba[1], 5.0)


# ---------------------------------------------------------------------------
# 3. TrialFunction / TestFunction unpacking
# ---------------------------------------------------------------------------


def test_trialfunction_testfunction_unpack_two_components():
    W, T0, T1 = _two_component_space(N)
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")
    assert u.functionspace.dim == T0.dim
    assert p.functionspace.dim == T1.dim
    assert v.functionspace.dim == T0.dim
    assert q.functionspace.dim == T1.dim


def test_trialfunction_testfunction_unpack_nested():
    """Unpacking from a nested CartesianProductSpace (Stokes-like)."""
    W, V, Q = _stokes_like_space(N)
    u_vec, p = TrialFunction(W, name="up")
    assert u_vec.functionspace.dim == V.dim
    assert p.functionspace.dim == Q.dim


# ---------------------------------------------------------------------------
# 4. inner() over heterogeneous CartesianProductSpace
#
# Each inner() call handles one (test_block, trial_block) coupling.
# Combine the resulting BlockMatrix objects with +.
# ---------------------------------------------------------------------------


def test_inner_per_block_returns_blockmatrix():
    """Each individual inner() call over a component of W returns a BlockMatrix."""
    W, T0, T1 = _two_component_space(N)
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")

    A0 = inner(v * u, sparse=True, kind="bilinear", num_quad_points=(N, N))
    A1 = inner(q * p, sparse=True, kind="bilinear", num_quad_points=(N, N))

    assert isinstance(A0, BlockMatrix)
    assert isinstance(A1, BlockMatrix)
    assert A0.shape == (W.dim, W.dim)
    assert A1.shape == (W.dim, W.dim)


def test_inner_block_sum_is_block_diagonal():
    """Adding per-block matrices gives a block-diagonal BlockMatrix."""
    W, T0, T1 = _two_component_space(N)
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")

    A = inner(v * u, sparse=True, kind="bilinear", num_quad_points=(N, N)) + inner(
        q * p, sparse=True, kind="bilinear", num_quad_points=(N, N)
    )
    assert isinstance(A, BlockMatrix)
    dense = A.todense()
    assert dense.shape == (W.dim, W.dim)
    s0 = T0.dim
    # Off-diagonal coupling blocks must be zero for decoupled L2 mass
    assert jnp.allclose(dense[:s0, s0:], 0.0, atol=1e-12)
    assert jnp.allclose(dense[s0:, :s0], 0.0, atol=1e-12)


def test_inner_system_heterogeneous_solve():
    """Solve two decoupled L2 projections assembled as a single block system.

    Each block is assembled separately, combined via +, then solved.
    The result must match independent single-block solves for each component.
    """
    W, T0, T1 = _two_component_space(N)
    x, y = W.system.base_scalars()
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")

    ue = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    pe = sp.cos(sp.pi * x) * sp.cos(sp.pi * y)

    A0, b0 = inner(v * (u - ue), sparse=True, kind="system", num_quad_points=(N, N))
    A1, b1 = inner(q * (p - pe), sparse=True, kind="system", num_quad_points=(N, N))
    A = A0 + A1
    b = cast(BlockArray, b0) + cast(BlockArray, b1)

    assert isinstance(A, BlockMatrix)
    assert isinstance(b, BlockArray)

    x_joint = A.solve(b)
    assert isinstance(x_joint, BlockArray)

    # Compare against independent single-block solves
    u0 = TrialFunction(T0)
    v0 = TestFunction(T0)
    A_ref0, b_ref0 = inner(v0 * (u0 - ue), sparse=True, kind="system")
    x_ref0 = A_ref0.solve(b_ref0)

    v1 = TestFunction(T1)
    A_ref1, b_ref1 = inner(v1 * (p - pe), sparse=True, kind="system")
    x_ref1 = A_ref1.solve(b_ref1)

    assert jnp.linalg.norm(x_joint[0] - x_ref0) < jnp.sqrt(ulp(1000))
    assert jnp.linalg.norm(x_joint[1] - x_ref1) < jnp.sqrt(ulp(1000))


# ---------------------------------------------------------------------------
# 5. evaluate / evaluate_mesh
#
# Both act on the same mesh for all components.  When components have
# different polynomial degrees, use _same_n_two_component_space (or pass
# explicit N) so the outputs all share the same shape and can be stacked.
# ---------------------------------------------------------------------------


def test_evaluate_stacks_components():
    """evaluate() stacks component outputs into a single Array."""
    W, T0, T1 = _two_component_space(N)
    rng = np.random.default_rng(42)
    c0 = jnp.array(rng.standard_normal(T0.num_dofs))
    c1 = jnp.array(rng.standard_normal(T1.num_dofs))

    x_pts = jnp.linspace(-0.9, 0.9, 5)
    x_eval = jnp.stack(jnp.meshgrid(x_pts, x_pts), axis=-1).reshape(-1, 2)

    result = W.evaluate(x_eval, (c0, c1))
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2, x_eval.shape[0])
    assert jnp.allclose(result[0], T0.evaluate(x_eval, c0))
    assert jnp.allclose(result[1], T1.evaluate(x_eval, c1))


def test_evaluate_mesh_stacks_components():
    """evaluate_mesh() stacks component outputs into a single Array.

    Uses same-N components so all share the same quadrature grid size
    without needing to pass explicit N.
    """
    W, T0, T1 = _same_n_two_component_space(N)
    rng = np.random.default_rng(7)
    c0 = jnp.array(rng.standard_normal(T0.num_dofs))
    c1 = jnp.array(rng.standard_normal(T1.num_dofs))

    result = W.evaluate_mesh((c0, c1))
    assert isinstance(result, jnp.ndarray)
    assert result.shape[0] == 2
    assert jnp.allclose(result[0], T0.evaluate_mesh(c0))
    assert jnp.allclose(result[1], T1.evaluate_mesh(c1))


# ---------------------------------------------------------------------------
# 6. forward / backward / scalar_product
# ---------------------------------------------------------------------------


def test_forward_backward_roundtrip():
    """forward followed by backward recovers the original physical arrays."""
    W, T0, T1 = _same_n_two_component_space(N)
    rng = np.random.default_rng(13)
    u0 = jnp.array(rng.standard_normal(T0.num_quad_points))
    u1 = jnp.array(rng.standard_normal(T1.num_quad_points))

    u_stacked = jnp.stack([u0, u1])
    coeffs = W.forward(u_stacked)
    assert len(coeffs) == 2
    ua = W.backward(coeffs)
    assert isinstance(ua, jnp.ndarray)
    assert ua.shape == u_stacked.shape
    assert jnp.linalg.norm(ua[0] - u0) < jnp.sqrt(ulp(100))
    assert jnp.linalg.norm(ua[1] - u1) < jnp.sqrt(ulp(100))


# ===========================================================================
# 1D CartesianProductSpace and BlockMatrix
# ===========================================================================
#
# These tests cover CartesianProduct of OrthogonalSpace (or DirectSum)
# components, which produces a CartesianProductSpace (dims=1), and the
# corresponding BlockMatrix assembled by inner().
#
# API note: like the ND case, each inner() call handles one
# (test_block, trial_block) coupling; combine results with +.
# ===========================================================================


def _1d_two_component_space(N: int):
    """C = CartesianProduct(L0, L1) — two Legendre spaces of different sizes."""
    L0 = Legendre.Legendre(N)
    L1 = Legendre.Legendre(N - 2)
    return CartesianProduct(L0, L1, name="C"), L0, L1


def _1d_two_component_space_with_bcs(N: int):
    """C = CartesianProduct(D, S) — Legendre with and without Dirichlet BCs."""
    D = FunctionSpace(N, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    S = FunctionSpace(N, Legendre.Legendre)
    return CartesianProduct(D, S, name="C"), D, S


# ---------------------------------------------------------------------------
# 7. 1D Space structure
# ---------------------------------------------------------------------------


def test_1d_cartesian_product_space_properties():
    C, L0, L1 = _1d_two_component_space(N)
    assert isinstance(C, CartesianProductSpace)
    assert not isinstance(C, CartesianTensorProductSpace)
    assert C.dims == 1
    assert C.rank == -1
    assert C.num_components == 2
    assert C.dim == L0.dim + L1.dim
    assert C.block_sizes == (L0.dim, L1.dim)


def test_1d_cartesian_product_flatten_and_global_index():
    C, _, _ = _1d_two_component_space(N)
    flat = C.flatten()
    assert len(flat) == 2
    assert all(isinstance(s, OrthogonalSpace) for s in flat)
    s0, s1 = flat
    assert isinstance(s0, OrthogonalSpace) and isinstance(s1, OrthogonalSpace)
    assert s0.global_index == 0
    assert s1.global_index == 1
    assert s0.leaf is C
    assert s1.leaf is C


def test_1d_cartesian_product_num_dofs_normalized():
    """num_dofs wraps bare int from OrthogonalSpace into a 1-tuple."""
    C, L0, L1 = _1d_two_component_space(N)
    nd = C.num_dofs
    assert nd == ((L0.num_dofs,), (L1.num_dofs,))


# ---------------------------------------------------------------------------
# 8. 1D BlockArray
# ---------------------------------------------------------------------------


def test_1d_block_array_from_flat_roundtrip():
    C, L0, L1 = _1d_two_component_space(N)
    x0 = jnp.arange(L0.dim, dtype=float)
    x1 = jnp.arange(L1.dim, dtype=float) + L0.dim
    flat = BlockArray(C, tuple_array=(x0, x1)).flatten()
    ba = BlockArray(C, flat_array=flat)
    assert jnp.allclose(ba[0], x0)
    assert jnp.allclose(ba[1], x1)


def test_1d_block_array_arithmetic():
    C, L0, L1 = _1d_two_component_space(N)
    a = BlockArray(C, tuple_array=(jnp.ones(L0.dim), jnp.ones(L1.dim) * 2.0))
    b = BlockArray(C, tuple_array=(jnp.ones(L0.dim) * 3.0, jnp.ones(L1.dim) * 4.0))
    assert jnp.allclose((a + b)[0], 4.0) and jnp.allclose((a + b)[1], 6.0)
    assert jnp.allclose((a - b)[0], -2.0) and jnp.allclose((a - b)[1], -2.0)
    assert jnp.allclose((2.0 * a)[0], 2.0) and jnp.allclose((a * 2.0)[1], 4.0)
    assert jnp.allclose((-a)[0], -1.0)


# ---------------------------------------------------------------------------
# 9. 1D TrialFunction / TestFunction unpacking
# ---------------------------------------------------------------------------


def test_1d_trial_test_function_unpack():
    C, L0, L1 = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    assert u.functionspace.dim == L0.dim
    assert p.functionspace.dim == L1.dim
    assert v.functionspace.dim == L0.dim
    assert q.functionspace.dim == L1.dim


def test_1d_component_global_index_after_unpack():
    """After unpacking, each component's global_index matches its block position."""
    C, _, _ = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    fs_u = u.functionspace
    fs_p = p.functionspace
    assert isinstance(fs_u, OrthogonalSpace | DirectSum)
    assert isinstance(fs_p, OrthogonalSpace | DirectSum)
    assert fs_u.global_index == 0
    assert fs_p.global_index == 1


# ---------------------------------------------------------------------------
# 10. inner() over 1D CartesianProductSpace → BlockMatrix
# ---------------------------------------------------------------------------


def test_1d_inner_bilinear_returns_block_matrix():
    C, _, _ = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    A0 = inner(v * u, kind="bilinear")
    A1 = inner(q * p, kind="bilinear")
    assert isinstance(A0, BlockMatrix)
    assert isinstance(A1, BlockMatrix)
    assert A0.shape == (C.dim, C.dim)
    assert A1.shape == (C.dim, C.dim)


def test_1d_inner_block_sum_is_block_diagonal():
    """Block-diagonal L2 mass matrix has zero off-diagonal coupling blocks."""
    C, L0, _ = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    A = inner(v * u, kind="bilinear") + inner(q * p, kind="bilinear")
    assert isinstance(A, BlockMatrix)
    dense = A.todense()
    assert dense.shape == (C.dim, C.dim)
    s0 = L0.dim
    assert jnp.allclose(dense[:s0, s0:], 0.0, atol=1e-12)
    assert jnp.allclose(dense[s0:, :s0], 0.0, atol=1e-12)


def test_1d_inner_linear_block_routing():
    """inner(x*q) places the result in block 1 (q is the block-1 test function)."""
    C, _, _ = _1d_two_component_space(N)
    _, q = TestFunction(C, name="vq")
    x = C.system.x
    b = cast(BlockArray, inner(x * q, kind="linear"))
    assert jnp.linalg.norm(b[0]) < ulp(100), "block 0 should be zero"
    assert jnp.linalg.norm(b[1]) > ulp(100), "block 1 should be nonzero"


def test_1d_inner_linear_block_routing_first_block():
    """inner(x*v) places the result in block 0 (v is the block-0 test function)."""
    C, _, _ = _1d_two_component_space(N)
    v, _ = TestFunction(C, name="vq")
    x = C.system.x
    b = cast(BlockArray, inner(x * v, kind="linear"))
    assert jnp.linalg.norm(b[0]) > ulp(100), "block 0 should be nonzero"
    assert jnp.linalg.norm(b[1]) < ulp(100), "block 1 should be zero"


# ---------------------------------------------------------------------------
# 11. BlockMatrix linear algebra
# ---------------------------------------------------------------------------


def test_1d_block_matrix_matvec():
    C, L0, L1 = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    A = inner(v * u, kind="bilinear") + inner(q * p, kind="bilinear")
    ones = BlockArray(C, flat_array=jnp.ones(C.dim))
    result = A @ ones
    assert isinstance(result, BlockArray)
    assert result[0].shape == (L0.dim,)
    assert result[1].shape == (L1.dim,)


def test_1d_block_matrix_scale():
    C, _, _ = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    A = inner(v * u, kind="bilinear") + inner(q * p, kind="bilinear")
    assert isinstance(A, BlockMatrix)
    A2 = A.scale(2.0)
    assert isinstance(A2, BlockMatrix)
    assert jnp.allclose(A2.todense(), 2.0 * A.todense(), atol=ulp(100))


def test_1d_block_matrix_add_sub():
    C, _, _ = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    A = inner(v * u, kind="bilinear") + inner(q * p, kind="bilinear")
    assert isinstance(A, BlockMatrix)
    A2 = A + A
    assert jnp.allclose(A2.todense(), 2.0 * A.todense(), atol=ulp(100))
    A0 = A2 - A
    assert jnp.allclose(A0.todense(), A.todense(), atol=ulp(100))


def test_1d_block_matrix_solve_residual():
    """Solve A x = b and verify ‖A x - b‖ < tol."""
    C, _, _ = _1d_two_component_space(N)
    u, p = TrialFunction(C, name="up")
    v, q = TestFunction(C, name="vq")
    x = C.system.x
    A = inner(v * u, kind="bilinear") + inner(q * p, kind="bilinear")
    assert isinstance(A, BlockMatrix)
    b0 = cast(BlockArray, inner(x * v, kind="linear"))
    b1 = cast(BlockArray, inner(x * q, kind="linear"))
    b = b0 + b1
    uh = A.solve(b)
    assert isinstance(uh, BlockArray)
    res = A @ uh
    assert jnp.linalg.norm(res.flatten() - b.flatten()) < ulp(100)


# ---------------------------------------------------------------------------
# 12. Coupled 1D system — mixed formulation
#
# Poisson in first-order form:  s = u',  -s' = f
#   Test with v:  (s, v) = (u', v)  →  inner(s*v) = inner(u.diff(x)*v)
#   Test with q: -(s', q) = (f, q)  →  inner(s.diff(x)*q) = -inner(f*q)
# Manufactured solution: u_e(x) = sin(πx), domain [-1,1].
# ---------------------------------------------------------------------------


def test_1d_coupled_system_solve():
    """Mixed Poisson assembled over a 1D CartesianProductSpace.

    System (first-order form): s' = u'',  u' = s.
    Manufactured solution: u_e = sin(πx), BCs homogeneous Dirichlet.
    Mirrors the coupled1D.py example.
    """
    C, D, _ = _1d_two_component_space_with_bcs(N)
    u, s = TrialFunction(C, name="us")
    v, q = TestFunction(C, name="vq")
    x = C.system.x
    ue = sp.sin(sp.pi * x)

    A, a = inner(s.diff(x) * v - ue.diff(x, 2) * v, kind="system", sparse=True)
    B = inner(u.diff(x) * q - s * q, kind="bilinear", sparse=True)

    H = A + B
    h = cast(BlockArray, a)

    uh = H.solve(h, method="banded", pivot=True)
    assert isinstance(uh, BlockArray)

    # Residual check
    assert jnp.linalg.norm((H @ uh).flatten() - h.flatten()) < ulp(100)

    # Solution accuracy: evaluate u (block 0) on quadrature mesh
    xj = D.mesh()
    u_phys = C.backward(uh)[0]
    ue_phys = jnp.sin(jnp.pi * xj)
    assert jnp.linalg.norm(u_phys - ue_phys) < jnp.sqrt(ulp(100))
