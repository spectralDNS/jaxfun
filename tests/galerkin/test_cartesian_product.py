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
    VectorTensorProductSpace,
)
from jaxfun.la import BlockArray, BlockTPMatrix, IndexedArray
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
    assert isinstance(W, CartesianProductSpace)
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
    ba.accumulate(IndexedArray(0, jnp.ones(T0.num_dofs)))
    ba.accumulate(IndexedArray(0, jnp.ones(T0.num_dofs)))
    ba.accumulate(IndexedArray(1, jnp.ones(T1.num_dofs) * 5.0))
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
# Combine the resulting BlockTPMatrix objects with +.
# ---------------------------------------------------------------------------


def test_inner_per_block_returns_blocktpmatrix():
    """Each individual inner() call over a component of W returns a BlockTPMatrix."""
    W, T0, T1 = _two_component_space(N)
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")

    A0 = inner(v * u, sparse=True, kind="bilinear", num_quad_points=(N, N))
    A1 = inner(q * p, sparse=True, kind="bilinear", num_quad_points=(N, N))

    assert isinstance(A0, BlockTPMatrix)
    assert isinstance(A1, BlockTPMatrix)
    assert A0.shape == (W.dim, W.dim)
    assert A1.shape == (W.dim, W.dim)


def test_inner_block_sum_is_block_diagonal():
    """Adding per-block matrices gives a block-diagonal BlockTPMatrix."""
    W, T0, T1 = _two_component_space(N)
    u, p = TrialFunction(W, name="up")
    v, q = TestFunction(W, name="vq")

    A = inner(v * u, sparse=True, kind="bilinear", num_quad_points=(N, N)) + inner(
        q * p, sparse=True, kind="bilinear", num_quad_points=(N, N)
    )
    assert isinstance(A, BlockTPMatrix)
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

    assert isinstance(A, BlockTPMatrix)
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
