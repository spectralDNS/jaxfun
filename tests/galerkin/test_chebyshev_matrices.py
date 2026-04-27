"""Unit tests for jaxfun.galerkin.Chebyshev.matrices.

Checks all five supported (i, j) combinations as well as unsupported ones,
verifying shape, DiaMatrix type, transpose symmetry, and correctness of
entries against brute-force numerical quadrature.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxfun.galerkin import TestFunction, TrialFunction, inner
from jaxfun.galerkin.Chebyshev import Chebyshev, matrices
from jaxfun.la import DiaMatrix
from jaxfun.utils.common import ulp


def _cheb(N: int) -> Chebyshev:
    return Chebyshev(N)


def _quad_matrix(space: Chebyshev, dv: int, du: int) -> jnp.ndarray:
    """Compute (v.N x u.N) integral matrix numerically via Gauss-Chebyshev quad.

    Entry (i, j) = integral_{-1}^{1} T_i^{(dv)}(x) * T_j^{(du)}(x) dx
    weighted by the Chebyshev weight 1/sqrt(1-x^2) (baked into quad weights).
    """
    v = TestFunction(space)
    u = TrialFunction(space)
    x = space.system.x
    A = inner(v.diff(x, dv) * u.diff(x, du), use_precomputed_matrices=False)
    return A.todense()


class TestMatricesReturnType:
    @pytest.mark.parametrize("ij", [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)])
    def test_returns_dia_matrix(self, ij):
        i, j = ij
        v = _cheb(8)
        M = matrices((v, i), (v, j))
        assert isinstance(M, DiaMatrix), (
            f"expected DiaMatrix for ({i},{j}), got {type(M)}"
        )

    @pytest.mark.parametrize("ij", [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)])
    def test_shape_square(self, ij):
        i, j = ij
        N = 8
        v = _cheb(N)
        M = matrices((v, i), (v, j))
        assert M is not None
        assert M.shape == (N, N)

    @pytest.mark.parametrize("ij", [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)])
    def test_shape_rectangular(self, ij):
        i, j = ij
        Nv, Nu = 6, 10
        v, u = _cheb(Nv), _cheb(Nu)
        M = matrices((v, i), (u, j))
        assert M is not None
        assert M.shape == (Nv, Nu)

    @pytest.mark.parametrize("ij", [(1, 1), (1, 2), (2, 1), (3, 0), (0, 3)])
    def test_unsupported_returns_none(self, ij):
        i, j = ij
        v = _cheb(8)
        result = matrices((v, i), (v, j))
        assert result is None, f"expected None for ({i},{j}), got {result}"


class TestMassMatrix:
    def test_diagonal_only(self):
        N = 8
        v = _cheb(N)
        M = matrices((v, 0), (v, 0))
        assert M is not None
        assert set(M.offsets) == {0}

    def test_entries_match_norm_squared(self):
        N = 8
        v = _cheb(N)
        M = matrices((v, 0), (v, 0))
        assert M is not None
        expected = v.norm_squared()
        assert jnp.allclose(M.diagonal(0), expected, atol=ulp(10))

    def test_rectangular_mass(self):
        Nv, Nu = 5, 9
        v, u = _cheb(Nv), _cheb(Nu)
        M = matrices((v, 0), (u, 0))
        assert M is not None
        # Should be a square Nv×Nu matrix; off-diagonal entries zero for ortho basis
        assert M.shape == (Nv, Nu)
        assert jnp.allclose(M.diagonal(0), v.norm_squared(), atol=ulp(10))


class TestTransposeSymmetry:
    """matrices((v,j),(u,0)).T == matrices((u,0),(v,j)) for j in {1,2}."""

    @pytest.mark.parametrize("j", [1, 2])
    def test_transpose_relation_square(self, j):
        N = 8
        v, u = _cheb(N), _cheb(N)
        M_fwd = matrices((v, 0), (u, j))
        M_bwd = matrices((u, j), (v, 0))
        assert M_fwd is not None and M_bwd is not None
        assert jnp.allclose(M_fwd.todense(), M_bwd.T.todense(), atol=ulp(10))

    @pytest.mark.parametrize("j", [1, 2])
    def test_transpose_relation_rectangular(self, j):
        Nv, Nu = 6, 10
        v, u = _cheb(Nv), _cheb(Nu)
        M_fwd = matrices((v, 0), (u, j))
        M_bwd = matrices((u, j), (v, 0))
        assert M_fwd is not None and M_bwd is not None
        assert jnp.allclose(M_fwd.todense(), M_bwd.T.todense(), atol=ulp(10))


class TestFirstDerivativeMatrix:
    """(0,1) matrix entries verified by numerical quadrature."""

    def test_offsets_are_odd(self):
        N = 9
        v = _cheb(N)
        M = matrices((v, 0), (v, 1))
        assert M is not None
        for off in M.offsets:
            assert off % 2 == 1, f"expected odd offset, got {off}"

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_values_match_quadrature(self, N):
        v = _cheb(N)
        M = matrices((v, 0), (v, 1))
        assert M is not None
        ref = _quad_matrix(v, 0, 1)
        assert jnp.allclose(M.todense(), jnp.array(ref), atol=ulp(1000)), (
            f"N={N}: max err {float(jnp.abs(M.todense() - jnp.array(ref)).max())}"
        )

    def test_first_column_zero(self):
        """T_0' = 0, so the first column should be all zeros."""
        N = 8
        v = _cheb(N)
        M = matrices((v, 0), (v, 1))
        assert M is not None
        assert jnp.allclose(M.get_column(0), jnp.zeros(N), atol=ulp(10))


class TestSecondDerivativeMatrix:
    """(0,2) matrix entries verified by numerical quadrature."""

    def test_offsets_are_even_positive(self):
        N = 9
        v = _cheb(N)
        M = matrices((v, 0), (v, 2))
        assert M is not None
        for off in M.offsets:
            assert off % 2 == 0 and off >= 0, f"unexpected offset {off}"

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_values_match_quadrature(self, N):
        v = _cheb(N)
        M = matrices((v, 0), (v, 2))
        assert M is not None
        ref = _quad_matrix(v, 0, 2)
        err = float(jnp.linalg.norm(M.todense() - ref))
        scale = float(jnp.linalg.norm(M.todense()))
        assert err / max(scale, 1.0) < ulp(100), (
            f"N={N}: relative err {err / scale:.2e}"
        )

    def test_first_two_columns_zero(self):
        """T_0'' = T_1'' = 0, so columns 0 and 1 should be all zeros."""
        N = 8
        v = _cheb(N)
        M = matrices((v, 0), (v, 2))
        assert M is not None
        assert jnp.allclose(M.get_column(0), jnp.zeros(N), atol=ulp(10))
        assert jnp.allclose(M.get_column(1), jnp.zeros(N), atol=ulp(10))
