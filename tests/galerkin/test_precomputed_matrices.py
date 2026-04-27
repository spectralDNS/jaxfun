"""Unit tests for precomputed matrices across all jaxfun polynomial spaces.

Checks supported (i, j) combinations, unsupported ones (expect None), shape,
DiaMatrix type, transpose symmetry, and correctness of entries against
brute-force numerical quadrature.

Spaces covered:
  * Chebyshev      -- (0,0), (0,1), (1,0), (0,2), (2,0)
  * Legendre       -- (0,0), (0,1), (1,0), (0,2), (2,0)
  * Fourier        -- any (i,j); always diagonal
  * ChebyshevU     -- (0,0) only
  * Ultraspherical -- (0,0) only
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxfun.galerkin import TestFunction, TrialFunction, inner
from jaxfun.galerkin.Chebyshev import Chebyshev, matrices as cheb_matrices
from jaxfun.galerkin.ChebyshevU import ChebyshevU, matrices as chebU_matrices
from jaxfun.galerkin.Fourier import Fourier, matrices as fourier_matrices
from jaxfun.galerkin.Legendre import Legendre, matrices as leg_matrices
from jaxfun.galerkin.Ultraspherical import Ultraspherical, matrices as ultra_matrices
from jaxfun.la import DiaMatrix
from jaxfun.utils.common import ulp

# ---------------------------------------------------------------------------
# Space factory helpers
# ---------------------------------------------------------------------------


def _cheb(N: int) -> Chebyshev:
    return Chebyshev(N)


def _leg(N: int) -> Legendre:
    return Legendre(N)


def _four(N: int) -> Fourier:
    return Fourier(N if N % 2 == 0 else N + 1)


def _chebu(N: int) -> ChebyshevU:
    return ChebyshevU(N)


def _ultra05(N: int) -> Ultraspherical:
    return Ultraspherical(N, lambda_=0.5)


def _quad_matrix(space, dv: int, du: int) -> jnp.ndarray:
    """Compute integral matrix numerically via the space's own quadrature.

    Entry (i, j) = integral phi_i^{(dv)} * phi_j^{(du)} dx (with weight).
    Uses inner() with use_precomputed_matrices=False to bypass any table lookup.
    """
    v = TestFunction(space)
    u = TrialFunction(space)
    x = space.system.x
    A = inner(v.diff(x, dv) * u.diff(x, du), use_precomputed_matrices=False)
    return A.todense()


# ---------------------------------------------------------------------------
# Parametrize sets
# ---------------------------------------------------------------------------

# Spaces that support (0,0), (0,1), (1,0), (0,2), (2,0)
_POLY5 = [
    pytest.param(_cheb, cheb_matrices, id="Chebyshev"),
    pytest.param(_leg, leg_matrices, id="Legendre"),
]

# Spaces that support (0,0) only
_MASS_ONLY = [
    pytest.param(_chebu, chebU_matrices, id="ChebyshevU"),
    pytest.param(_ultra05, ultra_matrices, id="Ultraspherical"),
]

_POLY5_SUPPORTED = [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)]
_POLY5_UNSUPPORTED = [(1, 1), (1, 2), (2, 1), (3, 0), (0, 3)]
_MASS_ONLY_UNSUPPORTED = [(0, 1), (1, 0), (0, 2), (2, 0)]


# ---------------------------------------------------------------------------
# Shared tests: Chebyshev and Legendre (five supported (i,j) pairs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space_fn,mats", _POLY5)
class TestPoly5ReturnType:
    @pytest.mark.parametrize("ij", _POLY5_SUPPORTED)
    def test_returns_dia_matrix(self, space_fn, mats, ij):
        i, j = ij
        v = space_fn(8)
        M = mats((v, i), (v, j))
        assert isinstance(M, DiaMatrix), (
            f"expected DiaMatrix for ({i},{j}), got {type(M)}"
        )

    @pytest.mark.parametrize("ij", _POLY5_SUPPORTED)
    def test_shape_square(self, space_fn, mats, ij):
        i, j = ij
        N = 8
        v = space_fn(N)
        M = mats((v, i), (v, j))
        assert M is not None
        assert M.shape == (N, N)

    @pytest.mark.parametrize("ij", _POLY5_SUPPORTED)
    def test_shape_rectangular(self, space_fn, mats, ij):
        i, j = ij
        Nv, Nu = 6, 10
        v, u = space_fn(Nv), space_fn(Nu)
        M = mats((v, i), (u, j))
        assert M is not None
        assert M.shape == (Nv, Nu)

    @pytest.mark.parametrize("ij", _POLY5_UNSUPPORTED)
    def test_unsupported_returns_none(self, space_fn, mats, ij):
        i, j = ij
        v = space_fn(8)
        assert mats((v, i), (v, j)) is None, f"expected None for ({i},{j})"


@pytest.mark.parametrize("space_fn,mats", _POLY5)
class TestPoly5MassMatrix:
    def test_diagonal_only(self, space_fn, mats):
        v = space_fn(8)
        M = mats((v, 0), (v, 0))
        assert M is not None
        assert set(M.offsets) == {0}

    def test_entries_match_norm_squared(self, space_fn, mats):
        v = space_fn(8)
        M = mats((v, 0), (v, 0))
        assert M is not None
        assert jnp.allclose(M.diagonal(0), v.norm_squared(), atol=ulp(10))

    def test_rectangular_entries(self, space_fn, mats):
        Nv, Nu = 5, 9
        v, u = space_fn(Nv), space_fn(Nu)
        M = mats((v, 0), (u, 0))
        assert M is not None
        assert M.shape == (Nv, Nu)
        assert jnp.allclose(M.diagonal(0), v.norm_squared(), atol=ulp(10))

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_values_match_quadrature(self, space_fn, mats, N):
        v = space_fn(N)
        M = mats((v, 0), (v, 0))
        assert M is not None
        ref = _quad_matrix(v, 0, 0)
        assert jnp.allclose(M.todense(), ref, atol=ulp(100)), (
            f"N={N}: max err {float(jnp.abs(M.todense() - ref).max())}"
        )


@pytest.mark.parametrize("space_fn,mats", _POLY5)
class TestPoly5TransposeSymmetry:
    """mats((v,0),(u,j)).T == mats((u,j),(v,0)) for j in {1, 2}."""

    @pytest.mark.parametrize("j", [1, 2])
    def test_transpose_square(self, space_fn, mats, j):
        N = 8
        v = space_fn(N)
        M_fwd = mats((v, 0), (v, j))
        M_bwd = mats((v, j), (v, 0))
        assert M_fwd is not None and M_bwd is not None
        assert jnp.allclose(M_fwd.todense(), M_bwd.T.todense(), atol=ulp(10))

    @pytest.mark.parametrize("j", [1, 2])
    def test_transpose_rectangular(self, space_fn, mats, j):
        Nv, Nu = 6, 10
        v, u = space_fn(Nv), space_fn(Nu)
        M_fwd = mats((v, 0), (u, j))
        M_bwd = mats((u, j), (v, 0))
        assert M_fwd is not None and M_bwd is not None
        assert jnp.allclose(M_fwd.todense(), M_bwd.T.todense(), atol=ulp(10))


@pytest.mark.parametrize("space_fn,mats", _POLY5)
class TestPoly5FirstDerivativeMatrix:
    """(0,1) matrix entries verified by numerical quadrature."""

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_values_match_quadrature(self, space_fn, mats, N):
        v = space_fn(N)
        M = mats((v, 0), (v, 1))
        assert M is not None
        ref = _quad_matrix(v, 0, 1)
        assert jnp.allclose(M.todense(), jnp.array(ref), atol=ulp(10000)), (
            f"N={N}: max err {float(jnp.abs(M.todense() - jnp.array(ref)).max())}"
        )

    def test_first_column_zero(self, space_fn, mats):
        """phi_0' = 0, so the first column should be all zeros."""
        N = 8
        v = space_fn(N)
        M = mats((v, 0), (v, 1))
        assert M is not None
        assert jnp.allclose(M.get_column(0), jnp.zeros(N), atol=ulp(10))


@pytest.mark.parametrize("space_fn,mats", _POLY5)
class TestPoly5SecondDerivativeMatrix:
    """(0,2) matrix entries verified by numerical quadrature."""

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_values_match_quadrature(self, space_fn, mats, N):
        v = space_fn(N)
        M = mats((v, 0), (v, 2))
        assert M is not None
        ref = _quad_matrix(v, 0, 2)
        err = float(jnp.linalg.norm(M.todense() - ref))
        scale = float(jnp.linalg.norm(M.todense()))
        assert err / max(scale, 1.0) < ulp(100), (
            f"N={N}: relative err {err / scale:.2e}"
        )

    def test_first_two_columns_zero(self, space_fn, mats):
        """phi_0'' = phi_1'' = 0, so columns 0 and 1 should be all zeros."""
        N = 8
        v = space_fn(N)
        M = mats((v, 0), (v, 2))
        assert M is not None
        assert jnp.allclose(M.get_column(0), jnp.zeros(N), atol=ulp(10))
        assert jnp.allclose(M.get_column(1), jnp.zeros(N), atol=ulp(10))


# ---------------------------------------------------------------------------
# Chebyshev-specific: sparsity structure of derivative matrices
# ---------------------------------------------------------------------------


class TestChebyshevOffsets:
    """Chebyshev derivative matrices have characteristic offset patterns."""

    def test_first_deriv_offsets_are_odd(self):
        N = 9
        v = _cheb(N)
        M = cheb_matrices((v, 0), (v, 1))
        assert M is not None
        for off in M.offsets:
            assert off % 2 == 1, f"expected odd offset, got {off}"

    def test_second_deriv_offsets_are_even_positive(self):
        N = 9
        v = _cheb(N)
        M = cheb_matrices((v, 0), (v, 2))
        assert M is not None
        for off in M.offsets:
            assert off % 2 == 0 and off >= 0, f"unexpected offset {off}"


# ---------------------------------------------------------------------------
# Fourier -- always returns DiaMatrix, always purely diagonal
# ---------------------------------------------------------------------------


class TestFourierReturnType:
    @pytest.mark.parametrize("ij", [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1)])
    def test_returns_dia_matrix(self, ij):
        i, j = ij
        v = _four(8)
        M = fourier_matrices((v, i), (v, j))
        assert isinstance(M, DiaMatrix), (
            f"expected DiaMatrix for ({i},{j}), got {type(M)}"
        )

    @pytest.mark.parametrize("ij", [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)])
    def test_shape_square(self, ij):
        i, j = ij
        N = 8
        v = _four(N)
        M = fourier_matrices((v, i), (v, j))
        assert M is not None
        assert M.shape == (N, N)

    @pytest.mark.parametrize("ij", [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)])
    def test_always_diagonal(self, ij):
        i, j = ij
        v = _four(8)
        M = fourier_matrices((v, i), (v, j))
        assert M is not None
        assert set(M.offsets) == {0}, (
            f"expected only offset 0 for ({i},{j}), got {M.offsets}"
        )


class TestFourierMassMatrix:
    def test_entries_match_norm_squared(self):
        N = 8
        v = _four(N)
        M = fourier_matrices((v, 0), (v, 0))
        assert M is not None
        assert jnp.allclose(M.diagonal(0), v.norm_squared(), atol=ulp(10))

    @pytest.mark.parametrize("N", [8, 10])
    def test_values_match_quadrature(self, N):
        v = _four(N)
        M = fourier_matrices((v, 0), (v, 0))
        assert M is not None
        ref = _quad_matrix(v, 0, 0)
        assert jnp.allclose(M.todense(), ref.real, atol=ulp(100))


class TestFourierDerivativeFormula:
    """Diagonal entries for (i,j) should equal (ik)^j * (-ik)^i * norm_squared."""

    def test_first_derivative_diagonal(self):
        N = 8
        v = _four(N)
        k = v.wavenumbers()
        M = fourier_matrices((v, 0), (v, 1))
        expected = (1j * k) * v.norm_squared()
        assert jnp.allclose(M.diagonal(0).real, expected.real, atol=ulp(10))
        assert jnp.allclose(M.diagonal(0).imag, expected.imag, atol=ulp(10))

    def test_second_derivative_diagonal(self):
        N = 8
        v = _four(N)
        k = v.wavenumbers()
        M = fourier_matrices((v, 0), (v, 2))
        expected = ((1j * k) ** 2 * v.norm_squared()).real
        assert jnp.allclose(M.diagonal(0), expected, atol=ulp(10))


# ---------------------------------------------------------------------------
# Shared tests: ChebyshevU and Ultraspherical (mass matrix only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("space_fn,mats", _MASS_ONLY)
class TestMassOnlyReturnType:
    def test_mass_returns_dia_matrix(self, space_fn, mats):
        v = space_fn(8)
        M = mats((v, 0), (v, 0))
        assert isinstance(M, DiaMatrix)

    @pytest.mark.parametrize("ij", _MASS_ONLY_UNSUPPORTED)
    def test_unsupported_returns_none(self, space_fn, mats, ij):
        i, j = ij
        v = space_fn(8)
        assert mats((v, i), (v, j)) is None, f"expected None for ({i},{j})"


@pytest.mark.parametrize("space_fn,mats", _MASS_ONLY)
class TestMassOnlyMassMatrix:
    def test_shape(self, space_fn, mats):
        N = 8
        v = space_fn(N)
        M = mats((v, 0), (v, 0))
        assert M is not None
        assert M.shape == (N, N)

    def test_diagonal_only(self, space_fn, mats):
        N = 8
        v = space_fn(N)
        M = mats((v, 0), (v, 0))
        assert M is not None
        assert set(M.offsets) == {0}

    def test_entries_match_norm_squared(self, space_fn, mats):
        N = 8
        v = space_fn(N)
        M = mats((v, 0), (v, 0))
        assert M is not None
        assert jnp.allclose(M.diagonal(0), v.norm_squared(), atol=ulp(10))

    @pytest.mark.parametrize("N", [6, 8, 10])
    def test_values_match_quadrature(self, space_fn, mats, N):
        v = space_fn(N)
        M = mats((v, 0), (v, 0))
        assert M is not None
        ref = _quad_matrix(v, 0, 0)
        assert jnp.allclose(M.todense(), ref, atol=ulp(100)), (
            f"N={N}: max err {float(jnp.abs(M.todense() - ref).max())}"
        )
