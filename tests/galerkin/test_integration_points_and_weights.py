"""Unweighted quadrature rules from `integration_points_and_weights`."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.galerkin import FunctionSpace
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Jacobi import Jacobi
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.utils.common import ulp


@pytest.mark.parametrize("family", [Legendre, Chebyshev, ChebyshevU])
@pytest.mark.parametrize("n", [7, 12])
def test_polynomials_below_n_are_integrated_exactly(family, n):
    space = family(8)
    x, q = space.integration_points_and_weights(n)
    assert jnp.array_equal(x, space.quad_points_and_weights(n)[0])
    k = np.arange(n)
    exact = np.where(k % 2 == 0, 2 / (k + 1), 0.0)
    assert jnp.abs(q @ x[:, None] ** k - exact).max() < ulp(100)


def test_fourier_integrates_trigonometric_polynomials():
    x, q = Fourier(8).integration_points_and_weights()
    k = np.arange(8)
    exact = np.where(k == 0, 2 * np.pi, 0.0)
    assert jnp.abs(q @ jnp.cos(x[:, None] * k) - exact).max() < ulp(100)


def test_boundary_spaces_use_their_orthogonal_rule():
    V = FunctionSpace(8, Chebyshev, bcs={"left": {"D": 1}, "right": {"D": 0}})
    assert isinstance(V, DirectSum)
    x0, q0 = Chebyshev(8).integration_points_and_weights(12)
    for space in (V, *V.basespaces):
        x, q = space.integration_points_and_weights(12)
        assert jnp.array_equal(x, x0) and jnp.array_equal(q, q0)


def test_other_weight_functions_are_not_implemented():
    with pytest.raises(NotImplementedError):
        Jacobi(8, alpha=1, beta=1).integration_points_and_weights()
