import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.utils.fastgl import leggauss, leggauss_float64


@pytest.mark.parametrize("N", [1, 2, 7, 100, 101, 256])
def test_leggauss_float64(N: int) -> None:
    x, w = leggauss_float64(N)
    # N <= 100 is tabulated, larger N uses the asymptotic expansion.
    assert np.allclose(x, np.polynomial.legendre.leggauss(N)[0], atol=1e-15)
    assert np.all(np.diff(x) > 0)
    j = np.arange(N)
    moments = (w[:, None] * x[:, None] ** (2 * j)).sum(axis=0)
    assert np.allclose(moments, 2 / (2 * j + 1), atol=1e-15)


def test_leggauss_dtype() -> None:
    assert leggauss(0).shape == (2, 0)
    assert leggauss(5).dtype == jnp.result_type(float)
