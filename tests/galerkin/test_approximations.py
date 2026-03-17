import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.galerkin import JAXFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.Ultraspherical import Ultraspherical
from jaxfun.utils import ulp


# Add pytest fixtures for constants
@pytest.fixture
def N() -> int:
    return 10


@pytest.fixture
def x(N: int) -> jnp.ndarray:
    return jnp.linspace(-1, 1, N + 1)


@pytest.fixture
def xn(x: jnp.ndarray) -> np.ndarray:
    return np.array(x)


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_vandermonde(
    space: type[Legendre] | type[Chebyshev], x: jnp.ndarray, xn: np.ndarray, N: int
) -> None:
    V = space(N)
    np_res = {
        "Legendre": np.polynomial.legendre.legvander(xn, N - 1),
        "Chebyshev": np.polynomial.chebyshev.chebvander(xn, N - 1),
    }[space.__name__]
    jax_res = V.vandermonde(x)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < ulp(10.0)


@pytest.mark.parametrize("k", (0, 1, 2, 3))
@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_evaluate_basis_derivative(
    space: type[Legendre] | type[Chebyshev],
    x: jnp.ndarray,
    xn: np.ndarray,
    N: int,
    k: int,
) -> None:
    V = space(N)
    np_res = {
        "Legendre": np.polynomial.legendre.legvander(xn, N - 1),
        "Chebyshev": np.polynomial.chebyshev.chebvander(xn, N - 1),
    }[space.__name__]
    der = {
        "Legendre": np.polynomial.legendre.legder,
        "Chebyshev": np.polynomial.chebyshev.chebder,
    }[space.__name__]
    P = np_res.shape[-1]
    if k > 0:
        D = np.zeros((P, P))
        D[:-k] = der(np.eye(P, P), k)
        np_res = np.dot(np_res, D)
    jax_res = V.evaluate_basis_derivative(x, k=k)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < ulp(10 ** (k + 2))


@pytest.mark.parametrize("space", (Legendre, Chebyshev, Ultraspherical, Fourier))
def test_backward_primitive(space):
    N = 16
    space = space(N)
    x = space.system.x
    f = sp.sin(x)
    uf = JAXFunction(f, space)
    du = JAXFunction(sp.diff(f, x), space)
    df = space.backward_primitive(uf.array, 1)
    error = jnp.linalg.norm(df - du.backward())
    assert error < ulp(10000)
