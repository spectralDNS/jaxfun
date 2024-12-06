import jax
import jax.numpy as jnp
import numpy as np
import pytest

import Chebyshev
import Legendre

jax.config.update("jax_enable_x64", True)


# Add pytest fixtures for constants
@pytest.fixture
def N() -> int:
    return 10


@pytest.fixture
def M() -> int:
    return 1000


@pytest.fixture
def C() -> int:
    return 10


@pytest.fixture
def x(N: int) -> jnp.ndarray:
    return jnp.linspace(-1, 1, N)


@pytest.fixture
def xn(x: jnp.ndarray) -> np.ndarray:
    return np.array(x)


@pytest.fixture
def c(C: int) -> jnp.ndarray:
    return jnp.ones(C)


@pytest.fixture
def cn(c: jnp.ndarray) -> np.ndarray:
    return np.array(c)


def test_legvander(x: jnp.ndarray, xn: np.ndarray, N: int) -> None:
    np_res = np.polynomial.legendre.legvander(xn, N - 1)
    jax_res = Legendre.legvander(x, N)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8


def test_legval(x: jnp.ndarray, xn: np.ndarray, c: jnp.ndarray, cn: np.ndarray) -> None:
    np_res = np.polynomial.legendre.legval(xn, cn)
    jax_res = Legendre.legval(x, c)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8


def test_chebvander(x: jnp.ndarray, xn: np.ndarray, N: int) -> None:
    np_res = np.polynomial.chebyshev.chebvander(xn, N - 1)
    jax_res = Chebyshev.chebvander(x, N)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8


def test_chebval(
    x: jnp.ndarray, xn: np.ndarray, c: jnp.ndarray, cn: np.ndarray
) -> None:
    np_res = np.polynomial.chebyshev.chebval(xn, cn)
    jax_res = Chebyshev.chebval(x, c)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8
