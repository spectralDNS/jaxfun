import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.Chebyshev import Chebyshev
from jaxfun.Legendre import Legendre
from jaxfun.utils.common import evaluate

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
    return jnp.linspace(-1, 1, N + 1)


@pytest.fixture
def xn(x: jnp.ndarray) -> np.ndarray:
    return np.array(x)


@pytest.fixture
def c(C: int) -> jnp.ndarray:
    return jnp.ones(C)


@pytest.fixture
def cn(c: jnp.ndarray) -> np.ndarray:
    return np.array(c)


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_vandermonde(space, x: jnp.ndarray, xn: np.ndarray, N: int) -> None:
    space = space(N)
    np_res = {
        "Legendre": np.polynomial.legendre.legvander(xn, N),
        "Chebyshev": np.polynomial.chebyshev.chebvander(xn, N),
    }[space.__class__.__name__]
    jax_res = space.vandermonde(x)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_evaluate(
    space, x: jnp.ndarray, xn: np.ndarray, c: jnp.ndarray, cn: np.ndarray, N: int
) -> None:
    space = space(N)
    np_res = {
        "Legendre": np.polynomial.legendre.legval(xn, cn),
        "Chebyshev": np.polynomial.chebyshev.chebval(xn, cn),
    }[space.__class__.__name__]
    jax_res = space.evaluate(x, c)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8


@pytest.mark.parametrize("k", (1, 2, 3))
@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_evaluate_basis_derivative(
    space, x: jnp.ndarray, xn: np.ndarray, N: int, k: int
) -> None:
    space = space(N)
    np_res = {
        "Legendre": np.polynomial.legendre.legvander(xn, N),
        "Chebyshev": np.polynomial.chebyshev.chebvander(xn, N),
    }[space.__class__.__name__]
    der = {
        "Legendre": np.polynomial.legendre.legder,
        "Chebyshev": np.polynomial.chebyshev.chebder,
    }[space.__class__.__name__]
    P = np_res.shape[-1]
    if k > 0:
        D = np.zeros((P, P))
        D[:-k] = der(np.eye(P, P), k)
        np_res = np.dot(np_res, D)
    jax_res = space.evaluate_basis_derivative(x, k=k)
    diff = jnp.linalg.norm(jnp.array(np_res) - jax_res)
    assert diff < 1e-8


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_evaluate_multidimensional(space, N: int) -> None:
    x = jnp.array([-1.0, 1.0])
    c = jnp.ones((2, 2))
    space = space(N)
    y = evaluate(space.evaluate, (x,), c, (1,))
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [
                    0.0,
                    2.0,
                ],
                [0.0, 2.0],
            ]
        ),
    )
    y = evaluate(space.evaluate, (x,), c, (0,))
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [
                    0.0,
                    0.0,
                ],
                [2.0, 2.0],
            ]
        ),
    )
    y = evaluate(space.evaluate, (x, x), c, (0, 1))
    assert jnp.allclose(
        y,
        jnp.array(
            [
                [
                    0.0,
                    0.0,
                ],
                [0.0, 4.0],
            ]
        ),
    )


if __name__ == "__main__":
    test_evaluate(Legendre, x, xn, c, cn)
