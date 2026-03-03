import jax
import jax.numpy as jnp

from jaxfun import Domain
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace


def _complex_coeffs(key: jax.Array, n: int) -> jax.Array:
    kr, ki = jax.random.split(key)
    return jax.random.normal(kr, (n,)) + 1j * jax.random.normal(ki, (n,))


def test_fourier_evaluate_nonlinear_primitive_matches_evaluate_derivative() -> None:
    N = 32
    V = Fourier(N, domain=Domain(-1, 1))
    c = _complex_coeffs(jax.random.PRNGKey(1), N)
    xj = V.mesh()

    for k in (0, 1, 3):
        got = V.evaluate_nonlinear_primitive(c, derivative_order=k)
        expected = V.backward(c) if k == 0 else V.evaluate_derivative(xj, c, k=k)
        if k == 0:
            assert jnp.allclose(got, expected, atol=1e-8, rtol=1e-8)
            continue
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 1e-5


def test_chebyshev_evaluate_nonlinear_primitive_matches_evaluate_derivative() -> None:
    N = 33
    V = Chebyshev(N, domain=Domain(0, 2))
    c = jax.random.normal(jax.random.PRNGKey(2), (N,))
    xj = V.mesh()

    for k in (0, 1, 2):
        got = V.evaluate_nonlinear_primitive(c, derivative_order=k)
        expected = V.backward(c) if k == 0 else V.evaluate_derivative(xj, c, k=k)
        if k == 0:
            assert jnp.allclose(got, expected, atol=1e-8, rtol=1e-8)
            continue
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 2e-5


def test_directsum_evaluate_nonlinear_primitive_includes_boundary_lift() -> None:
    N = 24
    V = FunctionSpace(
        N,
        Chebyshev,
        bcs={"left": {"D": 1.0}, "right": {"D": -0.5}},
        domain=Domain(-1, 1),
    )
    c = jax.random.normal(jax.random.PRNGKey(3), (V.num_dofs,))
    xj = V.mesh()

    for k in (0, 1):
        got = V.evaluate_nonlinear_primitive(c, derivative_order=k)
        expected = V.backward(c) if k == 0 else V.evaluate_derivative(xj, c, k=k)
        if k == 0:
            assert jnp.allclose(got, expected, atol=1e-8, rtol=1e-8)
            continue
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 2e-5
