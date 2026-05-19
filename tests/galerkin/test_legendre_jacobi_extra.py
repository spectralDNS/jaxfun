import jax.numpy as jnp

from jaxfun.galerkin import Jacobi, Legendre
from jaxfun.utils.common import Domain, ulp


def test_legendre_short_series_and_matrices_branches():
    # Trigger nd==1 and nd==2 branches in evaluate
    L = Legendre.Legendre(3)
    c1 = jnp.array([2.0])
    v1 = L.evaluate(0.25, c1)
    assert v1 == c1[0]
    c2 = jnp.array([1.0, 0.5])
    v2 = L.evaluate(0.3, c2)
    expected = c2[0] + c2[1] * 0.3
    assert jnp.isclose(v2, expected, atol=ulp(2.0))
    # matrices lookup branches

    _ = L.matrices(0, (L, 0))
    _ = L.matrices(0, (L, 1))
    _ = L.matrices(1, (L, 0))
    _ = L.matrices(0, (L, 2))
    _ = L.matrices(2, (L, 0))
    assert L.matrices(3, (L, 3)) is None


def test_jacobi_edge_matrices_and_evaluate_short():
    J = Jacobi.Jacobi(3, alpha=1, beta=0, domain=Domain(-1, 1))
    # Short series evaluate path len=1,2
    c1 = jnp.array([1.0])
    _ = J.evaluate(0.2, c1)
    c2 = jnp.array([1.0, 0.3])
    _ = J.evaluate(0.4, c2)
    # Longer series evaluate for stability
    e1 = J.evaluate(0.4, jnp.array([1.0, 0.3, 0.2]))
    assert jnp.isfinite(e1)
