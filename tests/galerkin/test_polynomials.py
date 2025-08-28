import jax.numpy as jnp

from jaxfun.galerkin import Chebyshev, Jacobi, Legendre
from jaxfun.utils.common import Domain, ulp


def test_chebyshev_evaluate_variants():
    C = Chebyshev.Chebyshev(8)
    x = jnp.linspace(-1, 1, 17)
    c = jnp.arange(8.0)
    u1 = jnp.array([C.evaluate(xi, c) for xi in x])
    u2 = jnp.array([C.evaluate2(xi, c) for xi in x])
    assert jnp.linalg.norm(u1 - u2) < 100 * ulp(1.0)
    # Edge cases length 1 and 2
    for m in (1, 2):
        c0 = jnp.arange(float(m))
        u = C.evaluate(0.3, c0)
        assert jnp.isfinite(u)


def test_legendre_evaluate_variants_and_domain_mapping():
    L = Legendre.Legendre(6, domain=Domain(-2, 2))
    x = jnp.linspace(-2, 2, 9)
    c = jnp.arange(6.0)
    u1 = jnp.array([L.evaluate(xi, c) for xi in x])
    u2 = jnp.array([L.evaluate2(xi, c) for xi in x])
    assert jnp.linalg.norm(u1 - u2) < 100 * ulp(1.0)
    # Mapping check: reference domain is (-1,1)
    Xref = jnp.array([L.map_reference_domain(xi) for xi in x])
    assert Xref.min() >= -1 - 1e-12 and Xref.max() <= 1 + 1e-12


def test_jacobi_general_parameters():
    J = Jacobi.Jacobi(5, alpha=1, beta=2)
    x = jnp.linspace(-1, 1, 13)
    c = jnp.arange(5.0)
    u = jnp.array([J.evaluate(xi, c) for xi in x])
    assert jnp.isfinite(u).all()
    # Mass matrix diagonal
    M = J.mass_matrix().todense()
    assert jnp.allclose(M.diagonal(), J.norm_squared() / J.domain_factor)
