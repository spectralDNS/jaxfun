from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from jaxfun import Domain
from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    Jacobi,
    JAXFunction,
    Legendre,
    OrthogonalExpansion,
    Ultraspherical,
)
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.utils.common import ulp

polynomial_spaces = (
    lambda N, d: Chebyshev.Chebyshev(N, domain=d),
    lambda N, d: Legendre.Legendre(N, domain=d),
    lambda N, d: Jacobi.Jacobi(N, domain=d, alpha=1, beta=2),
    lambda N, d: Ultraspherical.Ultraspherical(N, domain=d),
)
polynomial_ids = ("Chebyshev", "Legendre", "Jacobi", "Ultraspherical")


@pytest.fixture(params=polynomial_spaces, ids=polynomial_ids)
def pspace(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=(None, Domain(0, 1)), ids=("domain-default", "domain-mapped"))
def domain(request: pytest.FixtureRequest) -> Domain | None:
    return request.param


def random_expansion(S: OrthogonalSpace, seed: int) -> OrthogonalExpansion:
    rng = np.random.default_rng(seed)
    return OrthogonalExpansion(jnp.array(rng.standard_normal(S.N)), S)


def points(S: OrthogonalSpace) -> jax.Array:
    a, b = float(S.domain.lower), float(S.domain.upper)
    return jnp.linspace(a, b, 7)


def test_product(pspace, domain: Domain | None):
    a = random_expansion(pspace(5, domain), 0)
    b = random_expansion(pspace(3, domain), 1)
    X = points(a.space)
    p = a * b
    assert p.N == 7
    assert jnp.allclose(p(X), a(X) * b(X), atol=ulp(1000))
    assert jnp.allclose((a**2)(X), a(X) ** 2, atol=ulp(1000))
    assert (a**2).N == 9


def test_mul_x(pspace, domain: Domain | None):
    a = random_expansion(pspace(4, domain), 2)
    x = a.space.system.x
    X = points(a.space)
    for p in (x * a, a * x):
        assert p.N == 5
        assert jnp.allclose(p(X), X * a(X), atol=ulp(100))


def test_add_sub_scale(pspace):
    a = random_expansion(pspace(5, None), 3)
    b = random_expansion(pspace(3, None), 4)
    X = points(a.space)
    assert (a + b).N == 5
    assert jnp.allclose((a + b)(X), a(X) + b(X), atol=ulp(100))
    assert jnp.allclose((b - a)(X), b(X) - a(X), atol=ulp(100))
    assert jnp.allclose((2 - a)(X), 2 - a(X), atol=ulp(100))
    assert jnp.allclose((a + 1.5)(X), a(X) + 1.5, atol=ulp(100))
    assert jnp.allclose((3 * a / 2)(X), 1.5 * a(X), atol=ulp(100))
    assert jnp.allclose((a**0)(X), 1.0, atol=ulp(100))


def test_deriv(domain: Domain | None):
    # u = x**2 in the Chebyshev basis
    T = Chebyshev.Chebyshev(3, domain=domain)
    x = OrthogonalExpansion.from_expr(T.system.x, T)
    u = x * x
    X = points(T)
    assert jnp.allclose(u.deriv()(X), 2 * X, atol=ulp(100))
    assert jnp.allclose(u.deriv(2)(X), 2.0, atol=ulp(1000))


def test_fourier_product():
    F = Fourier.Fourier(6)
    a = random_expansion(F, 5)
    b = random_expansion(Fourier.Fourier(4), 6)
    X = points(F)
    p = a * b
    assert p.N == 10
    assert jnp.allclose(p(X), a(X) * b(X), atol=ulp(1000))
    with pytest.raises(NotImplementedError):
        a.mul_x()


def test_different_bases_raise():
    a = random_expansion(Chebyshev.Chebyshev(3), 0)
    b = random_expansion(Legendre.Legendre(3), 0)
    with pytest.raises(TypeError):
        a + b


def test_jaxfunction_round_trip():
    D = FunctionSpace(5, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 2}})
    u = JAXFunction(jnp.arange(1.0, D.dim + 1), D)
    p = OrthogonalExpansion.from_jaxfunction(u)
    assert p.N == 5
    assert jnp.allclose(p(points(p.space)), u(points(p.space)), atol=ulp(100))
    assert jnp.allclose(
        cast(Array, p.to_jaxfunction(D).array), cast(Array, u.array), atol=ulp(100)
    )
    with pytest.raises(ValueError):
        (p * p).to_jaxfunction(D)


def test_pytree():
    a = random_expansion(Chebyshev.Chebyshev(4), 7)
    X = points(a.space)
    f = jax.jit(lambda p: (p * p)(X))
    assert jnp.allclose(f(a), a(X) ** 2, atol=ulp(1000))
    g = jax.grad(lambda c: (OrthogonalExpansion(c, a.space) ** 2)(X).sum())(a.coef)
    assert g.shape == (4,)


def test_jaxfunction_to_expansion():
    D = FunctionSpace(5, Chebyshev.Chebyshev, bcs={"left": {"D": 1}, "right": {"D": 2}})
    u = JAXFunction(jnp.arange(1.0, D.dim + 1), D)
    p = u.to_expansion()
    assert isinstance(p, OrthogonalExpansion)
    assert isinstance(p.space, Chebyshev.Chebyshev) and p.N == 5
    X = points(p.space)
    assert jnp.allclose(p(X), u(X), atol=ulp(100))
    assert jnp.allclose((p * p)(X), u(X) ** 2, atol=ulp(1000))


def test_sizes():
    a = random_expansion(Legendre.Legendre(4), 0)
    p = a * a
    assert (p.N, p.dim, p.num_dofs, p.shape, p.degree) == (7, 7, 7, (7,), 6)
