import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.inner import project
from jaxfun.galerkin.Jacobi import Jacobi
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.utils import Domain, ulp


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


@pytest.fixture(
    params=(Domain(-1, 1), Domain(-2, 2)), ids=("domain-default", "domain-mapped")
)
def domain(request: pytest.FixtureRequest) -> Domain:
    return request.param


@pytest.fixture(
    params=(Legendre, Chebyshev, ChebyshevU, Jacobi),
    ids=("Legendre", "Chebyshev", "ChebyshevU", "Jacobi"),
)
def jspace(request: pytest.FixtureRequest) -> type[Jacobi]:
    return request.param


def test_to_from_orthogonal_1d(jspace: type[Jacobi], domain: Domain):
    N = 16
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    x = D.system.x
    f = sp.sin(x * sp.pi)
    u = project(f, D)
    assert jnp.linalg.norm(u - D.from_orthogonal(D.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_1d_directsum(jspace: type[Jacobi], domain: Domain):
    N = 16
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    x = D.system.x
    f = sp.cos(x * sp.pi)
    u = project(f, D)
    assert jnp.linalg.norm(u - D.from_orthogonal(D.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_2d(jspace: type[Jacobi], domain: Domain):
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    f = sp.sin(x * sp.pi) * sp.sin(y * sp.pi)
    u = project(f, T)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_2d_directsum(jspace: type[Jacobi], domain: Domain):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    f = sp.cos(x * sp.pi) * sp.cos(y * sp.pi)
    u = project(f, T)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_3d(jspace: type[Jacobi], domain: Domain):
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D, D)
    x, y, z = T.system.base_scalars()
    f = sp.sin(x * sp.pi) * sp.sin(y * sp.pi) * sp.sin(z * sp.pi)
    u = project(f, T)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)
