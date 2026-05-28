import jax.numpy as jnp
import pytest

from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Jacobi import Jacobi
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.utils import Domain, ulp

pytestmark = pytest.mark.spmd


@pytest.fixture(
    params=(Domain(-1, 1), Domain(-2, 2)), ids=("domain-default", "domain-mapped")
)
def domain(request: pytest.FixtureRequest) -> Domain:
    return request.param


@pytest.fixture(
    params=(Legendre, Chebyshev),
    ids=("Legendre", "Chebyshev"),
)
def jspace(request: pytest.FixtureRequest) -> type[Jacobi]:
    return request.param


def test_to_from_orthogonal_2d(jspace: type[Jacobi], domain: Domain):
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    u = jnp.ones(T.num_dofs)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_2d_directsum(jspace: type[Jacobi], domain: Domain):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    u = jnp.ones(T.num_dofs)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_3d(jspace: type[Jacobi], domain: Domain):
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D, D)
    u = jnp.ones(T.num_dofs)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(1000)
