import jax
import jax.numpy as jnp
import pytest

from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Jacobi import Jacobi
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.sharding import spectral_sharding
from jaxfun.utils import ulp

pytestmark = pytest.mark.spmd

assert jax.device_count() in (2, 4), "SPMD tests require 2 or 4 devices"


@pytest.fixture(
    params=(Legendre, Chebyshev),
    ids=("Legendre", "Chebyshev"),
)
def jspace(request: pytest.FixtureRequest) -> type[Jacobi]:
    return request.param


def test_to_from_orthogonal_2d(jspace: type[Jacobi]):
    N = 10
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, bcs=bcs)
    T = TensorProduct(D, D)
    u = jax.device_put(jnp.ones(T.num_dofs), spectral_sharding)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_2d_directsum(jspace: type[Jacobi]):
    N = 10
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs)
    T = TensorProduct(D, D)
    u = jax.device_put(jnp.ones(T.num_dofs), spectral_sharding)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_2d_fourier_directsum(jspace: type[Jacobi]):
    N = 10
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs)
    F = FunctionSpace(N, Fourier, name="F", fun_str="E")
    T = TensorProduct(F, D)
    u = jax.device_put(jnp.ones(T.num_dofs), spectral_sharding)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(100)


def test_to_from_orthogonal_3d(jspace: type[Jacobi]):
    N = 10
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, bcs=bcs)
    T = TensorProduct(D, D, D)
    u = jax.device_put(jnp.ones(T.num_dofs), spectral_sharding)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(1000)


def test_to_from_orthogonal_fourier_3d(jspace: type[Jacobi]):
    N = 10
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, bcs=bcs)
    F = FunctionSpace(N, Fourier, name="F", fun_str="E")
    T = TensorProduct(F, D, D)
    u = jax.device_put(jnp.ones(T.num_dofs), spectral_sharding)
    assert jnp.linalg.norm(u - T.from_orthogonal(T.to_orthogonal(u))) < ulp(1000)
