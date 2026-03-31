import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.galerkin import FunctionSpace, JAXFunction, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Jacobi import Jacobi
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.Ultraspherical import Ultraspherical
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


def test_backward_primitive(jspace: type[Jacobi], domain: Domain):
    N = 16
    D = jspace(N, domain=domain)
    x = D.system.x
    f = sp.sin(x)
    uf = JAXFunction(f, D)
    du = JAXFunction(sp.diff(f, x), D)
    df = D.backward_primitive(uf.array, 1)
    error = jnp.linalg.norm(df - du.backward())
    assert error < jnp.sqrt(ulp(10))
    if jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        du = JAXFunction(sp.diff(f, x, 2), D)
        df = D.backward_primitive(uf.array, 2)
        error = jnp.linalg.norm(df - du.backward())
        assert error < jnp.sqrt(ulp(100)), error


def test_backward_primitive_composite(jspace: type[Jacobi], domain: Domain):
    N = 24
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain)
    x = D.system.x
    f = sp.sin(x * sp.pi)
    uf = JAXFunction(f, D)
    du = JAXFunction(sp.diff(f, x), D.orthogonal)  # Cannot use D due to bcs
    df = D.backward_primitive(uf.array, 1)
    error = jnp.linalg.norm(df - du.backward())
    assert error < jnp.sqrt(ulp(10))
    if jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        du = JAXFunction(sp.diff(f, x, 2), D.orthogonal)
        df = D.backward_primitive(uf.array, 2)
        error = jnp.linalg.norm(df - du.backward())
        assert error < jnp.sqrt(ulp(100)), error


def test_backward_primitive_directsum(jspace: type[Jacobi], domain: Domain):
    from jaxfun.coordinates import x

    N = 24
    f = sp.cos(x * sp.pi)
    bcs = {
        "left": {"D": f.subs(x, domain.lower)},
        "right": {"D": f.subs(x, domain.upper)},
    }
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain)
    (x,) = D.system.base_scalars()
    f = D.system.expr_psi_to_base_scalar(f)
    uf = JAXFunction(f, D)
    du = JAXFunction(sp.diff(f, x), D.orthogonal)  # Cannot use D due to bcs
    df = D.backward_primitive(uf.array, 1)
    error = jnp.linalg.norm(df - du.backward())
    assert error < jnp.sqrt(ulp(100))
    if jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        du = JAXFunction(sp.diff(f, x, 2), D.orthogonal)
        df = D.backward_primitive(uf.array, 2)
        error = jnp.linalg.norm(df - du.backward())
        assert error < jnp.sqrt(ulp(100)), error


@pytest.mark.parametrize(
    "space", (Legendre, Chebyshev, ChebyshevU, Ultraspherical, Fourier)
)
def test_backward_primitive_2d(space):
    N = 16
    D = space(N)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    f = sp.sin(x) * sp.cos(y)
    uf = JAXFunction(f, T)
    du = JAXFunction(sp.diff(f, x, y), T)
    df = T.backward_primitive(uf.array, (1, 1))
    error = jnp.linalg.norm(df - du.backward())
    assert error < jnp.sqrt(ulp(100))


def test_backward_primitive_directsum_2d(jspace: type[Jacobi], domain: Domain):
    from jaxfun.coordinates import x, y

    if not jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        return

    N = 24
    f = sp.cos(x * sp.pi) * sp.cos(y * sp.pi)
    bcsx = {
        "left": {"D": f.subs(x, domain.lower)},
        "right": {"D": f.subs(x, domain.upper)},
    }
    bcsy = {
        "left": {"D": f.subs(y, domain.lower)},
        "right": {"D": f.subs(y, domain.upper)},
    }
    Dx = FunctionSpace(N, jspace, bcs=bcsx, domain=domain)
    Dy = FunctionSpace(N, jspace, bcs=bcsy, domain=domain)
    T = TensorProduct(Dx, Dy)
    x, y = T.system.base_scalars()
    f = T.system.expr_psi_to_base_scalar(f)
    uf = JAXFunction(f, T)
    du = JAXFunction(sp.diff(f, x, y), T.get_orthogonal())
    df = T.backward_primitive(uf.array, (1, 1))
    error = jnp.linalg.norm(df - du.backward())
    assert error < jnp.sqrt(ulp(1000)), error
