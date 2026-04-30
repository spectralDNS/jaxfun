import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import (
    Chebyshev,
    ChebyshevU,
    Fourier,
    FunctionSpace,
    Jacobi,
    JAXFunction,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    Ultraspherical,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.galerkin.tensorproductspace import TPMatrices
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import ulp


@pytest.fixture(params=(None, Domain(-2, 2)), ids=("domain-default", "domain-mapped"))
def domain(request: pytest.FixtureRequest) -> Domain | None:
    return request.param


@pytest.fixture(
    params=(
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Fourier.Fourier,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
    ids=("Legendre", "Chebyshev", "ChebyshevU", "Fourier", "Jacobi", "Ultraspherical"),
)
def space(request: pytest.FixtureRequest) -> type[OrthogonalSpace]:
    return request.param


@pytest.fixture(
    params=(
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
    ids=("Legendre", "Chebyshev", "ChebyshevU", "Jacobi", "Ultraspherical"),
)
def jspace(request: pytest.FixtureRequest) -> type[Jacobi.Jacobi]:
    return request.param


def test_jaxfunction(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    u = TrialFunction(D)
    v = TestFunction(D)
    A = inner(u * v)
    uf = JAXFunction(jnp.ones(N), D)
    b0 = A @ uf
    b1 = inner(v * uf)
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_jaxfunction_directsum(jspace: type[Jacobi.Jacobi], domain: Domain | None):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain, name="D")
    u = TrialFunction(D)
    v = TestFunction(D)
    A, b = inner(u * v)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    b0 = A @ uf - b
    b1 = inner(v * uf)
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_jaxfunction_diff(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    x = D.system.x
    u = TrialFunction(D)
    v = TestFunction(D)
    A = inner(u.diff(x) * v)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    b0 = A @ uf
    b1 = inner(v * uf.diff(x))
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_jaxfunction_nonlin(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A = inner(u * v * uf)
    b0 = A @ uf
    b1 = inner(v * uf**2)
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(100))


def test_jaxfunction_directsum_nonlin(
    jspace: type[Jacobi.Jacobi], domain: Domain | None
):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain, name="D")
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A, b = inner(u * v * uf)
    b0 = A @ uf - b
    b1 = inner(v * uf**2)
    assert jnp.linalg.norm(b0 - b1) < ulp(1000)


def test_jaxfunction_nonlin_diff(space: type[OrthogonalSpace], domain: Domain | None):
    N = 6
    D = space(N, domain=domain)
    x = D.system.x
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A = inner(u.diff(x) * v * uf.diff(x))
    b0 = A @ uf
    b1 = inner(v * (uf.diff(x)) ** 2)
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1000))


def test_jaxfunction_2d(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    T = TensorProduct(D, D)
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(u * v)
    uf = JAXFunction(jnp.ones((N, N)), T)
    b0 = A[0].mat @ uf.array.ravel()
    b1 = A[0].mats[0] @ uf @ A[0].mats[1].T
    assert jnp.linalg.norm(b0 - b1.ravel()) < ulp(10)
    z = inner(uf * v)
    assert jnp.linalg.norm(z.ravel() - b0) < ulp(10)


def test_jaxfunction_directsum_2d(jspace: type[Jacobi.Jacobi], domain: Domain | None):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain, name="D")
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    A, b = inner(u * v)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="uf")
    b0 = A[0] @ w - b
    b1 = inner(v * w)
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1))
    C, c = inner(Div(Grad(u)) * v)
    b0 = TPMatrices(C) @ w - c
    b1 = inner(v * Div(Grad(w)))
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1))


def test_jaxfunction_diff_2d(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(Div(Grad(u)) * v)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    b0 = TPMatrices(A) @ w
    b1 = inner(v * Div(Grad(w)))
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(10))


def test_jaxfunction_nonlin_2d(space: type[OrthogonalSpace], domain: Domain | None):
    N = 4
    D = space(N, domain=domain)
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    A = inner(u * v * w)
    b0 = A[0] @ w
    b1 = inner(v * w**2)
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1000))


def test_jaxfunction_nonlingrad_2d(
    jspace: type[OrthogonalSpace], domain: Domain | None
):
    N = 4
    D = jspace(N, domain=domain)
    T = TensorProduct(D, D, name="T")
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    A = inner(w * Dot(Grad(v), Grad(u)))
    b0 = A[0] @ w + A[1] @ w
    b1 = inner(0.5 * Dot(Grad(v), Grad(w**2)))
    assert jnp.linalg.norm(b0 - b1) / jnp.linalg.norm(b0) < jnp.sqrt(ulp(10))


def test_jaxfunction_2d_vector(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    T = TensorProduct(D, D)
    V = VectorTensorProductSpace(T, name="V")
    u = TrialFunction(V)
    v = TestFunction(V)
    A = inner(Dot(u, v))
    uf = JAXFunction(jnp.ones((2, N, N)), V)
    b0 = jnp.stack(
        (
            A[1] @ uf.array[A[1].global_indices[1]],
            A[0] @ uf.array[A[0].global_indices[1]],
        )
    )
    b1 = inner(Dot(uf, v))
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(10))


def test_evaluate_derivative():
    N = 24 if jax.config.jax_enable_x64 else 16  # ty:ignore[unresolved-attribute]
    D = Legendre.Legendre(N)
    T = TensorProduct(D, D, name="T")
    x, y = T.system.base_scalars()
    ue = sp.sin(sp.pi * x) * sp.cos(sp.pi * y)
    w = JAXFunction(ue, T, name="w")
    xj, yj = T.mesh()

    dw_dx = T.evaluate_derivative((xj, yj), w.array, k=(1, 0))
    dw_dx_analytic = JAXFunction(sp.diff(ue, x), T).backward()
    assert jnp.linalg.norm(dw_dx - dw_dx_analytic) < jnp.sqrt(ulp(100))

    dw_dy = T.evaluate_derivative((xj, yj), w.array, k=(0, 1))
    dw_dy_analytic = JAXFunction(sp.diff(ue, y), T).backward()
    assert jnp.linalg.norm(dw_dy - dw_dy_analytic) < jnp.sqrt(ulp(100))

    dw_dx_dy = T.evaluate_derivative((xj, yj), w.array, k=(1, 1))
    dw_dx_dy_analytic = JAXFunction(sp.diff(ue, x, y), T).backward()
    assert jnp.linalg.norm(dw_dx_dy - dw_dx_dy_analytic) < jnp.sqrt(ulp(1000))

    # dw_dx_dy2 = T.evaluate_derivative((xj, yj), w.array, k=(1, 2))
    # dw_dx_dy2_analytic = JAXFunction(sp.diff(ue, x, y, y), T).backward()
    # assert jnp.linalg.norm(dw_dx_dy2 - dw_dx_dy2_analytic) < jnp.sqrt(ulp(1000))


def test_jaxfunction_vector(jspace: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = jspace(N, domain=domain)
    T = TensorProduct(D, D)
    V = VectorTensorProductSpace(T, name="V")
    x, y = T.system.base_scalars()
    R = T.system
    ue = sp.Mul(x, R.i) + sp.Mul(y, R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    assert jnp.linalg.norm(uj[0] - T.mesh()[0]) < jnp.sqrt(ulp(10))
    assert jnp.linalg.norm(uj[1] - T.mesh()[1]) < jnp.sqrt(ulp(10))

    ue = sp.Mul((1 - x**2) * (1 - y**2), R.i + R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    xj, yj = T.mesh()
    assert jnp.linalg.norm(uj[0] - (1 - xj**2) * (1 - yj**2)) < jnp.sqrt(ulp(10))
    assert jnp.linalg.norm(uj[1] - (1 - xj**2) * (1 - yj**2)) < jnp.sqrt(ulp(10))


def test_jaxfunction_vector_composite(
    jspace: type[OrthogonalSpace], domain: Domain | None
):
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    V = VectorTensorProductSpace(T, name="V")
    x, y = T.system.base_scalars()
    R = T.system

    a = D.domain.upper
    ue = sp.Mul((a**2 - x**2) * (a**2 - y**2), R.i + R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    xj, yj = T.mesh()
    assert jnp.linalg.norm(uj[0] - (a**2 - xj**2) * (a**2 - yj**2)) < jnp.sqrt(ulp(10))
    assert jnp.linalg.norm(uj[1] - (a**2 - xj**2) * (a**2 - yj**2)) < jnp.sqrt(ulp(10))


def test_jaxfunction_vector_directsum(
    jspace: type[OrthogonalSpace], domain: Domain | None
):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    V = VectorTensorProductSpace(T, name="V")
    x, y = T.system.base_scalars()
    R = T.system

    a = D.domain.upper
    ue = sp.Mul((a**2 - x**2) * (a**2 - y**2) + 1, R.i + R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    xj, yj = T.mesh()
    assert jnp.linalg.norm(uj[0] - ((a**2 - xj**2) * (a**2 - yj**2) + 1)) < jnp.sqrt(
        ulp(10)
    )
    assert jnp.linalg.norm(uj[1] - ((a**2 - xj**2) * (a**2 - yj**2) + 1)) < jnp.sqrt(
        ulp(10)
    )
