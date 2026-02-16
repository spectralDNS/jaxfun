import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    Jacobi,
    JAXFunction,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.inner import inner, project
from jaxfun.galerkin.tensorproductspace import TPMatrices
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import ulp


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction(space):
    N = 8
    D = space(N)
    u = TrialFunction(D)
    v = TestFunction(D)
    A = inner(u * v)
    uf = JAXFunction(jnp.ones(N), D)
    b0 = A @ uf
    b1 = inner(v * uf)
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Jacobi.Jacobi)
)
def test_jaxfunction_directsum(space):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, space, bcs=bcs, name="D")
    u = TrialFunction(D)
    v = TestFunction(D)
    A, b = inner(u * v)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    b0 = A @ uf - b
    b1 = inner(v * uf)
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_diff(space):
    N = 8
    D = space(N)
    x = D.system.x
    u = TrialFunction(D)
    v = TestFunction(D)
    A = inner(u.diff(x) * v)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    b0 = A @ uf
    b1 = inner(v * uf.diff(x))
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_nonlin(space):
    N = 8
    D = space(N)
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A = inner(u * v * uf)
    b0 = A @ uf
    b1 = inner(v * uf**2)
    assert jnp.linalg.norm(b0 - b1) < ulp(1000)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Jacobi.Jacobi)
)
def test_jaxfunction_directsum_nonlin(space):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, space, bcs=bcs, name="D")
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A, b = inner(u * v * uf)
    b0 = A @ uf - b
    b1 = inner(v * uf**2)
    assert jnp.linalg.norm(b0 - b1) < ulp(1000)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_nonlin_diff(space):
    N = 8
    D = space(N)
    x = D.system.x
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A = inner(u.diff(x) * v * uf.diff(x))
    b0 = A @ uf
    b1 = inner(v * (uf.diff(x)) ** 2)
    assert jnp.linalg.norm(b0 - b1) < ulp(100000)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_2d(space):
    N = 8
    D = space(N)
    T = TensorProduct(D, D)
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(u * v)
    uf = JAXFunction(jnp.ones((N, N)), T)
    b0 = A[0].mat @ uf.array.ravel()
    b1 = A[0].mats[0] @ uf @ A[0].mats[1].T
    assert jnp.linalg.norm(b0 - b1.ravel()) < ulp(100)
    z = inner(uf * v)
    assert jnp.linalg.norm(z.ravel() - b0) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Jacobi.Jacobi)
)
def test_jaxfunction_directsum_2d(space):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, space, bcs=bcs, name="D")
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    A, b = inner(u * v)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="uf")
    b0 = A[0] @ w - b
    b1 = inner(v * w)
    assert jnp.linalg.norm(b0 - b1) < ulp(100)
    C, c = inner(Div(Grad(u)) * v)
    b0 = TPMatrices(C) @ w - c
    b1 = inner(v * Div(Grad(w)))
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_diff_2d(space):
    N = 8
    D = space(N)
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(Div(Grad(u)) * v)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    b0 = TPMatrices(A) @ w
    b1 = inner(v * Div(Grad(w)))
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_nonlin_2d(space):
    N = 8
    D = space(N)
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    A = inner(u * v * w)
    b0 = A[0] @ w
    b1 = inner(v * w**2)
    assert jnp.linalg.norm(b0 - b1) < ulp(100000)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_jaxfunction_2d_vector(space):
    N = 8
    D = space(N)
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
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_evaluate_derivative():
    N = 8
    D = Legendre.Legendre(N)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    ue = sp.sin(x) * sp.cos(y)
    uh = project(ue, T)
    w = JAXFunction(uh, T, name="w")
    xj, yj = T.mesh()
    k = (1, 1)
    dw_dx = T.evaluate_derivative((xj, yj), w.array, k=k)
    from IPython import embed

    embed()
    dw_dx_analytic = JAXFunction(project(sp.diff(ue, x, y), T), T).backward()
    assert jnp.linalg.norm(dw_dx - dw_dx_analytic) < ulp(1000)


if __name__ == "__main__":
    # test_jaxfunction_2d(Chebyshev.Chebyshev)
    # test_jaxfunction_nonlin_diff(Chebyshev.Chebyshev)
    test_evaluate_derivative()
