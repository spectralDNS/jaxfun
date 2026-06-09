from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp
from jax import Array

from jaxfun.galerkin import (
    CartesianProduct,
    Chebyshev,
    Fourier,
    FunctionSpace,
    JAXFunction,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.inner import inner
from jaxfun.la import TensorMatrix, TPMatrices, TPMatrix
from jaxfun.operators import Dot
from jaxfun.utils.common import lambdify, ulp

pytestmark = pytest.mark.integration


def test_tensorproduct_forward_backward_padding_fourier():
    # Padding path for Fourier
    F = Fourier.Fourier(8)
    T = TensorProduct(F, F)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    M, b = inner(v * (u - sp.sin(x) * sp.sin(y)), kind="system")
    assert isinstance(M, TPMatrix)
    # Solve
    uh = M.solve(b)
    # Backward on padded grid
    up = T.backward(uh, N=(12, 8))
    # Make sure shape matches requested padding (only first axis padded)
    assert up.shape == (12, 8)


def test_tensorproduct_directsum_tps_forward_backward():
    # Create 2D with one inhomogeneous dirichlet dimension
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    C = FunctionSpace(6, Legendre.Legendre, bcs=bcs)
    L = Legendre.Legendre(6)
    T = TensorProduct(C, L)
    assert hasattr(T, "tpspaces")  # DirectSumTPS
    assert isinstance(C, DirectSum)
    # Build form (Poisson like)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    A, b = inner(
        sp.diff(u, x) * sp.diff(v, x) + sp.diff(u, y) * sp.diff(v, y),
        kind="system",
    )
    # Ensure we got list of TPMatrix objects
    assert isinstance(A, TPMatrices)
    # Add rhs from non-zero bc lifting by calling backward on zero coeffs.
    # Need homogeneous tensor space to get coefficient shape.
    # Build shape from homogeneous part (first component of direct sum)
    # and second plain Legendre space
    hom0 = C[0]
    uh = jnp.zeros((hom0.dim, L.dim))
    _ = T.backward(uh)


def test_tp_matrix_and_preconditioner():
    C = Chebyshev.Chebyshev(6)
    L = Legendre.Legendre(6)
    T = TensorProduct(C, L)
    v = TestFunction(T)
    u = TrialFunction(T)
    A = inner(v * u, kind="bilinear")
    assert isinstance(A, TPMatrix)
    X = jax.random.normal(jax.random.PRNGKey(0), shape=T.num_dofs)
    Y = A @ X
    # Check manual matmul versus kron
    Y2 = A.mats[0] @ X @ A.mats[1].T
    assert jnp.linalg.norm(Y - Y2) < ulp(100)


def test_vectortensorproductspace_forward_backward_roundtrip():
    N = 8
    C = Chebyshev.Chebyshev(N)
    T = TensorProduct(C, C)
    V = CartesianProduct(T, T, name="V", rank=1)
    coeffs = tuple(
        jax.random.normal(jax.random.PRNGKey(i), shape=Vi.num_dofs)
        for i, Vi in enumerate(V.flatten())
    )
    u = JAXFunction(coeffs, V)
    ua = V.backward(cast(tuple[Array, ...], u.array))
    coeffs_rt = V.forward(ua)
    assert all(
        jnp.linalg.norm(u.array[i] - coeffs_rt[i]) < ulp(1000) for i in range(len(V))
    )


def test_vectortensorproductspace_padded_backward_forward_truncation_roundtrip():
    N: int = 8
    pad: tuple[int, int] = (N + 4, N + 3)

    C = Chebyshev.Chebyshev(N)
    T = TensorProduct(C, C)
    V = CartesianProduct(T, T, name="V", rank=1)

    coeffs = tuple(
        jax.random.normal(jax.random.PRNGKey(i), shape=Vi.num_dofs)
        for i, Vi in enumerate(V.flatten())
    )
    u = JAXFunction(coeffs, V)

    ua = V.backward(cast(tuple[Array, ...], u.array), N=(pad, pad))
    coeffs_rt = V.forward(ua)

    assert ua[0].shape == pad and ua[1].shape == pad
    assert all(
        jnp.linalg.norm(u.array[i] - coeffs_rt[i]) < ulp(1000) for i in range(len(V))
    )


def test_vectortensorproductspace_project_different_tpspaces():
    N = 20
    bcsD = {"left": {"D": 0}, "right": {"D": 0}}
    bcsB = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
    C0 = FunctionSpace(N, Chebyshev.Chebyshev, bcs=bcsD)
    C1 = FunctionSpace(N, Chebyshev.Chebyshev, bcs=bcsB)
    T0 = TensorProduct(C0, C0)
    T1 = TensorProduct(C1, C1)
    V = CartesianProduct(T0, T1, name="V", rank=1)
    x, y = V.system.base_scalars()
    i, j = V.system.base_vectors()
    coeffs = (
        sp.sin(x * sp.pi) * sp.sin(y * sp.pi) * i + ((1 - x**2) * (1 - y**2)) ** 2 * j
    )
    u = JAXFunction(coeffs, V)
    uej = (
        lambdify((x, y), Dot(coeffs, i).doit())(*V.mesh()),
        lambdify((x, y), Dot(coeffs, j).doit())(*V.mesh()),
    )
    ua = u.backward()
    assert all(
        jnp.linalg.norm(ua[i] - uej[i]) < jnp.sqrt(ulp(10)) for i in range(len(V))
    )


def test_compositetensorproductspace_project():
    N = 20
    bcsD = {"left": {"D": 0}, "right": {"D": 0}}
    bcsB = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
    C0 = FunctionSpace(N, Chebyshev.Chebyshev, bcs=bcsD)
    C1 = FunctionSpace(N, Chebyshev.Chebyshev, bcs=bcsB)
    T0 = TensorProduct(C0, C0)
    T1 = TensorProduct(C1, C1)
    V = CartesianProduct(T0, T1, name="V", rank=1)
    W = CartesianProduct(V, T1, name="W")
    x, y = W.system.base_scalars()
    i, j = W.system.base_vectors()

    coeffs = (
        sp.sin(x * sp.pi) * sp.sin(y * sp.pi) * i + ((1 - x**2) * (1 - y**2)) ** 2 * j
    )
    u = JAXFunction(coeffs, V)
    uej = (
        lambdify((x, y), Dot(coeffs, i).doit())(*V.mesh()),
        lambdify((x, y), Dot(coeffs, j).doit())(*V.mesh()),
    )
    ua = u.backward()
    assert all(
        jnp.linalg.norm(ua[i] - uej[i]) < jnp.sqrt(ulp(10)) for i in range(len(V))
    )


def test_inner_multivar_expression():
    # Multivariable coefficient sqrt(x+y) * u * v
    C = Chebyshev.Chebyshev(5)
    L = Legendre.Legendre(5)
    T = TensorProduct(C, L)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(sp.sqrt(x + y) * u * v, kind="bilinear")
    # For multivar coefficient we expect TensorMatrix object(s)
    assert isinstance(A, TensorMatrix)


def test_inner_linear_only():
    # Only linear form L(v)
    C = Chebyshev.Chebyshev(6)
    v = TestFunction(C)
    x = C.system.x
    b = inner(sp.sin(x) * v, kind="linear")
    assert b.shape[0] == C.N


def test_inner_returns_matrix_and_vector_with_bcs():
    # Non-homogeneous boundary conditions triggers vector addition in bilinear form
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    V = FunctionSpace(6, Legendre.Legendre, bcs=bcs)
    v = TestFunction(V)
    u = TrialFunction(V)
    x = V[0].system.x if isinstance(V, DirectSum) else V.system.x
    A, b = inner(v * u + x * v * u, kind="system")
    # A is dense matrix, b vector
    assert A.shape[0] == A.shape[1]
    assert b.shape[0] == A.shape[0]


def test_vectortensorproductspace_project():
    N: int = 8

    C = Chebyshev.Chebyshev(N)
    T = TensorProduct(C, C)
    V = CartesianProduct(T, T, name="V", rank=1)
    x, y = V.system.base_scalars()
    i, j = V.system.base_vectors()

    u = JAXFunction(y * i + x * j, V)

    ua = V.backward(cast(tuple[Array, ...], u.array))
    xi, yj = T.mesh()
    assert jnp.linalg.norm(ua[0] - yj) < ulp(1000)
    assert jnp.linalg.norm(ua[1] - xi) < ulp(1000)
