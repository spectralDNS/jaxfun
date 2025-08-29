import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import tpmats_to_scipy_kron
from jaxfun.utils.common import ulp


def test_tensorproduct_forward_backward_padding_fourier():
    # Padding path for Fourier
    F = Fourier.Fourier(8)
    T = TensorProduct((F, F))
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    M, b = inner(v * (u - sp.sin(x) * sp.sin(y)))
    # Solve
    uh = jnp.linalg.solve(M[0].mat, b.flatten()).reshape(T.dim())
    # Backward on padded grid
    up = T.backward(uh, N=(12, 8))
    # Make sure shape matches requested padding (only first axis padded)
    assert up.shape == (12, 8)


def test_tensorproduct_directsum_tps_forward_backward():
    # Create 2D with one inhomogeneous dirichlet dimension
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    C = FunctionSpace(6, Legendre.Legendre, bcs=bcs)
    L = Legendre.Legendre(6)
    T = TensorProduct((C, L))
    assert hasattr(T, "tpspaces")  # DirectSumTPS
    # Build form (Poisson like)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    A, b = inner(sp.diff(u, x) * sp.diff(v, x) + sp.diff(u, y) * sp.diff(v, y))
    # Ensure we got list of TPMatrix objects
    for tp in A:
        _ = tp.mat  # access kron
    # Add rhs from non-zero bc lifting by calling backward on zero coeffs.
    # Need homogeneous tensor space to get coefficient shape.
    # Build shape from homogeneous part (first component of direct sum)
    # and second plain Legendre space
    hom0 = C[0]  # homogeneous composite
    uh = jnp.zeros((hom0.dim, L.dim))
    _ = T.backward(uh)


def test_tp_matrix_and_preconditioner():
    C = Chebyshev.Chebyshev(6)
    L = Legendre.Legendre(6)
    T = TensorProduct((C, L))
    v = TestFunction(T)
    u = TrialFunction(T)
    A = inner(v * u)
    tp = A[0]
    X = jax.random.normal(jax.random.PRNGKey(0), shape=T.dim())
    Y = tp(X)
    # Check manual matmul versus kron
    Y2 = tp.mats[0] @ X @ tp.mats[1].T
    assert jnp.linalg.norm(Y - Y2) < ulp(100)
    # Preconditioner
    Z = tp.precond(X)
    assert Z.shape == X.shape
    # Convert to scipy kron
    _ = tpmats_to_scipy_kron([tp])


def test_inner_multivar_expression():
    # Multivariable coefficient sqrt(x+y) * u * v
    C = Chebyshev.Chebyshev(5)
    L = Legendre.Legendre(5)
    T = TensorProduct((C, L))
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(sp.sqrt(x + y) * u * v)
    # For multivar coefficient we expect TensorMatrix object(s)
    assert any(hasattr(a, "mat") and not hasattr(a, "mats") for a in A)


def test_process_results_linear_only():
    # Only linear form L(v)
    C = Chebyshev.Chebyshev(6)
    v = TestFunction(C)
    x = C.system.x
    b = inner(sp.sin(x) * v)
    assert b.shape[0] == C.N


def test_inner_returns_matrix_and_vector_with_bcs():
    # Non-homogeneous boundary conditions triggers vector addition in bilinear form
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    V = FunctionSpace(6, Legendre.Legendre, bcs=bcs)
    v = TestFunction(V)
    u = TrialFunction(V)
    x = V[0].system.x if hasattr(V, "basespaces") else V.system.x
    A, b = inner(v * u + x * v * u)
    # A is dense matrix, b vector
    assert A.shape[0] == A.shape[1]
    assert b.shape[0] == A.shape[0]
