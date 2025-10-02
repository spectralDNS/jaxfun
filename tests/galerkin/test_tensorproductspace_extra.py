import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import (
    DirectSumTPS,
    TPMatrices,
    tpmats_to_scipy_sparse,
)


def test_directsum_two_inhomogeneous_bnd_assembly_and_backward():
    # Exercise two_inhomogeneous branch lines 211-233 etc.
    bcs1 = {"left": {"D": 1}, "right": {"D": 2}}
    bcs2 = {"left": {"D": 3}, "right": {"D": 4}}
    F1 = FunctionSpace(5, Legendre.Legendre, bcs=bcs1)
    F2 = FunctionSpace(6, Legendre.Legendre, bcs=bcs2)
    T = TensorProduct(F1, F2)
    assert isinstance(T, DirectSumTPS)
    # Coefficient array for homogeneous parts
    hom0 = F1[0]
    hom1 = F2[0]
    c = jnp.zeros((hom0.dim, hom1.dim))
    u = T.backward(c)  # triggers boundary reconstruction path
    assert u.shape[0] == hom0.num_quad_points and u.shape[1] == hom1.num_quad_points


def test_tensorproduct_get_homogeneous_and_tpmatrices_precond():
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    F = FunctionSpace(5, Chebyshev.Chebyshev, bcs=bcs)
    L = Legendre.Legendre(5)
    T = TensorProduct(F, L)
    assert isinstance(T, DirectSumTPS)
    H = T.get_homogeneous()
    assert isinstance(H, type(next(iter(T.tpspaces.values()))))
    # Use homogeneous space for matrix assembly (avoids bc vector-only path)
    v = TestFunction(H)
    u = TrialFunction(H)
    A = inner(v * u)
    mats = TPMatrices([m for m in A if hasattr(m, "M")])
    X = jax.random.normal(jax.random.PRNGKey(0), shape=T.num_dofs)
    Z = mats.precond(X)
    assert Z.shape == X.shape


def test_multivar_and_linear_bcs_branch():
    # Craft expr with multivar coefficient and boundary conditions to hit
    # combined branches
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    F = FunctionSpace(4, Legendre.Legendre, bcs=bcs)
    L = Legendre.Legendre(4)
    T = TensorProduct(F, L)
    x, y = T.system.base_scalars()
    v = TestFunction(T)
    u = TrialFunction(T)
    # multivar coefficient (x+y) and linear JAXFunction coefficient in same expression
    coeffs = jax.random.normal(jax.random.PRNGKey(1), shape=T.num_dofs)
    from jaxfun.galerkin.arguments import JAXFunction

    jf = JAXFunction(coeffs, T)
    A = inner((sp.sqrt(x + y) * u * v) + jf * v, return_all_items=True)
    # Should return tuple (aresults,bresults)
    assert isinstance(A, tuple)


def test_tpmats_to_scipy_sparse():
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(4)
    T = TensorProduct(C, L)
    v = TestFunction(T)
    u = TrialFunction(T)
    A = inner(v * u)
    S = tpmats_to_scipy_sparse(A)
    assert len(S) == len(A)
