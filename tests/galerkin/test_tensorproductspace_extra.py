from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    ChebyshevU,
    DirectSum,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    Ultraspherical,
    VectorTensorProductSpace,
    tpmats_to_kron,
)
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import DirectSumTPS
from jaxfun.utils.common import ulp


def test_directsum_two_inhomogeneous_bnd_assembly_and_backward():
    # Exercise two_inhomogeneous branch lines 211-233 etc.
    bcs1 = {"left": {"D": 1}, "right": {"D": 2}}
    bcs2 = {"left": {"D": 3}, "right": {"D": 4}}
    F1 = FunctionSpace(5, Legendre.Legendre, bcs=bcs1)
    F2 = FunctionSpace(6, Legendre.Legendre, bcs=bcs2)
    from jaxfun.galerkin.composite import DirectSum

    assert isinstance(F1, DirectSum) and isinstance(F2, DirectSum)
    T = TensorProduct(F1, F2)
    assert isinstance(T, DirectSumTPS)
    # Coefficient array for homogeneous parts
    hom0 = F1[0]
    hom1 = F2[0]
    c = jnp.zeros((hom0.dim, hom1.dim))
    u = T.backward(c)  # triggers boundary reconstruction path
    assert u.shape[0] == hom0.num_quad_points and u.shape[1] == hom1.num_quad_points


def test_tensorproduct_get_homogeneous():
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
    assert isinstance(A, list)


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
    from jaxfun.galerkin import JAXFunction

    jf = JAXFunction(coeffs, T)
    A = inner((sp.sqrt(x + y) * u * v) + jf * v, return_all_items=True)
    # Should return tuple (aresults,bresults)
    assert isinstance(A, tuple)


def test_tpmats_to_kron():
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(4)
    T = TensorProduct(C, L)
    v = TestFunction(T)
    u = TrialFunction(T)
    A = inner(v * u)
    assert isinstance(A, list)
    S = tpmats_to_kron(A)
    assert S.shape == (T.dim, T.dim)
    # Compare to dense version
    A_dense = sum(mat.mat.todense() for mat in A)
    assert jnp.allclose(S.todense(), A_dense)


@pytest.mark.parametrize(
    "space",
    (
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Ultraspherical.Ultraspherical,
    ),
)
def test_tensorproductspace_to_orthogonal(space):
    N1, N2 = 5, 6
    F1 = FunctionSpace(N1, space, bcs={"left": {"D": 0}, "right": {"D": 0}})
    F2 = FunctionSpace(N2, space, bcs={"left": {"N": 0}, "right": {"N": 0}})
    T = TensorProduct(F1, F2)
    O = T.get_orthogonal()
    c = jax.random.normal(jax.random.PRNGKey(0), shape=T.num_dofs)
    c0 = T.to_orthogonal(c)
    y0 = T.backward(c)
    y1 = O.backward(c0)
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))


def test_directsumtps_to_orthogonal():
    bcs1 = {"left": {"D": 1}, "right": {"D": 2}}
    bcs2 = {"left": {"D": 3}, "right": {"D": 4}}
    F1 = FunctionSpace(5, Legendre.Legendre, bcs=bcs1)
    F2 = FunctionSpace(6, Legendre.Legendre, bcs=bcs2)
    assert isinstance(F1, DirectSum) and isinstance(F2, DirectSum)
    T = TensorProduct(F1, F2)
    O = T.get_orthogonal()
    assert isinstance(T, DirectSumTPS)
    c = jax.random.normal(jax.random.PRNGKey(0), shape=T.num_dofs)
    c0 = T.to_orthogonal(c)
    y0 = T.backward(c)
    y1 = O.backward(c0)
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))


def test_vectortensorproductspace_to_orthogonal():
    N1, N2 = 5, 6
    F1 = FunctionSpace(N1, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    F2 = FunctionSpace(N2, Legendre.Legendre, bcs={"left": {"N": 0}, "right": {"N": 0}})
    T = TensorProduct(F1, F2)
    V = VectorTensorProductSpace(T, name="V")
    O = V.get_orthogonal()
    c = jax.random.normal(jax.random.PRNGKey(0), shape=V.num_dofs)
    c0 = V.to_orthogonal(c)
    y0 = V.backward(c)
    y1 = O.backward(c0)
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))


def test_directsumtps():
    bcs1 = {"left": {"D": 1}, "right": {"D": 2}}
    bcs2 = {"left": {"D": 3}, "right": {"D": 4}}
    N = (5, 6)
    F1 = FunctionSpace(N[0], Legendre.Legendre, bcs=bcs1, domain=(-2, 2))
    F2 = FunctionSpace(N[1], Legendre.Legendre, bcs=bcs2, domain=(-2, 2))
    assert isinstance(F1, DirectSum) and isinstance(F2, DirectSum)
    T: DirectSumTPS = cast(DirectSumTPS, TensorProduct(F1, F2))

    c = jax.random.normal(jax.random.PRNGKey(0), shape=T.num_dofs)
    a = []
    for f, v in T.tpspaces.items():
        a.append(v.backward_primitive(T.bndvals.get(f, c), k=(1, 1), N=N))
    z0 = jnp.sum(jnp.array(a), axis=0)
    z1 = T.backward_primitive(c, k=(1, 1), N=N)
    assert jnp.linalg.norm(z0 - z1) < jnp.sqrt(ulp(100))

    a = []
    for f, v in T.tpspaces.items():
        a.append(v.backward(T.bndvals.get(f, c), N=N))
    z0 = jnp.sum(jnp.array(a), axis=0)
    z1 = T.backward(c, N=N)
    assert jnp.linalg.norm(z0 - z1) < jnp.sqrt(ulp(100))

    xj = T.flatmesh(kind="uniform", N=N)
    a = []
    for f, v in T.tpspaces.items():
        a.append(v.evaluate(xj, T.bndvals.get(f, c), True))
    z0 = jnp.sum(jnp.array(a), axis=0)
    z1 = T.evaluate(xj, c, True)
    assert jnp.linalg.norm(z0 - z1) < jnp.sqrt(ulp(100))

    a = []
    for f, v in T.tpspaces.items():
        a.append(v.evaluate_derivative(xj, T.bndvals.get(f, c), (1, 1)))
    z0 = jnp.sum(jnp.array(a), axis=0)
    z1 = T.evaluate_derivative(xj, c, (1, 1))
    assert jnp.linalg.norm(z0 - z1) < jnp.sqrt(ulp(100))

    a = []
    xj = T.mesh(kind="uniform", N=N)
    for f, v in T.tpspaces.items():
        a.append(v.evaluate_mesh(xj, T.bndvals.get(f, c), True))
    z0 = jnp.sum(jnp.array(a), axis=0)
    z1 = T.evaluate_mesh(xj, c, True)
    assert jnp.linalg.norm(z0 - z1) < jnp.sqrt(ulp(100))
