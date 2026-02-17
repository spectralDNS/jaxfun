from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.inner import inner, project
from jaxfun.galerkin.tensorproductspace import (
    TPMatrices,
    TPMatrix,
    tpmats_to_scipy_kron,
)
from jaxfun.utils.common import ulp


def test_tensorproductspace_broadcast_and_evaluate_2d():
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(5)
    T = TensorProduct(C, L)
    mesh = T.mesh()
    coeffs = jax.random.normal(jax.random.PRNGKey(0), shape=T.num_dofs)
    T.backward(coeffs)
    # broadcast_to_ndims path
    bx = T.broadcast_to_ndims(mesh[0], 0)
    by = T.broadcast_to_ndims(mesh[1], 1)
    assert bx.shape[0] == mesh[0].shape[0] and by.shape[1] == mesh[1].shape[0]
    # evaluate 2D path
    val = T.evaluate_mesh(mesh, coeffs)
    # evaluate inserts singleton axis for second dimension order (shape (N0,1,N1))
    assert val.shape[0] == coeffs.shape[0] and val.shape[-1] == coeffs.shape[1]


def test_tensorproductspace_forward_directsum():
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    F = Legendre.Legendre(5)
    # Chebyshev.Chebyshev(5)
    from jaxfun.galerkin import FunctionSpace

    DS = FunctionSpace(5, Legendre.Legendre, bcs=bcs)
    from jaxfun.galerkin.composite import DirectSum

    assert isinstance(DS, DirectSum)
    T = TensorProduct(DS, F)
    # Make a simple physical array in homogeneous shape (first subspace dim,
    # second plain dim)
    hom0 = DS[0]
    U = jax.random.normal(jax.random.PRNGKey(1), shape=(hom0.dim, F.dim))
    c = T.forward(T.backward(U))  # round trip through forward/backward on DirectSumTPS
    assert c.shape == U.shape
    assert jnp.allclose(c, U)


def test_tpmatrices_call_and_kron_3d():
    C = Chebyshev.Chebyshev(3)
    L = Legendre.Legendre(3)
    T3 = TensorProduct(C, L, C)
    v = TestFunction(T3)
    u = TrialFunction(T3)
    A = inner(v * u)
    assert isinstance(A, list)
    A = cast(list[TPMatrix], A)
    kron = tpmats_to_scipy_kron(A)
    # Build TPMatrices and apply to random u
    mats = TPMatrices(A)
    X = jax.random.normal(jax.random.PRNGKey(4), shape=T3.num_dofs)
    Y = mats(X)
    assert Y.shape == X.shape and kron.shape[0] == kron.shape[1]


def test_inner_linear_form_3d_outer_products():
    C = Chebyshev.Chebyshev(3)
    L = Legendre.Legendre(3)
    T3 = TensorProduct(C, L, C)
    v = TestFunction(T3)
    x, y, z = T3.system.base_scalars()
    b = inner((x + y + z) * v)
    assert isinstance(b, jax.Array)
    assert b.shape == T3.num_dofs


def test_inner_sparse_multivar_path():
    # multivar coeff with sparse=True to trigger sparse conversion in process_results
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(4)
    T = TensorProduct(C, L)
    v = TestFunction(T)
    u = TrialFunction(T)
    x, y = T.system.base_scalars()
    # Use plain bilinear form to ensure TPMatrix objects (with dims attr) returned
    A = inner(u * v, sparse=True)
    assert isinstance(A, list)
    A_tp = cast(list[TPMatrix], A)
    # Expect list of TPMatrix with sparse mats
    for tp in A_tp:
        assert all(hasattr(m, "data") for m in tp.mats)


@pytest.mark.slow
def test_directsum_two_inhomogeneous_bnd_evaluate():
    from jaxfun.coordinates import x, y

    ue = sp.exp(-(x**2 + y**2))
    N = 16
    bcsx = {"left": {"D": ue.subs(x, 0)}, "right": {"D": ue.subs(x, 1)}}
    bcsy = {"left": {"D": ue.subs(y, 0)}, "right": {"D": ue.subs(y, 1)}}
    Dx = FunctionSpace(N, Legendre.Legendre, bcs=bcsx, name="Dx", domain=(0, 1))
    Dy = FunctionSpace(N, Legendre.Legendre, bcs=bcsy, name="Dy", domain=(0, 1))
    T = TensorProduct(Dx, Dy, name="T")
    _v = TestFunction(T, name="v")
    _u = TrialFunction(T, name="u")
    ue = T.system.expr_psi_to_base_scalar(ue)
    uf = project(ue, T)
    x, y = T.system.base_scalars()
    u0 = T.evaluate(jnp.array([0.5, 0.5]), uf)  # triggers boundary reconstruction path
    assert abs(u0 - ue.subs({x: 0.5, y: 0.5})) < ulp(100)
    u1 = T.evaluate_mesh([jnp.array([[0.5]]), jnp.array([[0.5]])], uf)
    assert abs(u1[0, 0] - ue.subs({x: 0.5, y: 0.5})) < ulp(100)
    u0 = T.evaluate(jnp.array([[0.5, 0.5], [0.6, 0.6]]), uf)
    assert abs(u0[0] - ue.subs({x: 0.5, y: 0.5})) < ulp(100)
    assert abs(u0[1] - ue.subs({x: 0.6, y: 0.6})) < ulp(100)
    u1 = T.evaluate_mesh([jnp.array([[0.5, 0.6]]), jnp.array([[0.5, 0.6]])], uf)
    assert abs(u1[0, 0] - ue.subs({x: 0.5, y: 0.5})) < ulp(100)
    assert abs(u1[0, 1] - ue.subs({x: 0.5, y: 0.6})) < ulp(100)
    assert abs(u1[1, 0] - ue.subs({x: 0.6, y: 0.5})) < ulp(100)
    assert abs(u1[1, 1] - ue.subs({x: 0.6, y: 0.6})) < ulp(100)
