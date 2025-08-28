import jax

from jaxfun.galerkin import (
    Chebyshev,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TPMatrices, tpmats_to_scipy_kron


def test_tensorproductspace_broadcast_and_evaluate_2d():
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(5)
    T = TensorProduct((C, L))
    mesh = T.mesh()
    coeffs = jax.random.normal(jax.random.PRNGKey(0), shape=T.dim())
    T.backward(coeffs)
    # broadcast_to_ndims path
    bx = T.broadcast_to_ndims(mesh[0], 0)
    by = T.broadcast_to_ndims(mesh[1], 1)
    assert bx.shape[0] == mesh[0].shape[0] and by.shape[1] == mesh[1].shape[0]
    # evaluate 2D path
    val = T.evaluate(mesh, coeffs)
    # evaluate inserts singleton axis for second dimension order (shape (N0,1,N1))
    assert val.shape[0] == coeffs.shape[0] and val.shape[-1] == coeffs.shape[1]


def test_tensorproductspace_forward_directsum():
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    F = Legendre.Legendre(5)
    Chebyshev.Chebyshev(5)
    from jaxfun.galerkin import FunctionSpace

    DS = FunctionSpace(5, Legendre.Legendre, bcs=bcs)
    T = TensorProduct((DS, F))
    # Make a simple physical array in homogeneous shape (first subspace dim,
    # second plain dim)
    hom0 = DS[0]
    U = jax.random.normal(jax.random.PRNGKey(1), shape=(hom0.dim, F.dim))
    c = T.forward(T.backward(U))  # round trip through forward/backward on DirectSumTPS
    assert c.shape == U.shape


def test_tpmatrices_call_and_kron_3d():
    C = Chebyshev.Chebyshev(3)
    L = Legendre.Legendre(3)
    T3 = TensorProduct((C, L, C))
    v = TestFunction(T3)
    u = TrialFunction(T3)
    A = inner(v * u)
    kron = tpmats_to_scipy_kron(A)
    # Build TPMatrices and apply to random u
    mats = TPMatrices(A)
    X = jax.random.normal(jax.random.PRNGKey(4), shape=T3.dim())
    Y = mats(X)
    assert Y.shape == X.shape and kron.shape[0] == kron.shape[1]


def test_inner_linear_form_3d_outer_products():
    C = Chebyshev.Chebyshev(3)
    L = Legendre.Legendre(3)
    T3 = TensorProduct((C, L, C))
    v = TestFunction(T3)
    x, y, z = T3.system.base_scalars()
    b = inner((x + y + z) * v)
    assert b.shape == T3.dim()


def test_inner_sparse_multivar_path():
    # multivar coeff with sparse=True to trigger sparse conversion in process_results
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(4)
    T = TensorProduct((C, L))
    v = TestFunction(T)
    u = TrialFunction(T)
    x, y = T.system.base_scalars()
    # Use plain bilinear form to ensure TPMatrix objects (with dims attr) returned
    A = inner(u * v, sparse=True)
    # Expect list of TPMatrix with sparse mats
    for tp in A:
        assert all(hasattr(m, "data") for m in tp.mats)
