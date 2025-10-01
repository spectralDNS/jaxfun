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
from jaxfun.galerkin.arguments import JAXFunction, ScalarFunction, VectorFunction
from jaxfun.galerkin.forms import split_coeff
from jaxfun.galerkin.inner import inner, project
from jaxfun.galerkin.tensorproductspace import DirectSumTPS, VectorTensorProductSpace
from jaxfun.utils.common import Domain, ulp


def test_vector_tensor_product_space_and_jaxfunction_latex_and_matmul():
    C = Chebyshev.Chebyshev(4)
    # Need at least 2D tensorspace for sub_system logic
    TP = TensorProduct(C, C)
    _VT = VectorTensorProductSpace(TP)  # rank 1 space
    coeffs = jax.random.normal(jax.random.PRNGKey(0), shape=(C.N, C.N))
    jf = JAXFunction(coeffs, TP, name="U")
    # Latex function (no bold since rank==0 for scalar TensorProductSpace)
    _ = jf._latex()
    # __matmul__ / __rmatmul__
    a = jnp.ones((C.N, C.N))
    left = jf @ a
    right = a @ jf
    assert left.shape == (C.N, C.N)
    assert right.shape == (C.N, C.N)


def test_inner_return_all_items_and_sparse_paths():
    C = Chebyshev.Chebyshev(5)
    x = C.system.x
    u = TrialFunction(C)
    v = TestFunction(C)
    # return_all_items True
    Aall, Ball = inner(v * u + x * v * u, return_all_items=True)
    assert isinstance(Aall, list) and len(Aall) >= 1
    # sparse conversion 1D
    As = inner(v * u, sparse=True)
    # Should be sparse matrix (BCOO) or list containing them
    from jax.experimental.sparse import BCOO

    assert isinstance(As, BCOO)
    # Pure linear form only vector return
    b = inner(sp.sin(x) * v)
    assert b.shape[0] == C.N
    # 2D sparse path (matrices become sparse individually)
    T = TensorProduct(Chebyshev.Chebyshev(4), Chebyshev.Chebyshev(4))
    v2 = TestFunction(T)
    u2 = TrialFunction(T)
    A2 = inner(v2 * u2, sparse=True)
    # Expect list of TPMatrix with sparse mats
    for tp in A2:
        from jax.experimental.sparse import BCOO

        assert all(isinstance(m, BCOO) for m in tp.mats)


def test_split_coeff_mul_and_add_jaxf():
    C = Chebyshev.Chebyshev(4)
    coeffs = jax.random.normal(jax.random.PRNGKey(1), shape=(C.N,))
    jf = JAXFunction(coeffs, C, name="U")
    # Add (number + JAXFunction) path; scale stays 1, bilinear captured
    d2 = split_coeff(sp.Integer(2) + jf)
    assert d2["bilinear"] == 2 and d2["linear"]["jaxfunction"] is None


def test_directsum_tps_two_inhomogeneous():
    bcs1 = {"left": {"D": 1}, "right": {"D": 2}}
    bcs2 = {"left": {"D": 3}, "right": {"D": 4}}
    F1 = FunctionSpace(4, Legendre.Legendre, bcs=bcs1)
    F2 = FunctionSpace(4, Legendre.Legendre, bcs=bcs2)
    # Ensure both are DirectSum
    from jaxfun.galerkin.composite import DirectSum

    assert isinstance(F1, DirectSum) and isinstance(F2, DirectSum)
    # Access component spaces and ensure boundary values converted
    _ = F1[1].bcs.orderedvals(), F2[1].bcs.orderedvals()
    T = TensorProduct(F1, F2)
    assert isinstance(T, DirectSumTPS)


def test_functionspace_variants():
    # No bcs returns base space
    L = FunctionSpace(6, Legendre.Legendre)
    assert not hasattr(L, "basespaces")
    # Homogeneous bcs returns Composite
    H = FunctionSpace(6, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    assert H.bcs.is_homogeneous()
    # Non-homogeneous returns DirectSum
    D = FunctionSpace(6, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 2}})
    from jaxfun.galerkin.composite import DirectSum

    assert D.evaluate(jnp.array([1.0]), jnp.zeros(4)) == 2.0
    assert D.evaluate(jnp.array([-1.0]), jnp.zeros(4)) == 1.0
    assert isinstance(D, DirectSum)


def test_orthogonal_mappings_and_domain_factor():
    L = Legendre.Legendre(5, domain=Domain(-2, 2))
    x = L.system.x
    expr = x**2 + 1
    ref = L.map_expr_reference_domain(expr)
    tru = L.map_expr_true_domain(ref)
    # Roundtrip
    assert sp.simplify(tru - expr) == 0
    # Numeric mapping
    pts = jnp.linspace(-2.0, 2.0, 5)
    Xref = jnp.array([L.map_reference_domain(p) for p in pts])
    Xtrue = jnp.array([L.map_true_domain(X) for X in Xref])
    assert jnp.allclose(Xtrue, pts)


def test_fourier_backward_truncation():
    F = Fourier.Fourier(8)
    coeffs = jax.random.normal(jax.random.PRNGKey(2), shape=(F.N,))
    u4 = F.backward(coeffs, N=4)  # truncation path
    assert u4.shape[0] == 4


def test_chebyshev_matrices_branches():
    C = Chebyshev.Chebyshev(6)
    from jaxfun.galerkin import Chebyshev as Cmod

    # i=0,j=0
    m00 = Cmod.matrices((C, 0), (C, 0))
    assert m00 is not None
    # i=0,j=1 maybe None if not enough odd indices
    _ = Cmod.matrices((C, 0), (C, 1))
    # i=0,j=2 even derivative
    _ = Cmod.matrices((C, 0), (C, 2))
    # i=2,j=0 transpose path
    _ = Cmod.matrices((C, 2), (C, 0))
    # Unknown path returns None
    assert Cmod.matrices((C, 3), (C, 3)) is None


def test_tensorproductspace_3d_paths_and_mapping():
    C = Chebyshev.Chebyshev(3)
    T3 = TensorProduct(C, C, C)
    # Random coefficients
    u = jax.random.normal(jax.random.PRNGKey(3), shape=T3.num_dofs)
    # forward/backward roundtrip (approximate due to quadrature discretization)
    c = T3.forward(u)
    uh = T3.backward(c)
    assert uh.shape == u.shape
    # Evaluate path
    mesh = T3.mesh()
    val = T3.evaluate_mesh(mesh, c)
    # evaluation returns broadcasted shape (dim interleaved); collapse via squeeze
    assert jnp.squeeze(val).shape == u.shape
    # Padded path
    Tp = T3.get_padded((4, 3, 3))
    assert Tp.shape()[0] == 4
    # Mapping of expression through composed map_expr_true_domain
    x, y, z = T3.system.base_scalars()
    expr = x + y + z
    expr2 = T3.map_expr_true_domain(expr)
    assert expr2.free_symbols == expr.free_symbols


def test_project_function():
    # cover project (2D) path
    C = Chebyshev.Chebyshev(4)
    L = Legendre.Legendre(4)
    T = TensorProduct(C, L)
    x, y = T.system.base_scalars()
    ue = sp.chebyshevt(1, x) * sp.legendre(2, y)
    uh = project(ue, T)
    assert abs(uh[1, 2] - 1.0) < ulp(100)
    assert uh.shape == T.dim


def test_scalar_vector_function_pretty_and_sympy():
    C = Chebyshev.Chebyshev(3)
    s = ScalarFunction("f", C.system)
    v = VectorFunction("g", C.system)
    # exercise pretty and sympy str methods
    _ = s._pretty(), s._sympystr(None), v._pretty(), v._sympystr(None)
