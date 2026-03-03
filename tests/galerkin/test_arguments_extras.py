import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import (
    Chebyshev,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.arguments import (
    JAXFunction,
    ScalarFunction,
    VectorFunction,
    evaluate_jaxfunction_expr,
)


def test_jaxfunction_doit_and_matmul_rank1():
    C = Chebyshev.Chebyshev(4)
    T = TensorProduct(C, C)
    coeffs = jax.random.normal(jax.random.PRNGKey(10), shape=T.num_dofs)
    jf = JAXFunction(coeffs, T, name="A")
    expr = jf.doit(linear=True)
    # Should produce Jaxc * TrialFunction structure
    assert hasattr(expr, "args")
    assert expr.args[0].__class__.__name__ == "Jaxc"
    a = jnp.ones(T.num_dofs)
    _ = jf @ a
    _ = a @ jf


def test_scalar_vector_function_latex_rank1():
    C = Chebyshev.Chebyshev(3)
    s = ScalarFunction("f", C.system)
    v = VectorFunction("g", C.system)
    _ = s._latex(), v._latex()


def test_trial_test_function_str_repr_symmetry():
    C = Chebyshev.Chebyshev(3)
    v = TestFunction(C)
    u = TrialFunction(C)
    assert str(v) != str(u)  # ensure distinct naming
    assert v.functionspace is C and u.functionspace is C


def test_jaxfunction_no_bold_for_rank0():
    C = Chebyshev.Chebyshev(4)
    coeffs = jax.random.normal(jax.random.PRNGKey(11), shape=(C.N,))
    jf = JAXFunction(coeffs, C)
    assert "mathbf" not in jf._latex()


def test_evaluate_jaxfunction_expr_reference_points_1d():
    D = Legendre.Legendre(8, domain=Domain(-2, 2))
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    x = D.system.x
    X = D.quad_points_and_weights()[0]

    u_ref = evaluate_jaxfunction_expr(uf.doit(), X)
    assert jnp.linalg.norm(u_ref - D.evaluate(X, uf.array)) < 1e-10

    u2_ref = evaluate_jaxfunction_expr(uf.doit() ** 2, X)
    assert jnp.linalg.norm(u2_ref - u_ref**2) < 1e-8

    du_ref = evaluate_jaxfunction_expr(sp.diff(uf.doit(), x), X)
    assert (
        jnp.linalg.norm(du_ref - D.evaluate_derivative_reference(X, uf.array, k=1))
        < 1e-8
    )


def test_evaluate_jaxfunction_expr_reference_points_tensor():
    D = Legendre.Legendre(6, domain=Domain(-2, 2))
    T = TensorProduct(D, D)
    w = JAXFunction(jnp.ones(T.num_dofs), T)
    x, y = T.system.base_scalars()
    X = T.mesh_reference()

    w_ref = evaluate_jaxfunction_expr(w.doit(), X)
    assert jnp.linalg.norm(w_ref - T.evaluate_mesh_reference(X, w.array, True)) < 1e-8

    dw_dx_ref = evaluate_jaxfunction_expr(sp.diff(w.doit(), x), X)
    assert (
        jnp.linalg.norm(
            dw_dx_ref - T.evaluate_derivative_reference(X, w.array, k=(1, 0))
        )
        < 1e-8
    )

    dw_dy_ref = evaluate_jaxfunction_expr(sp.diff(w.doit(), y), X)
    assert (
        jnp.linalg.norm(
            dw_dy_ref - T.evaluate_derivative_reference(X, w.array, k=(0, 1))
        )
        < 1e-8
    )
