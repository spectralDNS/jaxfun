import jax
import jax.numpy as jnp

from jaxfun.galerkin import Chebyshev, TensorProduct, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction, ScalarFunction, VectorFunction


def test_jaxfunction_doit_and_matmul_rank1():
    C = Chebyshev.Chebyshev(4)
    T = TensorProduct(C, C)
    coeffs = jax.random.normal(jax.random.PRNGKey(10), shape=T.dim)
    jf = JAXFunction(coeffs, T, name="A")
    expr = jf.doit()
    # Should produce Jaxf * TrialFunction structure
    assert hasattr(expr, "args")
    assert expr.args[0].__class__.__name__ == "Jaxf"
    a = jnp.ones(T.dim)
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
