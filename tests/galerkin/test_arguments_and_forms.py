import jax
import jax.numpy as jnp

from jaxfun.galerkin import Chebyshev, FunctionSpace, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction, ScalarFunction, VectorFunction
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.inner import inner
from jaxfun.utils.common import ulp


def test_scalar_vector_function_print_and_backward():
    C = Chebyshev.Chebyshev(4)
    s = ScalarFunction("f", C.system)
    v = VectorFunction("g", C.system)
    # Printing / latex
    _ = str(s), s._latex(), str(v), v._latex()
    # Use JAXFunction in form
    coeffs = jax.random.normal(jax.random.PRNGKey(0), shape=(C.N,))
    jf = JAXFunction(coeffs, C, name="U")
    u = TrialFunction(C)
    v = TestFunction(C)
    M, b = inner(v * (u - jf), kind="system")
    uh = M.solve(b)
    assert jnp.linalg.norm(uh - coeffs) < ulp(100)
    # Should approximate jf coefficients (diagonal mass matrix scaling)
    assert uh.shape[0] == C.N


def test_forms_split_linear_and_bilinear():
    V = FunctionSpace(5, Chebyshev.Chebyshev, bcs={"left": {"D": 0}, "right": {"D": 0}})
    v = TestFunction(V)
    u = TrialFunction(V)
    x = V[0].system.x if isinstance(V, DirectSum) else V.system.x
    # Form with bilinear and linear parts
    A = inner(x * v * u + v * u, kind="bilinear")
    assert hasattr(A, "shape")
    assert A.shape[0] == A.shape[1]


if __name__ == "__main__":
    test_scalar_vector_function_print_and_backward()
    # test_forms_split_linear_and_bilinear()


def test_test_and_trial_functions_rebuild():
    import copy

    import pytest
    import sympy as sp

    V = Chebyshev.Chebyshev(4)
    x = V.system.x
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u")
    ut = TrialFunction(V, name="w", transient=True)
    for f in (v, u, ut):
        assert f.func(*f.args) is f
        assert copy.copy(f) is f
        assert f.free_symbols <= {x, V.system.base_time()}
    # Placeholders for basis functions have no values to substitute
    with pytest.raises(NotImplementedError, match="basis functions"):
        u.subs(x, 0.5)
    with pytest.raises(NotImplementedError, match="basis functions"):
        (u * v).subs(x, sp.Symbol("y"))


def test_dir_lists_the_jaxfun_api():
    import sympy as sp

    V = Chebyshev.Chebyshev(4)
    u = TrialFunction(V, transient=True)
    names = dir(u)
    assert {"functionspace", "name", "transient", "doit", "diff", "subs"} <= set(names)
    assert "evalf" not in names and "rewrite" not in names
    assert "evalf" in dir(JAXFunction(jnp.ones(4), V))
    # Hidden names still work
    assert u.rewrite(sp.cos) is u
