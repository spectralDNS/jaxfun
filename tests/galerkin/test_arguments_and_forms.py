from typing import cast

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
    M, b = inner(v * (u - jf))
    uh = jnp.linalg.solve(M, b)
    assert jnp.linalg.norm(uh - coeffs) < ulp(100)
    # Should approximate jf coefficients (diagonal mass matrix scaling)
    assert uh.shape[0] == C.N


def test_forms_split_linear_and_bilinear():
    V = FunctionSpace(5, Chebyshev.Chebyshev, bcs={"left": {"D": 0}, "right": {"D": 0}})
    v = TestFunction(V)
    u = TrialFunction(V)
    x = V[0].system.x if isinstance(V, DirectSum) else V.system.x
    # Form with bilinear and linear parts
    A = inner(x * v * u + v * u)
    assert hasattr(A, "shape")
    A = cast(jax.Array, A)
    assert A.shape[0] == A.shape[1]
