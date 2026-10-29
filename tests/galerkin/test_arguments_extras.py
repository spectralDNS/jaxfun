import jax
import jax.numpy as jnp
import sympy as sp
from sympy.vector import VectorAdd

from jaxfun import Domain
from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin import (
    CartesianProduct,
    Chebyshev,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.arguments import JAXFunction, ScalarFunction, VectorFunction
from jaxfun.utils.common import ulp


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


def test_vector_component_is_projection_of_the_field():
    r, theta = sp.symbols("r theta", real=True, positive=True)
    C = get_CoordSys("C", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))
    L0 = Legendre.Legendre(3, domain=Domain(1, 2))
    L1 = Legendre.Legendre(3, domain=Domain(0, 1))
    T = TensorProduct(L0, L1, system=C)
    u = TrialFunction(CartesianProduct(T, T, rank=1), name="u")
    # A component is the contravariant u^k = u·b^k, not a new trial function
    assert u[1].atoms(TrialFunction) == {u}
    full = u.doit()
    assert isinstance(full, VectorAdd)
    for k, bk in enumerate(C.base_vectors()):
        assert sp.simplify(u[k].doit() - full.components[bk]) == 0
    ux, uy = u
    assert ux == u[0] and uy == u[1]


def test_vector_projection_gives_contravariant_components():
    r, theta = sp.symbols("r theta", real=True, positive=True)
    C = get_CoordSys("C", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))
    L0 = Legendre.Legendre(4, domain=Domain(1, 2))
    L1 = Legendre.Legendre(4, domain=Domain(0, 1))
    T = TensorProduct(L0, L1, system=C)
    V = CartesianProduct(T, T, rank=1)
    r, theta = C.base_scalars()
    # The coefficients are u^i = u·b^i, so u^theta = r**2, not u_theta = r**4
    u = JAXFunction(r * theta * C.b_r + r**2 * C.b_theta, V)
    ua = V.backward(u.array)
    ri, tj = T.mesh()
    assert jnp.linalg.norm(ua[0] - ri * tj) < ulp(1000)
    assert jnp.linalg.norm(ua[1] - ri**2) < ulp(1000)
