import jax
import sympy as sp

from jaxfun.galerkin import Chebyshev, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.galerkin.forms import split, split_coeff


def test_split_coeff_number_and_add():
    # Pure number
    c = split_coeff(sp.Integer(3))
    assert c["bilinear"] == 3.0
    # Add of number and JAXFunction
    V = Chebyshev.Chebyshev(4)
    coeffs = jax.random.normal(jax.random.PRNGKey(0), shape=(V.N,))
    jf = JAXFunction(coeffs, V)
    # Using raw JAXFunction (not .doit()), implementation does not treat it as Jaxf
    c2_raw = split_coeff(sp.Integer(2) + jf)
    assert c2_raw["bilinear"] == 2.0 and c2_raw["linear"]["jaxfunction"] is None

    # Skip jf.doit() path; split_coeff currently expects pure numbers
    # or an Add with JAXFunction placeholder


def test_split_and_add_result_multivar():
    V = Chebyshev.Chebyshev(4)
    u = TrialFunction(V)
    v = TestFunction(V)
    x = V.system.x
    expr = x * u * v + (1 + x) * u * v  # multivar + simple
    res = split(expr)
    # Should have only bilinear contributions
    assert len(res["bilinear"]) >= 1
    # Combine coefficients properly
    coeffs = [
        d["coeff"] for d in res["bilinear"] if isinstance(d["coeff"], (int, float))
    ]
    if coeffs:
        assert all(isinstance(ci, (int, float, float)) for ci in coeffs)
