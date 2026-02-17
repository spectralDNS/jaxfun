import jax
import pytest
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
    jf = JAXFunction(coeffs, V).doit(linear=True)
    with pytest.raises(AssertionError):
        _ = split_coeff(sp.Integer(2) + jf)
    f = jf.args[0]
    c2_raw = split_coeff(sp.Integer(2) + f)
    assert c2_raw["bilinear"] == 2.0 and c2_raw["linear"]["jaxcoeff"] is f


def test_split_and_add_result():
    V = Chebyshev.Chebyshev(4)
    u = TrialFunction(V)
    v = TestFunction(V)
    x = V.system.x
    expr = x * u * v + (1 + x) * u * v
    res = split(expr)
    # Should have only bilinear contributions
    assert len(res["bilinear"]) >= 1
    assert len(res["linear"]) == 0
