import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin import Chebyshev, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction
from jaxfun.galerkin.forms import split, split_coeff


def test_split_coeff_add_mul_jaxf_and_numbers():
    C = Chebyshev.Chebyshev(4)
    # Use 1D space directly (TensorProduct expects dims>1 for sub-systems)
    v = TestFunction(C)
    u = TrialFunction(C)
    coeffs = jnp.arange(C.N)
    jf = JAXFunction(coeffs, C)
    expr = (2 * jf + 3) * u * v + 5 * u * v
    parts = split(expr)
    # Current split groups numeric + JAXFunction terms into bilinear entries only
    assert len(parts["bilinear"]) == 2 and len(parts["linear"]) == 0
    # split_coeff only supports numeric/Jaxf combos without raw JAXFunction
    # symbolic additions
    sc = split_coeff(sp.Integer(3))
    assert "bilinear" in sc and sc["bilinear"] == 3.0


def test_add_result_multivar_merge():
    C = Chebyshev.Chebyshev(4)
    v = TestFunction(C)
    u = TrialFunction(C)
    (x,) = C.system.base_scalars()
    expr = (sp.sin(x) + sp.cos(x)) * u * v + (sp.sin(x) + sp.cos(x)) * u * v
    parts = split(expr)
    # Each trigonometric piece appears separately as bilinear contribution
    assert len(parts["bilinear"]) == 2
    # Their coeff keys contain scalar multiples (no multivar merge for 1D)
    assert all("coeff" in d for d in parts["bilinear"])
