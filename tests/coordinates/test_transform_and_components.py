import sympy as sp

from jaxfun import divergence, gradient
from jaxfun.coordinates import get_CoordSys

r, theta, z = sp.symbols("r theta z", real=True, positive=True)
C = get_CoordSys(
    "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
)


def test_position_vector_forms():
    pv_tuple = C.position_vector(False)
    # Sympy Tuple
    from sympy import Tuple as SymTuple

    assert isinstance(pv_tuple, tuple | SymTuple)
    pv_vec = C.position_vector(True)
    # Should be a VectorAdd with three components
    assert pv_vec.is_Vector
    assert len(pv_vec.components) == 3


def test_to_from_cartesian_roundtrip():
    v = C.r * C.b_r + C.theta * C.b_theta
    cart = C.to_cartesian(v)
    back = C.from_cartesian(cart)
    # Expressed back in original system, components same symbols
    comp_orig = {k: sp.simplify(val) for k, val in v.components.items()}
    comp_back = {k: sp.simplify(val) for k, val in back.components.items()}
    assert comp_orig.keys() == comp_back.keys()


def test_contravariant_and_covariant_component_access():
    v = C.r * C.b_r + C.theta * C.b_theta
    # Contravariant component extraction
    c0 = C.get_contravariant_component(v, 0)
    c1 = C.get_contravariant_component(v, 1)
    assert c0 == C.r and c1 == C.theta
    # Covariant components involve metric
    cov0 = C.get_covariant_component(v, 0)
    cov1 = C.get_covariant_component(v, 1)
    assert sp.simplify(cov0 - C.r) == 0
    assert cov1.has(C.r)  # includes r^2 factor


def test_gradient_and_divergence_in_system():
    s = C.r * C.theta
    g = gradient(s)
    d = divergence(g)
    # Gradient known components
    assert any(sp.simplify(val - C.theta) == 0 for k, val in g.components.items())
    assert any(val.has(C.r) for k, val in g.components.items())
    assert d - C.theta / C.r == 0
    assert isinstance(d, sp.Expr)
