import sympy as sp

from jaxfun.coordinates import BaseDyadic, CartCoordSys, get_CoordSys
from jaxfun.operators import (
    Cross,
    Curl,
    Div,
    Dot,
    Grad,
    Outer,
    cross,
    curl,
    divergence,
    dot,
    express,
    gradient,
    outer,
)

# Coordinate systems
x, y, z = sp.symbols("x y z", real=True)
N = CartCoordSys("N", (x, y, z))
r, theta, zz = sp.symbols("r theta zz", real=True, positive=True)
C = get_CoordSys(
    "C", sp.Lambda((r, theta, zz), (r * sp.cos(theta), r * sp.sin(theta), zz))
)


def test_outer_with_add_and_scalar_factor():
    v = (2 * C.b_r) + C.b_theta
    o = outer(v, C.b_zz)
    # Ensure both basis combinations appear
    s = str(o)
    assert s.count("b_r") >= 1 and s.count("b_theta") >= 1 and "b_zz" in s


def test_cross_add_branch_and_vector_mul():
    v = (2 * C.b_r) + C.b_theta
    w = C.b_r + C.b_theta
    c = cross(v, w)
    # Should simplify; at least be a VectorAdd or VectorZero
    assert c.is_Vector
    assert c.components[C.b_zz] == C.r


def test_dot_various_combinations():
    dy = BaseDyadic(C.b_r, C.b_theta)
    # BaseVector·BaseDyadic (non-diagonal metric gives zero vector)
    res1 = dot(C.b_r, dy)
    assert res1.is_Vector
    # Dyadic·Vector
    res2 = dot(dy, C.b_theta)
    assert res2.is_Vector
    # Dyadic·Dyadic
    dy2 = BaseDyadic(C.b_theta, C.b_r)
    res3 = dot(dy, dy2)
    assert res3.is_Dyadic
    # Zero cases
    zero = sp.vector.VectorZero()
    assert dot(zero, C.b_r) == 0 and dot(C.b_r, zero) == 0


def test_gradient_vector_transpose_and_T_toggle():
    v = C.r * C.b_r + C.theta * C.b_theta
    g = gradient(v, transpose=True)
    assert g.is_Dyadic
    g2 = Grad(v, transpose=True)
    assert g2.T._transpose is False


def test_gradient_product_single_system_after_express():
    scalar_cart = N.x + N.y
    scalar_c = express(scalar_cart, C)  # express scalar in cylindrical system
    expr = (C.r * C.b_r) + scalar_c * C.b_r
    g = gradient(expr)
    assert g.is_Vector or g.is_Dyadic


def test_divergence_cross_returns_Div_unevaluated():
    cr = Cross(C.b_r, C.b_theta)
    d = divergence(cr, doit=False)
    assert isinstance(d, Div)
    assert d.doit().is_scalar


def test_curl_scalar_multiple_vector_mul_branch():
    v = (C.r * C.theta) * C.b_r
    c = curl(v)
    assert c.is_Vector


def test_curl_of_cross_returns_Curl():
    cr = Cross(C.b_r, C.b_theta)
    c = Curl(cr)
    assert isinstance(c, Curl)
    assert c.doit().is_Vector


def test_dot_class_and_outer_class():
    d = Dot(N.i + N.j, N.i + N.j)
    assert d.doit() == 2
    o = Outer(N.i, N.j)
    assert o.T.doit() == outer(N.j, N.i)
    assert o.transpose().doit() == outer(N.j, N.i)


def test_identity_and_constant():
    from jaxfun.operators import Constant, Identity

    c0 = Constant("c0", 5)
    assert c0.doit() == 5
    I = Identity(C)
    Id = I.doit()
    for k in (C.b_r, C.b_theta, C.b_zz):
        assert any(isinstance(a, BaseDyadic) and a.args == (k, k) for a in Id.args)


def test_express_between_systems():
    from jaxfun.operators import express

    # Express N.x*N.i (cartesian) in C system; should substitute x,y,z mapping
    expr = N.x + N.y
    out = express(expr, C)
    # Contains C.r and trigonometric functions
    assert out.has(C.r)


def test_global_doit_vector_and_dyadic_add():
    # Ensure patched Expr.doit returns VectorAdd/DyadicAdd
    vadd = (C.b_r + C.b_theta).doit()
    assert vadd.is_Vector
    dadd = ((C.b_r | C.b_r) + (C.b_theta | C.b_theta)).doit()
    assert dadd.is_Dyadic


def test_cross_zero_and_self():
    z = sp.vector.VectorZero()
    assert cross(z, C.b_r) == z
    assert cross(C.b_r, C.b_r) == z


def test_outer_transpose_property():
    o = Outer(C.b_r, C.b_theta)
    assert o.T.T.doit() == outer(C.b_r, C.b_theta)


def test_dot_dyadic_zero_cases():
    # Construct DyadicZero via zero vector inclusion
    dz = BaseDyadic(N.i, sp.vector.VectorZero())
    assert dz == sp.vector.DyadicZero()
    assert dot(dz, N.i) == sp.vector.VectorZero()
    assert dot(N.i, dz) == sp.vector.VectorZero()


def test_divergence_len2_coord_systems():
    # Use position vector expressed in Cartesian but formed from curvilinear scalars
    pv_cart = C.position_vector(True)  # vector in Cartesian basis
    # Convert back to C to ensure path hit; divergence of position vector
    val = divergence(pv_cart)
    assert isinstance(val, sp.Expr)


def test_gradient_mul_split():
    expr = (C.r * C.theta) * C.r
    g = gradient(expr)
    assert g.is_Vector
