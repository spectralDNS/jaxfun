import sympy as sp

from jaxfun.coordinates import BaseDyadic, BaseScalar, BaseVector, get_CoordSys
from jaxfun.operators import Div, Grad, dot


def get_polar():
    r, theta = sp.symbols("r,theta", real=True, positive=True)
    return get_CoordSys(
        "P", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
    )


def get_cylindrical():
    r, theta, z = sp.symbols("r_c,theta_c,z_c", real=True, positive=True)
    return get_CoordSys(
        "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    )


def get_polar_spherical():
    r = 1
    theta, phi = sp.symbols("theta_t, phi_t", real=True, positive=True)
    return get_CoordSys(
        "T",
        sp.Lambda(
            (theta, phi),
            (
                r * sp.sin(theta) * sp.cos(phi),
                r * sp.sin(theta) * sp.sin(phi),
                r * sp.cos(theta),
            ),
        ),
        assumptions=sp.Q.positive(theta)
        & sp.Q.positive(phi)
        & sp.Q.positive(sp.sin(theta)),
    )


def get_spherical():
    r, theta, phi = sp.symbols("r_s, theta_s, phi_s", real=True, positive=True)
    return get_CoordSys(
        "S",
        sp.Lambda(
            (r, theta, phi),
            (
                r * sp.sin(theta) * sp.cos(phi),
                r * sp.sin(theta) * sp.sin(phi),
                r * sp.cos(theta),
            ),
        ),
        assumptions=sp.Q.positive(theta)
        & sp.Q.positive(phi)
        & sp.Q.positive(r)
        & sp.Q.positive(sp.sin(theta)),
    )


coords = {
    "polar": get_polar(),
    "cylindrical": get_cylindrical(),
    "sphere": get_polar_spherical(),
}


def test_laplace():
    P = get_polar()
    r, theta = P.base_scalars()
    f = (r * theta) ** 2
    Lf = Div(Grad(f)).doit()
    assert Lf == 4 * theta**2 + 2

    P = get_cylindrical()
    r, theta, z = P.base_scalars()
    f = (r * theta * z) ** 2
    Lf = Div(Grad(f)).doit()
    assert Lf == 4 * theta**2 * z**2 + 2 * z**2 + 2 * r**2 * theta**2

def test_projection():
    P = get_polar()
    r, theta = P.base_scalars()
    rv = P.position_vector(True)
    rp = dot(rv, P.b_r) * P.b_r + dot(rv, P.b_theta) * P.b_theta
    assert isinstance(rp.args[1], BaseVector)
    assert rp.args[1]._latex_form == '\\mathbf{b_{r}}'
    rs = P.simplify(rp)
    assert isinstance(rs.args[1], BaseVector)
    assert rs.args[1]._latex_form == '\\mathbf{b_{r}}'