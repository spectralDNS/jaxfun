from typing import cast

import sympy as sp
from sympy.vector.vector import Vector

from jaxfun.coordinates import get_CoordSys

r, theta, z = sp.symbols("r theta z", positive=True, real=True)
C = get_CoordSys(
    "C",
    sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z)),
    assumptions=sp.Q.positive & sp.Q.real,
    replace=[(sp.sin(theta) ** 2 + sp.cos(theta) ** 2, 1)],
)


def test_refine_and_replace():
    expr = sp.sin(theta) ** 2 + sp.cos(theta) ** 2
    simplified = C.replace(expr)
    assert simplified == 1
    refined = C.refine(sp.sqrt(r**2))
    assert refined == C.r  # due to positive assumption


def test_simplify_vector_and_dyadic():
    v = C.r * C.b_r + (C.r * sp.sin(theta)) * C.b_theta
    simp = C.simplify(v)
    assert getattr(simp, "is_Vector", False)
    simp_v = cast(Vector, simp)
    # Should not alter structure, but ensure mapping back to base scalars
    assert all(hasattr(k, "_system") for k in simp_v.components)


def test_sympy_simplify_walks_into_base_scalars() -> None:
    # sp.simplify reaches the system held by a base scalar and calls
    # Basic.replace(query, value) on it, which CoordSys.replace must allow.
    assert sp.simplify(C.r * sp.sin(C.theta) ** 2 + C.r * sp.cos(C.theta) ** 2) == C.r


def test_replace_patterns_act_on_the_metric() -> None:
    mu, nu = sp.symbols("mu nu", real=True, positive=True)
    mapping = sp.Lambda((mu, nu), (sp.cosh(mu) * sp.cos(nu), sp.sinh(mu) * sp.sin(nu)))
    E = get_CoordSys(
        "E",
        mapping,
        replace=[
            (sp.cos(nu) ** 2, 1 - sp.sin(nu) ** 2),
            (sp.cosh(mu) ** 2, 1 + sp.sinh(mu) ** 2),
        ],
    )
    m, n = E.base_scalars()
    assert E.get_covariant_metric_tensor()[0, 0] == sp.sinh(m) ** 2 + sp.sin(n) ** 2
    assert E.sg == sp.sinh(m) ** 2 + sp.sin(n) ** 2
