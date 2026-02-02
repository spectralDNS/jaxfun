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
