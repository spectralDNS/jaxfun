from typing import cast

import pytest
import sympy as sp
import sympy.vector as sp_vector

from jaxfun.coordinates import BaseDyadic, CartCoordSys, get_CoordSys
from jaxfun.operators import (
    Constant,
    Cross,
    Curl,
    Div,
    Dot,
    Grad,
    Identity,
    Outer,
    cross,
    curl,
    divergence,
    dot,
    eijk,
    gradient,
    outer,
    sign,
)

x, y, z = sp.symbols("x y z", real=True)
N = CartCoordSys("N", (x, y, z))
r, theta, zz = sp.symbols("r theta z", real=True, positive=True)
C = get_CoordSys(
    "C", sp.Lambda((r, theta, zz), (r * sp.cos(theta), r * sp.sin(theta), zz))
)


def components(v):
    return getattr(v, "components", {})


# ---------------- Levi-Civita and sign helpers ---------------- #
@pytest.mark.parametrize(
    "i,j,k,expected",
    [
        (0, 1, 2, 1),
        (1, 2, 0, 1),
        (2, 0, 1, 1),
        (2, 1, 0, -1),
        (1, 0, 2, -1),
        (0, 2, 1, -1),
        (0, 0, 1, 0),
        (1, 1, 1, 0),
    ],
)
def test_eijk(i, j, k, expected):
    assert eijk(i, j, k) == expected


@pytest.mark.parametrize("i,j", [(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)])
def test_sign(i, j):
    assert sign(i, j) == (1 if ((i + 1) % 3 == j) else -1)


# ---------------- Outer product ---------------- #
def test_outer_basic_and_linear():
    v1, v2 = N.i, N.j
    o = outer(v1, v2)
    assert isinstance(o, BaseDyadic | sp_vector.DyadicAdd)
    if isinstance(o, BaseDyadic):
        assert o.args == (v1, v2)
    else:
        assert any(
            isinstance(arg, BaseDyadic) and arg.args == (v1, v2) for arg in o.args
        )

    v = v1 + v2
    o2 = outer(v, v2)
    produced = {str(o2)} if isinstance(o2, BaseDyadic) else {str(t) for t in o2.args}
    ov1v2 = outer(v1, v2)
    ov2v2 = outer(v2, v2)
    expected_terms: list[str] = []
    expected_terms.extend(
        [str(ov1v2)] if isinstance(ov1v2, BaseDyadic) else [str(t) for t in ov1v2.args]
    )
    expected_terms.extend(
        [str(ov2v2)] if isinstance(ov2v2, BaseDyadic) else [str(t) for t in ov2v2.args]
    )
    assert produced == set(expected_terms)


def test_outer_class_transpose_and_doit():
    o = Outer(N.i, N.j)
    assert o.T.doit() == outer(N.j, N.i)
    assert o.doit() == outer(N.i, N.j)


# ---------------- Cross product ---------------- #
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (N.i, N.j, N.k),
        (N.j, N.k, N.i),
        (N.k, N.i, N.j),
    ],
)
def test_cross_cartesian_basis(a, b, expected):
    assert cross(a, b) == expected
    assert cross(b, a) == -expected


def test_cross_same_vectors_zero():
    assert cross(N.i, N.i) == sp_vector.VectorZero()
    assert cross(N.j, N.j) == sp_vector.VectorZero()
    assert cross(N.k, N.k) == sp_vector.VectorZero()


def test_cross_cylindrical():
    res = cross(C.b_r, C.b_theta)
    comp = components(res)
    # Component should be C.r (the first base scalar of coordinate system C)
    assert C.b_z in comp and comp[C.b_z] == C.r


@pytest.mark.parametrize(
    "a1,a2,b1,b2",
    [
        (N.i, N.j, N.k, None),
        (N.i, None, N.j, N.k),
        (N.i, N.j, N.k, N.i),
    ],
)
def test_cross_linearity(a1, a2, b1, b2):
    A = a1 + a2 if a2 is not None else a1
    B = b1 + b2 if b2 is not None else b1
    res = cross(A, B)
    res_comp = components(res)
    expected: dict = {}
    a_terms = A.args if isinstance(A, sp.Add) else (A,)
    b_terms = B.args if isinstance(B, sp.Add) else (B,)
    for aa in a_terms:
        for bb in b_terms:
            term_comp = components(cross(aa, bb))
            for k, v in term_comp.items():
                expected[k] = sp.simplify(expected.get(k, 0) + v)
    assert set(res_comp) == set(expected)
    for k, v in expected.items():
        assert sp.simplify(res_comp[k] - v) == 0


def test_cross_with_scalars():
    expr = (2 * N.i).cross(3 * N.j)
    assert expr == 6 * N.k


def test_cross_zero_cases():
    zero = sp_vector.VectorZero()
    assert cross(N.i, zero) == zero
    assert cross(zero, N.j) == zero


# ---------------- Dot product ---------------- #
def test_dot_cartesian_orthonormal():
    assert dot(N.i, N.i) == 1
    assert dot(N.i, N.j) == 0


def test_dot_cylindrical_metric():
    assert dot(C.b_theta, C.b_theta) == C.r**2
    assert dot(C.b_r, C.b_theta) == 0


def test_dot_base_dyadic_vector():
    dy = BaseDyadic(C.b_r, C.b_theta)
    res = dot(dy, C.b_theta)
    comp = components(res)
    assert C.b_r in comp and comp[C.b_r] == C.r**2


def test_dot_vector_base_dyadic():
    dy = BaseDyadic(C.b_r, C.b_theta)
    res = dot(C.b_theta, dy)
    # For ordering (b_theta)·(b_r⊗b_theta) metric gives g_{theta,r}=0 so
    # zero vector (VectorZero)
    assert getattr(res, "is_Vector", False)
    assert getattr(res, "components", {}) == {}
    res = dot(dy, C.b_theta)
    assert getattr(res, "is_Vector", False)
    assert getattr(res, "components", {}) == {C.b_r: C.r**2}


def test_dot_dyadic_dyadic():
    dy1 = BaseDyadic(C.b_r, C.b_theta)
    dy2 = BaseDyadic(C.b_theta, C.b_r)
    res = dot(dy1, dy2)
    # Expect C.r**2 * (b_r|b_r) inside expression
    assert C.r**2 in list(sp.preorder_traversal(res))
    assert getattr(res, "components", {}) == {(C.b_r | C.b_r): C.r**2}


def test_dot_zero_cases():
    zero = sp_vector.VectorZero()
    assert dot(zero, N.i) == 0
    assert dot(N.i, zero) == 0


def test_dot_of_vectors_unevaluated():
    expr = Dot(N.i + N.j, N.i + N.j)
    assert expr.doit() == 2


# ---------------- Gradient / Divergence / Curl ---------------- #
def test_gradient_scalar_cylindrical():
    s = C.r * C.theta
    g = gradient(s)
    comp = components(g)
    assert sp.simplify(comp[C.b_r] - C.theta) == 0
    assert sp.simplify(comp[C.b_theta] - 1 / C.r) == 0


def test_gradient_vector_cartesian_transpose():
    f = N.x * N.i
    g = Grad(f)
    assert g == g.T.T


def test_divergence_doc_example():
    v = C.r * C.b_r + C.theta * C.b_theta
    assert sp.simplify(divergence(v) - 3) == 0
    assert sp.simplify(divergence(v + C.z * C.b_z) - 4) == 0


def test_divergence_dyadic():
    bb = C.base_dyadics()
    r, theta, z = psi = C.base_scalars()
    T = r * bb[0] + r**2 * bb[2] + theta * r * bb[3] + r * theta * bb[7]
    dT0 = divergence(T)
    # Use definition of divergence of dyadic to compute with alternative method
    dT1 = sp.Add.fromiter(
        Dot(T.diff(psi[i]), C.get_contravariant_basis_vector(i)).doit()
        for i in range(len(psi))
    )
    assert dT0 == dT1


def test_divergence_vector():
    bb = C.base_vectors()
    r, theta, z = psi = C.base_scalars()
    v = r * bb[0] + r**2 * bb[1] + theta * r * bb[2]
    dv0 = divergence(v)
    # Use definition of divergence of vector to compute with alternative method
    dv1 = sp.Add.fromiter(
        Dot(v.diff(psi[i]), C.get_contravariant_basis_vector(i)).doit()
        for i in range(len(psi))
    )
    assert dv0 == dv1


def test_curl_doc_example():
    v = C.b_r + C.b_theta
    res = curl(v)
    comp = components(res)
    assert C.b_z in comp and sp.simplify(comp[C.b_z] - 2) == 0


def test_divergence_of_gradient_unevaluated():
    s = C.r * C.theta
    g = Grad(s)
    d = divergence(g, doit=False)
    assert isinstance(d, Div)
    assert d.doit() == divergence(gradient(s))


def test_curl_of_cross_unevaluated():
    cr = Cross(C.b_r, C.b_theta)
    c = Curl(cr)
    assert isinstance(c, Curl)


# ---------------- Misc helpers / overrides ---------------- #
def test_constant_and_identity():
    c = Constant("c0", 3)
    assert c.doit() == 3
    ident = Identity(N)
    Id = ident.doit()
    for v in (N.i, N.j, N.k):
        assert any(
            isinstance(arg, BaseDyadic) and arg.args == (v, v) for arg in Id.args
        )


def test_vector_diff_override():
    f = N.x * N.i + N.y * N.j
    df_dx = f.diff(N.x)
    assert df_dx == N.i


def test_doit_vector_expression():
    f = N.x * N.y
    g = 4 * Grad(f)
    res = g.doit()
    comp = components(res)
    assert sp.simplify(comp[N.i] - 4 * N.y) == 0
    assert sp.simplify(comp[N.j] - 4 * N.x) == 0


@pytest.mark.parametrize(
    "perm",
    [
        (N.i, N.j, N.k),
        (N.j, N.k, N.i),
        (N.k, N.i, N.j),
    ],
)
def test_eijk_with_cross_identity(
    perm: tuple[sp_vector.BaseVector, sp_vector.BaseVector, sp_vector.BaseVector],
):
    a, b, cvec = perm
    cross_vec = cast(sp_vector.Vector, cross(a, b))
    val = dot(cross_vec, cvec)
    assert sp.simplify(val - 1) == 0
