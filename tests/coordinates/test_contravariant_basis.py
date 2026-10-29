from itertools import product

import pytest
import sympy as sp
from sympy.vector import Vector

from jaxfun.coordinates import (
    BaseDyadic,
    BaseVector,
    ContravariantBaseVector,
    CoordSys,
    R,
    get_CoordSys,
)
from jaxfun.operators import (
    Identity,
    covariant_diff,
    cross,
    curl,
    divergence,
    dot,
    gradient,
    outer,
)


def polar() -> CoordSys:
    r, t = sp.symbols("r theta", real=True, positive=True)
    return get_CoordSys("P", sp.Lambda((r, t), (r * sp.cos(t), r * sp.sin(t))))


def cylinder() -> CoordSys:
    r, t, z = sp.symbols("r theta z", real=True, positive=True)
    return get_CoordSys("C", sp.Lambda((r, t, z), (r * sp.cos(t), r * sp.sin(t), z)))


def curved() -> CoordSys:
    # Non-orthogonal, with non-zero Christoffel symbols
    u, v = sp.symbols("u v", real=True)
    return get_CoordSys("K", sp.Lambda((u, v), (u + v**2, v)))


def curved3() -> CoordSys:
    u, v, w = sp.symbols("u v w", real=True)
    return get_CoordSys("K3", sp.Lambda((u, v, w), (u + v**2, v, w)))


def is_zero(C: CoordSys, t) -> bool:
    """True if the vector or dyadic t vanishes, compared in Cartesian components."""
    return all(C.simplify(c) == 0 for c in C.to_cartesian(t).components.values())


def test_cartesian_has_no_separate_contravariant_basis():
    N = R(3)
    assert N.contravariant_base_vectors() == N.base_vectors()
    assert N.get_contravariant_basis_vector(1) is N.j
    assert gradient(N.x * N.y) == N.y * N.i + N.x * N.j


def test_contravariant_base_vector_symbols():
    C = polar()
    bt = C.contravariant_base_vectors()
    assert bt == (C.bt_r, C.bt_theta)
    assert C.get_contravariant_basis_vector(1) is C.bt_theta
    assert isinstance(C.bt_r, ContravariantBaseVector)
    assert C.bt_r.is_contravariant and not C.b_r.is_contravariant
    assert C.bt_theta.index == 1
    assert C.bt_r != C.b_r and len({C.bt_r, C.b_r}) == 2
    assert str(C.bt_theta) == "P.bt_theta"
    assert sp.latex(C.bt_theta) == r"\mathbf{b}^{\theta}"
    # Mixed sums keep both kinds of components when printed
    assert str(C.b_r + C.bt_r) == "P.b_r + P.bt_r"


@pytest.mark.parametrize("system", [polar, curved])
def test_dot_rules(system):
    C = system()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    g = C.get_covariant_metric_tensor()
    gt = C.get_contravariant_metric_tensor()
    for i, j in product(range(C.dims), repeat=2):
        delta = int(i == j)
        assert dot(bt[i], b[j]) == delta
        assert dot(b[j], bt[i]) == delta
        assert dot(bt[i], bt[j]) == gt[i, j]
        assert dot(b[i], b[j]) == g[i, j]
        # Dyadic legs contract with the same rules
        assert dot(b[i] | bt[j], b[j]) == b[i]
        assert dot(bt[i], b[i] | bt[j]) == bt[j]


def test_to_cartesian():
    C = curved()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    btc = C.get_contravariant_basis(True)
    for i in range(C.dims):
        assert C.to_cartesian(bt[i]) == btc[i]
        for j in range(C.dims):
            d = dot(C.to_cartesian(bt[i]), C.to_cartesian(b[j]))
            assert C.simplify(d) == int(i == j)
    d = C.to_cartesian(b[0] | bt[1])
    assert d == C.to_cartesian(b[0]) | btc[1]


def test_basis_conversions():
    C = curved()
    u, v = C.base_scalars()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    w = u * b[0] + v * bt[1]
    wc = C.to_covariant_basis(w)
    assert not wc.atoms(ContravariantBaseVector)
    assert is_zero(C, wc - w)
    assert is_zero(C, C.to_contravariant_basis(wc) - w)
    assert not C.to_contravariant_basis(wc).atoms(BaseVector) - set(bt)
    A = u * (b[0] | bt[1]) + (bt[0] | bt[0])
    Ac = C.to_covariant_basis(A)
    assert all(not a.is_contravariant for d in Ac.atoms(BaseDyadic) for a in d.args)
    assert is_zero(C, Ac - A)


def test_change_basis_per_dyadic_leg():
    C = curved()
    u, v = C.base_scalars()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    A = u * (b[0] | b[1]) + (bt[1] | bt[0])
    # The natural basis b_i ⊗ b^j
    An = C.change_basis(A, b, bt)
    assert {d.args for d in An.atoms(BaseDyadic)} <= set(product(b, bt))
    assert is_zero(C, An - A)


def test_change_basis_round_trip_through_cartesian():
    C = curved()
    u, v = C.base_scalars()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    w = u * b[0] + v * bt[1]
    wc = C.from_cartesian(C.to_cartesian(w))
    assert wc.atoms(BaseVector) <= set(b)
    assert is_zero(C, wc - w)


def test_change_basis_rejects_another_system():
    with pytest.raises(ValueError, match="not a base vector"):
        curved().to_cartesian(polar().b_r)


def test_components_of_mixed_tensors():
    C = curved()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    gt = C.get_contravariant_metric_tensor()
    w = 2 * b[0] + 3 * bt[1]
    assert C.get_contravariant_component(w, 0) == 2 + 3 * gt[1, 0]
    assert C.get_contravariant_component(w, 1) == 3 * gt[1, 1]
    assert C.get_covariant_component(bt[1], 1) == 1
    A = b[0] | bt[1]
    for k, j in product(range(C.dims), repeat=2):
        assert C.get_contravariant_component(A, k, j) == int(k == 0) * gt[1, j]


@pytest.mark.parametrize("system", [polar, curved])
def test_covariant_diff_matches_cartesian(system):
    C = system()
    q = C.base_scalars()
    b, bt = C.base_vectors(), C.contravariant_base_vectors()
    tensors = (
        b[0],
        bt[1],
        q[0] * b[1] + q[1] ** 2 * bt[0],
        outer(b[0], bt[1]),
        q[1] * outer(bt[0], bt[1]) + outer(b[1], b[0]),
    )
    for t, x in product(tensors, q):
        # Cartesian base vectors are constant, so differentiate the components
        tc = C.to_cartesian(t)
        ref = sum((sp.diff(c, x) * k for k, c in tc.components.items()), tc.zero)
        assert is_zero(C, C.to_cartesian(covariant_diff(t, x)) - ref)


def test_gradient_and_laplacian():
    C = curved()
    u, v = C.base_scalars()
    bt_u, bt_v = C.contravariant_base_vectors()
    N = C._root
    f = u**2 * v + sp.sin(v)
    g = gradient(f)
    assert g == sp.diff(f, u) * bt_u + sp.diff(f, v) * bt_v
    # f = x*y in Cartesian coordinates has gradient y*i + x*j
    gc = C.to_cartesian(gradient((u + v**2) * v))
    assert is_zero(C, gc - (v * N.i + (u + v**2) * N.j))
    # div(grad(f)) = 1/sqrt(g) d/dq^i (sqrt(g) g^ij df/dq^j)
    sg = C.sg
    gt = C.get_contravariant_metric_tensor()
    q = (u, v)
    lap = sum(
        sp.diff(sg * gt[i, j] * sp.diff(f, q[j]), q[i])
        for i, j in product(range(2), repeat=2)
    )
    assert C.simplify(divergence(g) - lap / sg) == 0


def test_identity():
    C = polar()
    r, t = C.base_scalars()
    w = r * C.b_r + t * C.b_theta
    Id = Identity(C).doit()
    assert dot(Id, w) == w
    assert is_zero(C, dot(Id, r * C.bt_theta) - r * C.bt_theta)


def test_cross_and_curl():
    C = cylinder()
    assert cross(C.b_r, C.b_theta) == C.r * C.bt_z
    assert cross(C.bt_r, C.bt_theta) == C.b_z / C.r

    K = curved3()
    N = K._root
    u, v, w = K.base_scalars()
    b_u, b_v, b_w = K.base_vectors()
    bt_u, bt_v, bt_w = K.contravariant_base_vectors()
    basis = (b_u, b_v, b_w, bt_u, bt_v, bt_w)
    for a, b in product(basis, repeat=2):
        ref = cross(K.to_cartesian(a), K.to_cartesian(b))
        assert is_zero(K, K.to_cartesian(cross(a, b)) - ref)
    # b_i x b^i is not zero in a non-orthogonal system
    assert cross(b_u, bt_u) != Vector.zero
    ref = cross(N.i, N.i - 2 * v * N.j)
    assert is_zero(K, K.to_cartesian(cross(b_u, bt_u)) - ref)

    assert curl(gradient(u**2 * v * w)) == Vector.zero
    assert K.simplify(divergence(curl(u * v * b_u + w * bt_v))) == 0
