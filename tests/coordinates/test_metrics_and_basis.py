from typing import cast

import sympy as sp

from jaxfun.coordinates import CartCoordSys, get_CoordSys

r, theta, z = sp.symbols("r theta z", real=True, positive=True)
C = get_CoordSys(
    "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
)


def test_covariant_basis_shape_and_entries():
    b = C.get_covariant_basis()
    assert b.shape == (3, 3)
    # First row corresponds to d(position)/dr -> (cos, sin, 0)
    assert str(b[0, 0]) == str(sp.cos(theta))
    assert str(b[0, 1]) == str(sp.sin(theta))


def test_contravariant_basis_properties():
    bt = C.get_contravariant_basis()
    g = C.get_covariant_metric_tensor()
    gt = C.get_contravariant_metric_tensor()
    # g * gt = I
    I = sp.eye(3)
    assert all(
        sp.simplify(sum(g[i, k] * gt[k, j] for k in range(3)) - I[i, j]) == 0
        for i in range(3)
        for j in range(3)
    )
    # bt rows are gt @ b rows
    b = C.get_covariant_basis()
    check = cast(sp.MatrixBase, C.simplify(gt @ b))
    bt = cast(sp.MatrixBase, bt)
    assert all(
        sp.simplify(bt[i, j] - check[i, j]) == 0 for i in range(3) for j in range(3)
    )


def test_scaling_factors_and_sqrt_det():
    hi = C.get_scaling_factors()
    assert hi.shape == (3,)
    assert sp.simplify(hi[0] - 1) == 0
    # hi[1] should be r (string compare to avoid triggering sympy simplify on CoordSys)
    assert str(hi[1]) == str(r)
    sg = C.get_sqrt_det_g(True)
    assert str(sg) == str(r)


def test_christoffel_nonzero_components():
    ct = C.get_christoffel_second()
    # In cylindrical coordinates:
    # Gamma^r_{theta theta} = -r,
    # Gamma^theta_{r theta} = Gamma^theta_{theta r} = 1/r
    _idx = {C.r: 0, C.theta: 1, C.z: 2}
    assert sp.simplify(ct[0, 1, 1] + C.r) == 0
    assert sp.simplify(ct[1, 0, 1] - 1 / C.r) == 0
    assert sp.simplify(ct[1, 1, 0] - 1 / C.r) == 0


def test_is_orthogonal():
    assert C.is_orthogonal
    N = CartCoordSys(
        "N", (r, z, theta)
    )  # intentionally permuted mapping, still Cartesian
    assert N.is_orthogonal
