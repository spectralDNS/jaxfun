import pytest
import sympy as sp

from jaxfun.coordinates import (
    BaseDyadic,
    BaseScalar,
    BaseVector,
    CartCoordSys,
    get_CoordSys,
)

x, y, z = sp.symbols("x y z", real=True)


def test_cartesian_construction_defaults():
    N = CartCoordSys("N", (x, y, z))
    assert N.is_cartesian
    assert len(N.base_vectors()) == 3
    assert {v._name for v in N.base_vectors()} == {"N.i", "N.j", "N.k"}
    assert all(isinstance(s, BaseScalar) for s in N.base_scalars())


def test_curvilinear_construction():
    r, theta, zeta = sp.symbols("r theta zeta", real=True, positive=True)
    C = get_CoordSys(
        "C", sp.Lambda((r, theta, zeta), (r * sp.cos(theta), r * sp.sin(theta), zeta))
    )
    assert not C.is_cartesian
    # base vector names derive from variable names
    names = {v._name for v in C.base_vectors()}
    assert any("b_r" in n for n in names)
    assert isinstance(C.r, BaseScalar) and isinstance(C.b_r, BaseVector)


def test_invalid_base_scalar_index():
    r = sp.symbols("r", real=True)
    C = get_CoordSys("C", sp.Lambda((r,), (r,)))
    with pytest.raises(ValueError):
        BaseScalar(5, C)


def test_invalid_base_vector_index():
    r = sp.symbols("r", real=True)
    C = get_CoordSys("C", sp.Lambda((r,), (r,)))
    with pytest.raises(ValueError):
        BaseVector(5, C)


def test_basedyadic_zero_shortcut():
    N = CartCoordSys("N", (x, y, z))
    dy = BaseDyadic(N.i, sp.vector.VectorZero())
    assert dy == sp.vector.DyadicZero()
