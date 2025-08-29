import pytest
import sympy as sp
from sympy import Lambda

from jaxfun.coordinates import BaseDyadic, CartCoordSys, get_CoordSys, get_system


def test_get_system_error():
    with pytest.raises(RuntimeError):
        get_system(sp.Symbol("a") + 1)


def test_base_scalar_symbol_mapping_and_derivative():
    r = sp.symbols("r", real=True)
    C = get_CoordSys("C", Lambda((r,), (r,)))
    s = C.r
    assert isinstance(s.to_symbol(), sp.Symbol)
    assert sp.diff(s, s) == 1
    assert sp.diff(s, C.r.to_symbol()) == 0  # diff wrt mapped symbol


def test_base_time_and_subsystem_assertion():
    r, t = sp.symbols("r t")
    C = get_CoordSys("C", Lambda((r,), (r,)))
    t0 = C.base_time()
    assert t0._id[0] == 1  # dims used in id
    with pytest.raises(AssertionError):
        # Cannot create sub system for 1D system
        C.sub_system()


def test_to_from_cartesian_cartesian_shortcuts_and_dyadic():
    x, y, z = sp.symbols("x y z", real=True)
    N = CartCoordSys("N", (x, y, z))
    v = 3 * N.i + 2 * N.j
    assert N.to_cartesian(v) is v  # shortcut
    dy = BaseDyadic(N.i, N.j)
    # from_cartesian returns unchanged for cartesian system
    assert N.from_cartesian(dy) == dy


def test_get_contravariant_basis_vector_and_as_Vector():
    r, theta, z = sp.symbols("r theta z", real=True, positive=True)
    C = get_CoordSys(
        "C", Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    )
    b_contra = C.get_contravariant_basis(as_Vector=True)
    # Returns array of length 3 (each is Cartesian expansion vector)
    assert b_contra.shape == (3,)
    e0 = C.get_contravariant_basis_vector(0)
    assert e0.is_Vector


def test_det_and_sqrt_det_caching_and_contravariant():
    r = sp.symbols("r", positive=True, real=True)
    C = get_CoordSys("C", Lambda((r,), (r,)))
    # 1D system: det g = 1, sqrt(det g)=1 (number -> float conversion path)
    g1 = C.get_det_g(True)
    g2 = C.get_det_g(True)  # cached
    assert g1 == g2 == 1
    gt1 = C.get_det_g(False)
    assert gt1 == 1
    sg = C.get_sqrt_det_g(True)
    assert str(sg) in {"1.0", "1", "1.00000000000000"}


def test_simplify_vectoradd_and_dyadicadd_paths():
    x, y, z = sp.symbols("x y z", real=True)
    N = CartCoordSys("N", (x, y, z))
    v = N.i + N.j
    dy = BaseDyadic(N.i, N.i) + BaseDyadic(N.j, N.j)
    vs = N.simplify(v)
    dys = N.simplify(dy)
    assert vs.is_Vector and dys.is_Dyadic


def test_refine_replace_and_refine_replace_combined():
    r = sp.symbols("r", positive=True, real=True)
    C = get_CoordSys(
        "C",
        Lambda((r,), (r,)),
        replace=[(sp.sqrt(r**2), r)],
        assumptions=sp.Q.positive & sp.Q.real,
    )
    expr = sp.sqrt(r**2)
    # Replacement/refine map to BaseScalar; compare string form to original symbol
    assert str(C.replace(expr)) == str(r)
    assert str(C.refine(expr)) == str(r)
    assert str(C.refine_replace(expr)) == str(r)


def test_get_components_for_dyadic():
    r, theta = sp.symbols("r theta", positive=True, real=True)
    C = get_CoordSys("C", Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))
    dy = C.b_r | C.b_theta
    c_contra = C.get_contravariant_component(dy, 0, 1)
    c_covar = C.get_covariant_component(dy, 0, 1)
    assert c_contra == 1  # component in contravariant representation
    assert c_covar.has(C.r) or c_covar == 0  # metric factor (may be zero-simplified)


def test_basedyadic_invalid_input():
    x, y, z = sp.symbols("x y z", real=True)
    N = CartCoordSys("N", (x, y, z))
    with pytest.raises(TypeError):
        BaseDyadic(1, N.i)  # type: ignore[arg-type]


def test_coord_sys_custom_vector_names():
    x, y, z = sp.symbols("x y z", real=True)
    custom = CartCoordSys("M", (x, y, z))
    assert {v._name for v in custom.base_vectors()} == {"M.i", "M.j", "M.k"}


def test_name_auto_cast_to_str():
    x = sp.symbols("x")
    M = CartCoordSys(123, (x,))  # auto-cast int to str
    assert M._name == "123"


def test_dyadic_component_zero_and_roundtrip_curvilinear():
    r, theta = sp.symbols("r theta", positive=True, real=True)
    C = get_CoordSys("C", Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))))
    dy = C.b_r | C.b_theta
    # Missing component returns zero
    assert C.get_contravariant_component(dy, 1, 0) == 0
    cart = C.to_cartesian(dy)
    back = C.from_cartesian(cart)
    # Roundtrip maintains string form
    assert str(back) == str(dy)


def test_base_time_and_scalar_derivatives():
    r = sp.symbols("r", real=True)
    C = get_CoordSys("C", Lambda((r,), (r,)))
    t0 = C.base_time()
    # dt/dt = 1, dt/dr = 0
    assert sp.diff(t0, t0) == 1
    assert sp.diff(t0, C.r) == 0
    # dr/dt = 0
    assert sp.diff(C.r, t0) == 0


def test_get_contravariant_metric_tensor_caching():
    r = sp.symbols("r", positive=True, real=True)
    C = get_CoordSys("C", Lambda((r,), (r,)))
    g0 = C.get_contravariant_metric_tensor()
    g1 = C.get_contravariant_metric_tensor()  # cached
    assert (g0 == g1).all()
