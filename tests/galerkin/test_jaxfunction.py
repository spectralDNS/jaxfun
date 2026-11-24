import copy
from typing import cast

import jax.numpy as jnp
import pytest
import sympy as sp
from jax import Array
from sympy.utilities.iterables import iterable

from jaxfun import Domain
from jaxfun.coordinates import R
from jaxfun.galerkin import (
    CartesianProduct,
    Chebyshev,
    ChebyshevU,
    Fourier,
    FunctionSpace,
    Jacobi,
    JAXFunction,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    Ultraspherical,
)
from jaxfun.galerkin.inner import inner, project
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.la import TPMatrix
from jaxfun.operators import Div, Dot, Grad
from jaxfun.utils.common import ulp


@pytest.fixture(params=(None, Domain(-2, 2)), ids=("domain-default", "domain-mapped"))
def domain(request: pytest.FixtureRequest) -> Domain | None:
    return request.param


@pytest.fixture(
    params=(
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Fourier.Fourier,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
    ids=("Legendre", "Chebyshev", "ChebyshevU", "Fourier", "Jacobi", "Ultraspherical"),
)
def space(request: pytest.FixtureRequest) -> type[OrthogonalSpace]:
    return request.param


@pytest.fixture(
    params=(
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
    ids=("Legendre", "Chebyshev", "ChebyshevU", "Jacobi", "Ultraspherical"),
)
def jspace(request: pytest.FixtureRequest) -> type[Jacobi.Jacobi]:
    return request.param


def test_jaxfunction(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    u = TrialFunction(D)
    v = TestFunction(D)
    A = inner(u * v, kind="bilinear")
    uf = JAXFunction(jnp.ones(N), D)
    b0 = A @ uf
    b1 = inner(v * uf, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_jaxfunction_directsum(jspace: type[Jacobi.Jacobi], domain: Domain | None):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain, name="D")
    u = TrialFunction(D)
    v = TestFunction(D)
    A, b = inner(u * v, kind="system")
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    b0 = A @ uf - b
    b1 = inner(v * uf, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_jaxfunction_diff(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    x = D.system.x
    u = TrialFunction(D)
    v = TestFunction(D)
    A = inner(u.diff(x) * v, kind="bilinear")
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    b0 = A @ uf
    b1 = inner(v * uf.diff(x), kind="linear")
    assert jnp.linalg.norm(b0 - b1) < ulp(100)


def test_jaxfunction_nonlin(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A = inner(u * v * uf, kind="bilinear")
    b0 = A @ uf
    b1 = inner(v * uf**2, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(100))


def test_jaxfunction_directsum_nonlin(
    jspace: type[Jacobi.Jacobi], domain: Domain | None
):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain, name="D")
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A, b = inner(u * v * uf, kind="system")
    b0 = A @ uf - b
    b1 = inner(v * uf**2, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < ulp(1000)


def test_jaxfunction_nonlin_diff(space: type[OrthogonalSpace], domain: Domain | None):
    N = 6
    D = space(N, domain=domain)
    x = D.system.x
    u = TrialFunction(D)
    v = TestFunction(D)
    uf = JAXFunction(jnp.ones(D.num_dofs), D)
    A = inner(u.diff(x) * v * uf.diff(x), kind="bilinear")
    b0 = A @ uf
    b1 = inner(v * (uf.diff(x)) ** 2, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1000))


def test_project_jaxfunction_onto_other_family(domain: Domain | None):
    N = 8
    T = Chebyshev.Chebyshev(N, domain=domain)
    U = ChebyshevU.ChebyshevU(N, domain=domain)
    x = T.system.x
    uf = JAXFunction(jnp.arange(1.0, N + 1), T)
    expr = uf.diff(x, 2) + uf
    UT = inner(TestFunction(U) * TrialFunction(T), kind="bilinear")
    UU = inner(TestFunction(U) * TrialFunction(U), kind="bilinear")
    expected = UU.solve(UT @ project(expr, T))
    error = jnp.linalg.norm(project(expr, U) - expected) / jnp.linalg.norm(expected)
    assert error < ulp(100)


def test_jaxfunction_coefficient_from_other_family(domain: Domain | None):
    N = 8
    T = Chebyshev.Chebyshev(N, domain=domain)
    U = ChebyshevU.ChebyshevU(N, domain=domain)
    u = TrialFunction(U)
    v = TestFunction(U)
    c = jnp.arange(1.0, N + 1)
    UT = inner(v * TrialFunction(T), kind="bilinear")
    UU = inner(v * u, kind="bilinear")
    A0 = inner(u * v * JAXFunction(UU.solve(UT @ c), U), kind="bilinear")
    A1 = inner(u * v * JAXFunction(c, T), kind="bilinear")
    A0, A1 = A0.todense(), A1.todense()
    assert jnp.linalg.norm(A1 - A0) / jnp.linalg.norm(A0) < ulp(100)


def test_jaxfunction_2d(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    T = TensorProduct(D, D)
    u = TrialFunction(T)
    v = TestFunction(T)
    A = cast(TPMatrix, inner(u * v, kind="bilinear"))
    uf = JAXFunction(jnp.ones((N, N)), T)
    b0 = A.todense() @ uf.get_array().ravel()
    b1 = A.mats[0] @ uf @ A.mats[1].T
    assert jnp.linalg.norm(b0 - b1.ravel()) < ulp(10)
    z = inner(uf * v, kind="linear")
    assert jnp.linalg.norm(cast(Array, z).ravel() - b0) < ulp(10)


def test_jaxfunction_directsum_2d(jspace: type[Jacobi.Jacobi], domain: Domain | None):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, bcs=bcs, domain=domain, name="D")
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    A, b = inner(u * v, kind="system")
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="uf")
    b0 = A @ w - b
    b1 = inner(v * w, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1))
    C, c = inner(Div(Grad(u)) * v, kind="system")
    b0 = C @ w - c
    b1 = inner(v * Div(Grad(w)), kind="linear")
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1))


def test_jaxfunction_diff_2d(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    A = inner(Div(Grad(u)) * v, kind="bilinear")
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    b0 = A @ w
    b1 = inner(v * Div(Grad(w)), kind="linear")
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(10))


def _chebyshev_to_u(N: int, domain: Domain | None) -> Array:
    """Return the matrix mapping Chebyshev coefficients to ChebyshevU ones."""
    T = Chebyshev.Chebyshev(N, domain=domain)
    U = ChebyshevU.ChebyshevU(N, domain=domain)
    M = inner(TestFunction(U) * TrialFunction(T), kind="bilinear").todense()
    B = inner(TestFunction(U) * TrialFunction(U), kind="bilinear").todense()
    return jnp.linalg.solve(B, M)


def test_project_jaxfunction_onto_other_family_2d(domain: Domain | None):
    N = 6
    T = Chebyshev.Chebyshev(N, domain=domain)
    U = ChebyshevU.ChebyshevU(N, domain=domain)
    TT = TensorProduct(T, T)
    x, y = TT.system.base_scalars()
    uf = JAXFunction(jnp.arange(1.0, N * N + 1).reshape(N, N), TT)
    expr = uf.diff(x, 2) + uf.diff(y)
    expected = _chebyshev_to_u(N, domain) @ project(expr, TT)
    result = project(expr, TensorProduct(U, T))
    assert jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected) < ulp(100)


def test_project_jaxfunction_onto_other_family_3d():
    N = 4
    T = Chebyshev.Chebyshev(N)
    U = ChebyshevU.ChebyshevU(N)
    TTT = TensorProduct(T, T, T)
    x, y, z = TTT.system.base_scalars()
    uf = JAXFunction(jnp.arange(1.0, N**3 + 1).reshape(N, N, N), TTT)
    expr = uf.diff(x) + uf.diff(z)
    C = _chebyshev_to_u(N, None)
    expected = jnp.einsum("ij,jkl->ikl", C, project(expr, TTT))
    result = project(expr, TensorProduct(U, T, T))
    assert jnp.linalg.norm(result - expected) / jnp.linalg.norm(expected) < ulp(100)


def test_jaxfunction_coefficient_from_other_family_2d(domain: Domain | None):
    N = 6
    T = Chebyshev.Chebyshev(N, domain=domain)
    U = ChebyshevU.ChebyshevU(N, domain=domain)
    UT = TensorProduct(U, T)
    u = TrialFunction(UT)
    v = TestFunction(UT)
    c = jnp.arange(1.0, N * N + 1).reshape(N, N)
    cU = _chebyshev_to_u(N, domain) @ c
    A0 = inner(u * v * JAXFunction(cU, UT), kind="bilinear")
    A1 = inner(u * v * JAXFunction(c, TensorProduct(T, T)), kind="bilinear")
    A0, A1 = A0.todense(), A1.todense()
    assert jnp.linalg.norm(A1 - A0) / jnp.linalg.norm(A0) < ulp(100)


def test_jaxfunction_nonlin_2d(space: type[OrthogonalSpace], domain: Domain | None):
    N = 4
    D = space(N, domain=domain)
    T = TensorProduct(D, D, name="T")
    u = TrialFunction(T)
    v = TestFunction(T)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    A = inner(u * v * w, kind="bilinear")
    b0 = A @ w
    b1 = inner(v * w**2, kind="linear")
    assert jnp.linalg.norm(b0 - b1) < jnp.sqrt(ulp(1000))


def test_jaxfunction_nonlingrad_2d(
    jspace: type[OrthogonalSpace], domain: Domain | None
):
    N = 4
    D = jspace(N, domain=domain)
    T = TensorProduct(D, D, name="T")
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    A = inner(w * Dot(Grad(v), Grad(u)), kind="bilinear")
    b0 = A @ w
    b1 = inner(0.5 * Dot(Grad(v), Grad(w**2)), kind="linear")
    assert jnp.linalg.norm(b0 - b1) / jnp.linalg.norm(b0) < jnp.sqrt(ulp(10))


def test_jaxfunction_2d_vector(space: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = space(N, domain=domain)
    T = TensorProduct(D, D)
    V = CartesianProduct(T, T, name="V", rank=1)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = inner(Dot(u, v), kind="bilinear")
    uf = JAXFunction(tuple((jnp.ones((N, N)), jnp.ones((N, N)))), V)
    b0 = A @ uf
    b1 = inner(Dot(uf, v), kind="linear")
    assert all(
        jnp.linalg.norm(b0[i] - b1[i]) < jnp.sqrt(ulp(10)) for i in range(len(b0))
    )


def test_jaxfunction_vector(jspace: type[OrthogonalSpace], domain: Domain | None):
    N = 8
    D = jspace(N, domain=domain)
    T = TensorProduct(D, D)
    V = CartesianProduct(T, T, name="V", rank=1)
    x, y = T.system.base_scalars()
    R = T.system
    ue = x * R.i + y * R.j
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    assert jnp.linalg.norm(uj[0] - T.mesh()[0]) < jnp.sqrt(ulp(10))
    assert jnp.linalg.norm(uj[1] - T.mesh()[1]) < jnp.sqrt(ulp(10))

    ue = (1 - x**2) * (1 - y**2) * (R.i + R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    xj, yj = T.mesh()
    assert jnp.linalg.norm(uj[0] - (1 - xj**2) * (1 - yj**2)) < jnp.sqrt(ulp(10))
    assert jnp.linalg.norm(uj[1] - (1 - xj**2) * (1 - yj**2)) < jnp.sqrt(ulp(10))


def test_jaxfunction_vector_composite(
    jspace: type[OrthogonalSpace], domain: Domain | None
):
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    V = CartesianProduct(T, T, name="V", rank=1)
    x, y = T.system.base_scalars()
    R = T.system

    a = D.domain.upper
    ue = sp.Mul((a**2 - x**2) * (a**2 - y**2), R.i + R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    xj, yj = T.mesh()
    assert jnp.linalg.norm(uj[0] - (a**2 - xj**2) * (a**2 - yj**2)) < jnp.sqrt(ulp(10))
    assert jnp.linalg.norm(uj[1] - (a**2 - xj**2) * (a**2 - yj**2)) < jnp.sqrt(ulp(10))


def test_jaxfunction_vector_directsum(
    jspace: type[OrthogonalSpace], domain: Domain | None
):
    N = 8
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(N, jspace, domain=domain, bcs=bcs)
    T = TensorProduct(D, D)
    V = CartesianProduct(T, T, name="V", rank=1)
    x, y = T.system.base_scalars()
    R = T.system

    a = D.domain.upper
    ue = sp.Mul((a**2 - x**2) * (a**2 - y**2) + 1, R.i + R.j)
    w = JAXFunction(ue, V, name="w")
    uj = w.backward()
    xj, yj = T.mesh()
    assert jnp.linalg.norm(uj[0] - ((a**2 - xj**2) * (a**2 - yj**2) + 1)) < jnp.sqrt(
        ulp(10)
    )
    assert jnp.linalg.norm(uj[1] - ((a**2 - xj**2) * (a**2 - yj**2) + 1)) < jnp.sqrt(
        ulp(10)
    )


def test_unnamed_jaxfunctions_are_distinct():
    D = Chebyshev.Chebyshev(4)
    ua = JAXFunction(jnp.arange(4.0), D)
    ub = JAXFunction(jnp.ones(4), D)
    s = (sp.Add(ua, ub, evaluate=False)).doit()
    assert len(s.args) == 2
    arrays = {tuple(float(a) for a in arg.array) for arg in s.args}
    assert arrays == {(0.0, 1.0, 2.0, 3.0), (1.0, 1.0, 1.0, 1.0)}


def test_equal_names_do_not_alias():
    D = Chebyshev.Chebyshev(4)
    ua = JAXFunction(jnp.arange(4.0), D, name="w")
    ub = JAXFunction(jnp.ones(4), D, name="w")
    assert ua.doit() != ub.doit()
    assert ua.doit() == ua.doit()


def test_jaxfunction_linear_algebra_same_space(space: type[OrthogonalSpace]):
    D = space(4)
    ua = JAXFunction(jnp.arange(4.0), D)
    ub = JAXFunction(jnp.ones(4), D)
    for w, expected in (
        (ua + ub, ua.get_array() + ub.get_array()),
        (ua - ub, ua.get_array() - ub.get_array()),
        (ua.scale(-1), -ua.get_array()),
        (ua.scale(2), 2 * ua.get_array()),
    ):
        assert isinstance(w, JAXFunction)
        assert w.functionspace is D
        assert jnp.allclose(cast(Array, w.array), expected)


def test_jaxfunction_linear_algebra_to_orthogonal():
    N = 6
    family = Chebyshev.Chebyshev
    D = FunctionSpace(N, family, bcs={"left": {"D": 0}, "right": {"D": 0}})
    B = FunctionSpace(N, family, bcs={"left": {"N": 0}, "right": {"N": 0}})
    L = FunctionSpace(N, family, bcs={"left": {"D": 1}, "right": {"D": 2}})
    ud = JAXFunction(jnp.ones(D.dim), D)
    un = JAXFunction(jnp.arange(B.dim, dtype=float), B)
    ul = JAXFunction(jnp.ones(L.dim), L)
    # All spaces share the quadrature points of the orthogonal space
    d, n, l = ud.backward(), un.backward(), ul.backward()
    for w, expected in (
        (ud + un, d + n),
        (un - ud, n - d),
        (ul + ud, l + d),
        (ul.scale(3), 3 * l),
    ):
        assert isinstance(w, JAXFunction)
        assert isinstance(w.functionspace, family)
        assert w.functionspace.N == N
        assert jnp.allclose(w.backward(), expected, atol=ulp(100))


def test_jaxfunction_linear_algebra_tensorproduct():
    D = Legendre.Legendre(4)
    T = TensorProduct(D, D)
    w = JAXFunction(jnp.ones((4, 4)), T)
    w2 = w.scale(3)
    assert (w + w2).functionspace is T
    assert jnp.allclose((w - w2).get_array(), -2 * w.get_array())


def test_jaxfunction_linear_algebra_vector():
    D = Legendre.Legendre(4)
    T = TensorProduct(D, D)
    V = CartesianProduct(T, T, rank=1)
    w = JAXFunction((jnp.ones((4, 4)), jnp.zeros((4, 4))), V)
    s = w + w.scale(2)
    assert s.functionspace is V
    assert jnp.allclose(s.array[0], 3.0) and jnp.allclose(s.array[1], 0.0)


def test_jaxfunction_nonlinear_stays_symbolic():
    D = Chebyshev.Chebyshev(4)
    x = D.system.x
    ua = JAXFunction(jnp.ones(4), D)
    ub = JAXFunction(jnp.ones(4), D)
    assert not isinstance(TestFunction(D) * ua, JAXFunction)
    assert not isinstance(ua * ub, JAXFunction)
    assert not isinstance(ua**2, JAXFunction)
    assert not isinstance(x * ua, JAXFunction)
    assert not isinstance(ua + 1, JAXFunction)
    # Numbers and the same JAXFunction keep SymPy's meaning, which SymPy's own
    # algorithms rely on
    assert not isinstance(2 * ua, JAXFunction)
    assert not isinstance(-ua, JAXFunction)
    assert ua - ua == 0
    assert ua + ua == 2 * ua
    # Different bases have no common orthogonal space
    uc = JAXFunction(jnp.ones(4), Legendre.Legendre(4))
    assert not isinstance(ua + uc, JAXFunction)


def test_jaxfunction_to_orthogonal():
    D = FunctionSpace(6, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 2}})
    u = JAXFunction(jnp.ones(D.dim), D)
    w = u.to_orthogonal()
    assert isinstance(w.functionspace, Legendre.Legendre)
    assert w.functionspace.N == 6
    assert jnp.allclose(w.backward(), u.backward(), atol=ulp(100))


def test_jaxfunction_rejects_time_dependent_bcs():
    R1 = R(1)
    t = R1.base_time()
    D = FunctionSpace(
        6, Legendre.Legendre, system=R1, bcs={"left": {"D": t}, "right": {"D": 0}}
    )
    with pytest.raises(ValueError, match="time-dependent"):
        JAXFunction(jnp.ones(D.dim), D)


def test_jaxfunction_sizes():
    D = FunctionSpace(6, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 2}})
    u = JAXFunction(jnp.ones(D.dim), D)
    assert (u.dim, u.num_dofs, u.dims) == (4, 4, 1)
    T = TensorProduct(Legendre.Legendre(3), Legendre.Legendre(4))
    w = JAXFunction(jnp.ones((3, 4)), T)
    assert (w.dim, w.num_dofs, w.dims) == (12, (3, 4), 2)


def test_jaxfunction_evaluates_through_sympy():
    D = Chebyshev.Chebyshev(4)
    x = D.system.x
    u = JAXFunction(x**2, D, name="u")
    value = float(u(0.5))
    for result in (u.subs(x, 0.5), u.evalf(subs={x: 0.5}), sp.N(u, subs={x: 0.5})):
        assert isinstance(result, sp.Float)
        assert abs(float(result) - value) < ulp(10)
    assert u.n() is u
    T = TensorProduct(D, D)
    xs, ys = T.system.base_scalars()
    w = JAXFunction(xs * ys, T)
    assert abs(float(w.subs({xs: 0.5, ys: -0.5})) + 0.25) < ulp(100)


def test_jaxfunction_rebuilds():
    D = Chebyshev.Chebyshev(4)
    x = D.system.x
    y = sp.Symbol("y")
    u = JAXFunction(x**2, D, name="u")
    assert u.func(*u.args) == u
    assert copy.copy(u) == u
    assert sp.simplify(u) == u
    assert u.xreplace({}) is u
    # Applied at other arguments, like any SymPy function
    v = u.subs(x, y / 2)
    assert isinstance(v, JAXFunction) and v.args[0] == y / 2
    assert abs(float(v.subs(y, 1.0)) - float(u(0.5))) < ulp(10)
    assert isinstance(sp.integrate(u, x), sp.Integral)


def test_scalar_functions_are_not_iterable():
    D = Chebyshev.Chebyshev(4)
    u = JAXFunction(jnp.ones(4), D)
    assert not iterable(u)
    assert not iterable(TrialFunction(D))
    T = TensorProduct(D, D)
    W = CartesianProduct(T, T, rank=1)
    assert iterable(TrialFunction(W))
