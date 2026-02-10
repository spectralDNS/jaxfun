import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.integrators import ETDRK4
from jaxfun.operators import Constant


def test_etdrk4_splits_weak_form_and_drops_time_from_rhs() -> None:
    F = Fourier(8, Domain(-1, 1))
    (x,) = F.system.base_scalars()
    t = F.system.base_time()

    v = TestFunction(F, name="v")
    u = TrialFunction(F, name="u", transient=True)

    nu = Constant("nu", 0.022)
    eta = Constant("eta", 1.0)
    eq = v * (u.diff(t) + eta * u * u.diff(x) + nu**2 * u.diff(x, 3))

    integrator = ETDRK4(F, eq, time=(0.0, 1.0), initial=sp.cos(sp.pi * x))

    assert integrator.split.time_order == 1
    assert any(a == t for a in integrator.split.time_terms.atoms())
    assert not any(a == t for a in integrator.split.rhs.atoms())

    u_ind = TrialFunction(F, name="u", transient=False)
    expected_linear = v * nu**2 * u_ind.diff(x, 3)
    expected_nonlinear = v * eta * u_ind * u_ind.diff(x)

    assert sp.simplify(integrator.split.rhs_linear - expected_linear) == 0
    assert sp.simplify(integrator.split.rhs_nonlinear - expected_nonlinear) == 0


def test_etdrk4_accepts_second_order_time_derivative() -> None:
    F = Fourier(8, Domain(-1, 1))
    (x,) = F.system.base_scalars()
    t = F.system.base_time()

    v = TestFunction(F, name="v")
    u = TrialFunction(F, name="u", transient=True)

    eq = v * (u.diff(t, 2) - u.diff(x, 2))

    integrator = ETDRK4(F, eq, time=(0.0, 1.0), initial=sp.sin(sp.pi * x))
    assert integrator.split.time_order == 2
