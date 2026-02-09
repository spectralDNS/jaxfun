import sympy as sp

from jaxfun.operators import Constant
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms


def test_kdv():
    eta = Constant("eta", 1.0)
    mu = Constant("mu", 0.022)
    x, t = sp.symbols("x t")
    u = sp.Function("u")(x, t)  # ty:ignore[call-non-callable]
    kdv = u.diff(t) + mu**2 * u.diff(x, 3) + eta * u * u.diff(x)

    lhs, rhs = split_time_derivative_terms(kdv, t)
    assert sp.simplify(lhs - u.diff(t)) == 0
    assert sp.simplify(rhs - (mu**2 * u.diff(x, 3) + eta * u * u.diff(x))) == 0

    rhs_linear, rhs_nonlinear = split_linear_nonlinear_terms(rhs, u)
    assert sp.simplify(rhs_linear - mu**2 * u.diff(x, 3)) == 0
    assert sp.simplify(rhs_nonlinear - eta * u * u.diff(x)) == 0


def test_wave():
    x, t = sp.symbols("x t")
    u = sp.Function("u")(x, t)  # ty:ignore[call-non-callable]
    c = sp.symbols("c")
    wave = u.diff(t, 2) - c**2 * u.diff(x, 2)

    lhs, rhs = split_time_derivative_terms(wave, t)
    assert sp.simplify(lhs - u.diff(t, 2)) == 0
    assert sp.simplify(rhs + c**2 * u.diff(x, 2)) == 0

    rhs_linear, rhs_nonlinear = split_linear_nonlinear_terms(rhs, u)
    assert sp.simplify(rhs_linear + c**2 * u.diff(x, 2)) == 0
    assert sp.simplify(rhs_nonlinear) == 0


def test_cahn_hilliard():
    M, kappa = sp.symbols("M kappa")
    x, t = sp.symbols("x t")
    u = sp.Function("u")(x, t)  # ty:ignore[call-non-callable]

    exp_lhs = u.diff(t)
    exp_rhs_lin = -M * u.diff(x, 4)
    exp_rhs_nonlin = -M * (
        2 * u.diff(x, 2) * u.diff(x, 2)
        + 4 * u.diff(x) * u.diff(x, 3)
        + kappa * (u**3).diff(x, 2)
    )
    cahn_hilliard = exp_lhs + exp_rhs_lin + exp_rhs_nonlin

    lhs, rhs = split_time_derivative_terms(cahn_hilliard, t)
    assert sp.simplify(lhs - exp_lhs) == 0
    assert sp.simplify(rhs - (exp_rhs_lin + exp_rhs_nonlin)) == 0

    rhs_linear, rhs_nonlinear = split_linear_nonlinear_terms(rhs, u)
    assert sp.simplify(rhs_linear - exp_rhs_lin) == 0
    assert sp.simplify(rhs_nonlinear - exp_rhs_nonlin) == 0
