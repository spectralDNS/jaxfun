import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import TensorProductSpace, TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.operators import Constant
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms


def test_trialfunction_transient_affects_time_diff_only() -> None:
    C = Chebyshev(4)
    t = C.system.base_time()

    u_static = TrialFunction(C, name="u", transient=False)
    u_transient = TrialFunction(C, name="u", transient=True)

    assert u_static.diff(t) == 0
    assert isinstance(u_transient.diff(t), sp.Derivative)

    # Transient flag should not affect the computational-domain representation.
    assert u_static.doit() == u_transient.doit()


def test_trialfunction_splitting_kdv() -> None:
    F = Fourier(8, Domain(-1, 1))

    t = F.system.base_time()
    (x,) = F.system.base_scalars()

    u = TrialFunction(F, name="u", transient=True)
    v = TestFunction(F, name="v")

    eta = Constant("eta", 1.0)
    mu = Constant("mu", 0.022)
    kdv = u.diff(t) + eta * u * u.diff(x) + mu**2 * u.diff(x, 3)

    eq = v * kdv
    lhs, rhs = split_time_derivative_terms(eq, t)

    # This breaks, can't evaluate term
    # assert sp.simplify(lhs) != 0

    assert sp.simplify(lhs - v * u.diff(t)) == 0
    assert sp.simplify(rhs - (v * eta * u * u.diff(x) + v * mu**2 * u.diff(x, 3))) == 0

    linear_rhs, nonlinear_rhs = split_linear_nonlinear_terms(rhs, u)
    assert sp.simplify(linear_rhs - v * mu**2 * u.diff(x, 3)) == 0
    assert sp.simplify(nonlinear_rhs - v * eta * u * u.diff(x)) == 0


def test_trialfunction_splitting_zk() -> None:
    F = Fourier(8, Domain(-1, 1))
    T = TensorProductSpace((F, F))

    u = TrialFunction(T, name="u", transient=True)
    v = TestFunction(T, name="v")

    t = T.system.base_time()
    x, y = T.system.base_scalars()

    u_t = u.diff(t)
    nonlinear_u = u * u.diff(x)
    laplace_u = u.diff(x, 2) + u.diff(y, 2)
    v_laplace_u = sp.expand(v * laplace_u.diff(x))
    zk = u_t + nonlinear_u + laplace_u.diff(x)
    eq = v * zk
    lhs, rhs = split_time_derivative_terms(eq, t)

    assert sp.simplify(lhs - v * u_t) == 0
    exp_rhs = v * nonlinear_u + v_laplace_u
    assert sp.simplify(rhs - exp_rhs) == 0

    linear_rhs, nonlinear_rhs = split_linear_nonlinear_terms(rhs, u)
    assert sp.simplify(linear_rhs - v_laplace_u) == 0
    assert sp.simplify(nonlinear_rhs - v * nonlinear_u) == 0


if __name__ == "__main__":
    test_trialfunction_splitting_kdv()
