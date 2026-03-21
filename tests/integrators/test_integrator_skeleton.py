import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import inner
from jaxfun.integrators import ETDRK4
from jaxfun.operators import Constant


# @pytest.mark.skip
def test_etdrk4_splits_weak_form_and_drops_time_from_rhs() -> None:
    F = Fourier(8, Domain(-1, 1))
    (x,) = F.system.base_scalars()
    t = F.system.base_time()

    v = TestFunction(F, name="v")
    u = TrialFunction(F, name="u", transient=True)

    nu = Constant("nu", 0.022)
    eta = Constant("eta", 1.0)
    eq = v * (u.diff(t) + eta * u * u.diff(x) + nu**2 * u.diff(x, 3))

    _integrator = ETDRK4(F, eq, time=(0.0, 1.0), initial=sp.cos(sp.pi * x))


def test_etdrk4_rejects_second_order_time_derivative() -> None:
    F = Fourier(8, Domain(-1, 1))
    (x,) = F.system.base_scalars()
    t = F.system.base_time()

    v = TestFunction(F, name="v")
    u = TrialFunction(F, name="u", transient=True)

    eq = v * (u.diff(t, 2) - u.diff(x, 2))

    with pytest.raises(ValueError, match="first-order time derivatives"):
        _integrator = ETDRK4(F, eq, time=(0.0, 1.0), initial=sp.sin(sp.pi * x))


def test_prepare_assembles_mass_and_linear_forms_for_first_order() -> None:
    F = Fourier(8, Domain(-1, 1))
    (x,) = F.system.base_scalars()
    t = F.system.base_time()

    v = TestFunction(F, name="v")
    u = TrialFunction(F, name="u", transient=True)

    nu = Constant("nu", 0.022)
    eta = Constant("eta", 1.0)
    eq = v * (u.diff(t) + eta * u * u.diff(x) + nu**2 * u.diff(x, 3))

    integrator = ETDRK4(
        F,
        eq,
        time=(0.0, 1.0),
        initial=sp.cos(sp.pi * x),
        sparse=True,
    )
    assert integrator.mass_operator is not None or integrator.mass_diag is not None

    u_ind = TrialFunction(F, name="u", transient=False)
    expected_mass = inner(v * u_ind, sparse=True)
    expected_mass_dense = expected_mass.todense()
    if integrator.mass_operator is not None:
        actual_mass_dense = integrator.mass_operator.todense()  # ty:ignore[unresolved-attribute]
    else:
        assert integrator.mass_diag is not None
        actual_mass_dense = jnp.diag(integrator.mass_diag)
    assert jnp.allclose(actual_mass_dense, expected_mass_dense)

    expected_linear = inner(-(v * nu**2 * u_ind.diff(x, 3)), sparse=True)
    assert integrator.linear_operator is not None or integrator.linear_diag is not None
    expected_linear_dense = expected_linear.todense()
    if integrator.linear_operator is not None:
        actual_linear_dense = integrator.linear_operator.todense()  # ty:ignore[unresolved-attribute]
    else:
        assert integrator.linear_diag is not None
        actual_linear_dense = jnp.diag(integrator.linear_diag)
    assert jnp.allclose(
        actual_linear_dense,
        expected_linear_dense,
    )

    # Nonlinear is not assembled yet, but should be stored with correct sign.
    jaxf = integrator._nonlinear_jaxfunction
    assert jaxf is not None
    expected_nonlin = -(eta * jaxf * jaxf.diff(x))
    assert sp.simplify(integrator.nonlinear_expr - expected_nonlin) == 0


def test_prepare_assembles_weighted_time_derivative_operator() -> None:
    F = Fourier(8, Domain(-1, 1))
    (x,) = F.system.base_scalars()
    t = F.system.base_time()

    v = TestFunction(F, name="v")
    u = TrialFunction(F, name="u", transient=True)

    rho = 2 + sp.cos(x)
    eq = v * (rho * u.diff(t) - u.diff(x, 2))

    integrator = ETDRK4(
        F,
        eq,
        time=(0.0, 1.0),
        initial=sp.sin(sp.pi * x),
        sparse=True,
    )

    u_ind = TrialFunction(F, name="u", transient=False)
    expected_mass = inner(v * rho * u_ind, sparse=True)
    expected_mass_dense = expected_mass.todense()

    assert integrator.mass_operator is not None
    actual_mass_dense = integrator.mass_operator.todense()  # ty:ignore[unresolved-attribute]
    assert jnp.allclose(actual_mass_dense, expected_mass_dense)
    assert not jnp.allclose(actual_mass_dense, jnp.eye(F.num_dofs))
