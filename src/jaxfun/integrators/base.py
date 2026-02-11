from abc import ABC
from typing import cast

import sympy as sp
from flax import nnx

from jaxfun.galerkin import TestFunction, TrialFunction
from jaxfun.galerkin.forms import get_basisfunctions
from jaxfun.galerkin.inner import project1D
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.pinns import FlaxFunction, Residual
from jaxfun.pinns.module import SpectralModule
from jaxfun.typing import Array
from jaxfun.utils import split_linear_nonlinear_terms, split_time_derivative_terms


def remove_test_function(expr: sp.Expr, test: TestFunction) -> sp.Expr:
    "Replace test functions in expr with 1."
    if expr.has(test):
        return expr.subs(test, 1)
    return expr


def replace_trial_with_flaxfunction(
    expr: sp.Expr, trial: TrialFunction, flax_func: FlaxFunction
) -> sp.Expr:
    "Replace trial function in expr with a FlaxFunction module."
    if expr.has(trial):
        return expr.replace(trial, flax_func)  # ty:ignore[invalid-return-type]
    return expr


class BaseIntegrator(ABC, nnx.Module):
    def __init__(
        self,
        V: OrthogonalSpace,
        equation: sp.Expr,
        u0: sp.Expr,
    ):
        t = V.system.base_time()
        lhs, rhs = split_time_derivative_terms(equation, t)

        test, trial = get_basisfunctions(rhs)
        assert isinstance(test, TestFunction), (
            "Currently only supports TestFunction in weak form"
        )
        assert isinstance(trial, TrialFunction), (
            "Currently only supports TrialFunction in weak form"
        )

        linear, nonlinear = split_linear_nonlinear_terms(rhs, trial)
        nonlinear = remove_test_function(nonlinear, test)
        flax_func = FlaxFunction(V, name=f"{trial.name}_flax")
        self.nonlinear_expr = replace_trial_with_flaxfunction(
            -nonlinear, trial, flax_func
        )
        # uh = project(u0, V)
        self.module = cast(SpectralModule, flax_func.module)
        # self.module.kernel.set_value(uh.reshape(1, -1))

        points = V.mesh()[:, None]
        self.residual = Residual(self.nonlinear_expr, points)
        self.functionspace = V

    def nonlinear_rhs(self, uh: Array) -> Array:
        self.module.kernel.set_value(uh.reshape(1, -1))
        return self.functionspace.forward(self.residual.evaluate(self.module))


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from jaxfun import Domain
    from jaxfun.galerkin.Fourier import Fourier
    from jaxfun.operators import Constant

    N = 512
    V = Fourier(N, Domain(-1, 1))
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    u = TrialFunction(V, name="u", transient=True)
    v = TestFunction(V, name="v")
    u0 = sp.cos(sp.pi * x)
    mu = Constant("mu", sp.Rational(11, 500))
    equation = v * (u.diff(t) + u * u.diff(x) + mu**2 * u.diff(x, 3))
    integrator = BaseIntegrator(V, equation, u0)
    uh = project1D(u0, V)
    # print(integrator.nonlinear_rhs(uh))
    U = integrator.nonlinear_rhs(uh)
    plt.plot(V.backward(U).real)

    ue = sp.lambdify(x, -u0 * u0.diff(x), "jax")
    uej = ue(V.mesh())
    uj = V.backward(U)
    print("Error: ", jnp.linalg.norm(uej - uj.real))
    plt.plot(ue(V.mesh()).real)
    plt.show()
