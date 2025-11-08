import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from flax import nnx

from jaxfun.coordinates import BaseScalar, CartCoordSys, x, y
from jaxfun.galerkin import FunctionSpace, Legendre
from jaxfun.operators import Grad
from jaxfun.pinns.loss import (
    LSQR,
    Residual,
    evaluate,
    expand,
    get_flaxfunction_args,
    get_flaxfunctions,
    get_fn,
)
from jaxfun.pinns.module import FlaxFunction, MLPSpace


@pytest.fixture
def base_scalars():
    C = CartCoordSys("C", (x, y))
    return C.base_scalars()


@pytest.fixture
def flax_func():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    return u


def test_get_flaxfunction_args_with_flaxfunction(flax_func):
    expr = flax_func + 1
    args = get_flaxfunction_args(expr)
    assert isinstance(args, tuple)
    assert isinstance(args[0], BaseScalar)


def test_get_flaxfunction_args_with_simple_expr(base_scalars):
    x_sym = base_scalars[0]
    expr = x_sym + 1
    args = get_flaxfunction_args(expr)
    assert args is None


def test_get_flaxfunction_args_none():
    # Test case where get_flaxfunction_args returns None
    expr = sp.sympify("x + 1")
    args = get_flaxfunction_args(expr)
    assert args is None


def test_get_flaxfunctions_single(flax_func):
    expr = flax_func + 1
    funcs = get_flaxfunctions(expr)
    assert len(funcs) == 1
    assert flax_func in funcs


def test_get_flaxfunctions_multiple(flax_func):
    mlp2 = MLPSpace(4, dims=2, rank=0, name="MLP2")
    v = FlaxFunction(mlp2, "v").doit()
    expr = flax_func * v
    funcs = get_flaxfunctions(expr)
    assert len(funcs) == 2
    assert flax_func in funcs
    assert v in funcs


def test_get_flaxfunctions_none(base_scalars):
    x_sym = base_scalars[0]
    expr = x_sym + 5
    funcs = get_flaxfunctions(expr)
    assert isinstance(funcs, set)
    assert len(funcs) == 0


def test_expand_with_constants_only(base_scalars):
    x_sym = base_scalars[0]
    expr = 2 * x_sym + 3
    const_part, flax_parts = expand(expr)
    assert const_part == 2 * x_sym + 3
    assert len(flax_parts) == 0


def test_expand_with_flaxfunctions(flax_func):
    x_sym = flax_func.functionspace.system.base_scalars()[0]
    expr = 2 * x_sym + flax_func + 3
    const_part, flax_parts = expand(expr)
    assert const_part == 2 * x_sym + 3
    assert len(flax_parts) == 1
    assert flax_func.doit() in flax_parts


def test_expand_mixed_terms(flax_func):
    x_sym = flax_func.functionspace.system.base_scalars()[0]
    expr = x_sym * flax_func + 2 * x_sym + 5
    const_part, flax_parts = expand(expr)
    assert const_part == 2 * x_sym + 5
    assert len(flax_parts) == 1


def test_residual_initialization_simple():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    f = sp.Integer(5)  # Simple constant expression
    residual = Residual(f, x)
    assert jnp.allclose(residual.target, -5.0)  # target - constant
    assert len(residual.eqs) == 0  # No FlaxFunctions


def test_residual_initialization_with_target():
    x = jnp.array([[1.0, 2.0]])
    f = sp.Integer(3)
    target = jnp.array([2.0])
    residual = Residual(f, x, target=target)
    assert jnp.allclose(residual.target, -1.0)  # 2.0 - 3.0


def test_evaluate(flax_func):
    x = flax_func.functionspace.system.base_scalars()[0]
    expr = flax_func.diff(x)  # derivative with respect to first coordinate
    xj = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = evaluate(expr, xj)
    assert result.shape[0] == xj.shape[0]
    w = x * (1 - x) * flax_func
    expr2 = w.diff(x)
    result2 = evaluate(expr2, xj)
    assert jnp.allclose(
        result2, xj[:, 0] * (1 - xj[:, 0]) * result + (1 - 2 * xj[:, 0]) * flax_func(xj)
    )


def test_residual_with_flaxfunction(flax_func):
    xj = jnp.array([[1.0, 2.0]])
    f = flax_func + 3
    residual = Residual(f, xj)
    assert len(residual.eqs) == 1
    assert jnp.allclose(residual.target, -3.0)
    x = flax_func.functionspace.system.base_scalars()[0]
    with pytest.raises(AssertionError):
        residual = Residual(f, xj, target=x)


def test_lsqr_vector_equation(flax_func):
    grad_u = Grad(flax_func)
    pts = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = LSQR((grad_u, pts))
    # Should create residuals for each component
    assert len(lsqr.residuals) == 2  # 2D vector = 2 components


def test_lsqr_vector_equation_with_target():
    # Test LSQR with vector equations
    mlp = MLPSpace([4, 4], dims=2, rank=1, name="MLP")
    u = FlaxFunction(mlp, "u")
    pts = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = LSQR((u, pts, u(pts), jnp.array([1.0, 1.0])))
    assert jnp.allclose(lsqr.residuals[0].target, u(pts)[:, 0])
    assert jnp.allclose(lsqr.residuals[1].target, u(pts)[:, 1])
    assert len(lsqr.residuals) == 2
    lsqr = LSQR((u, pts, 0, jnp.array([1.0, 1.0])))
    assert len(lsqr.residuals) == 2


def test_lsqr_compute_Li(flax_func):
    x = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((flax_func - 1, x))
    xs, targets = lsqr.args
    loss_i = lsqr.compute_Li(flax_func.module, xs, targets, 0)
    assert jnp.allclose(xs[0], x)
    assert targets[0] == 1.0
    assert isinstance(loss_i, jnp.ndarray)
    assert loss_i.shape == ()  # scalar loss
    assert jnp.allclose((flax_func(xs[0]) - 1.0) ** 2, loss_i)


def test_lsqr_norm_grad_loss_i():
    V = FunctionSpace(2, Legendre.Legendre, name="V")
    u = FlaxFunction(V, "u", kernel_init=nnx.initializers.ones)
    x = jnp.array([[0.0], [1.0]])
    lsqr = LSQR((u, x, u(x)))
    xs, targets = lsqr.args
    norm = lsqr.norm_grad_loss_i(u.module, xs, targets, 0)
    assert norm == 0


def test_lsqr_norm_grad_loss():
    V = FunctionSpace(2, Legendre.Legendre, name="V")
    u = FlaxFunction(V, "u", kernel_init=nnx.initializers.ones)
    x1 = jnp.array([[0.2, 0.5]])
    x2 = jnp.array([[0.3, 0.4]])
    lsqr = LSQR((u, x1, u(x1)), (u - 1, x2, u(x2) - 1))
    norms = lsqr.norm_grad_loss(u.module, *lsqr.args)
    assert norms.shape == (2,)  # Two residuals
    assert jnp.all(norms == 0)


def test_lsqr_compute_global_weights(flax_func):
    x1 = jnp.array([[1.0, 2.0]])
    x2 = jnp.array([[3.0, 4.0]])
    lsqr = LSQR((flax_func, x1), (flax_func - 1, x2))
    weights = lsqr.compute_global_weights(flax_func.module, *lsqr.args)
    assert weights.shape == (2,)
    assert jnp.all(weights > 0)


def test_lsqr_update_global_weights(flax_func):
    x = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((flax_func, x), (flax_func - 1, x, -1))
    xs, targets = lsqr.args
    gw = jnp.ones(len(lsqr.residuals), dtype=float)
    _ = lsqr.loss_with_gw(flax_func.module, gw, xs, targets)  # Initialize Js
    old_weights = jnp.ones(len(lsqr.residuals), dtype=float)
    new_weights = lsqr.update_global_weights(
        flax_func.module, old_weights, 0.5, xs, targets
    )
    assert not jnp.array_equal(old_weights, new_weights)


def test_lsqr_call_with_weights(flax_func):
    x = jnp.array([[1.0, 2.0]])
    weights = jnp.array([2.0])
    lsqr = LSQR((flax_func, x, 0, weights))
    loss = lsqr(flax_func.module)
    assert lsqr.residuals[0].weights[0] == 2.0
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()


def test_get_fn_complex_number():
    # Test with complex expression
    f = 1 + 2j
    fn = get_fn(sp.sympify(f), ())
    x = jnp.array([])
    result = fn(x, None)
    assert result == f


def test_residual_with_array_weights():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(0)
    weights = jnp.array([0.5, 1.5])
    residual = Residual(f, x, weights=weights)
    assert jnp.allclose(residual.weights, weights)


def test_lsqr_with_derivative_terms(flax_func):
    mlp = flax_func.functionspace
    x = mlp.system.x
    y = mlp.system.y
    dudx = flax_func.diff(x, 1)
    dudy = flax_func.diff(y, 1)
    xj = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((dudx, xj), (dudy, xj))
    _ = lsqr(flax_func.module)
    # Should have derivative terms in Js
    key = (id(xj), id(flax_func.module), 1)
    assert jnp.array_equal(lsqr.Js[key][:, 0, 0], evaluate(dudx, xj))
    assert jnp.array_equal(lsqr.Js[key][:, 0, 1], evaluate(dudy, xj))


def test_expand_with_complex_expression(flax_func):
    base_scalars = flax_func.functionspace.system.base_scalars()
    x_sym, y_sym = base_scalars[:2]
    expr = x_sym**2 + y_sym * flax_func + 3 * x_sym + 5
    const_part, flax_parts = expand(expr)
    assert sp.simplify(const_part - (x_sym**2 + 3 * x_sym + 5)) == 0
    assert len(flax_parts) == 1


def test_residual_initialization_with_weights():
    x = jnp.array([[1.0, 2.0]])
    f = sp.Integer(0)
    weights = jnp.array([0.5])
    residual = Residual(f, x, weights=weights)
    assert jnp.allclose(residual.weights, 0.5)


def test_residual_call_no_flaxfunctions():
    x = jnp.array([[1.0, 2.0]])
    f = sp.Integer(5)
    residual = Residual(f, x)
    result = residual(x, 0, None)
    assert residual.target == -5.0
    assert jnp.allclose(result, 0.0)


def test_residual_with_symbolic_expression(flax_func):
    x_sym, y_sym = flax_func.functionspace.system.base_scalars()
    f = flax_func + 2 * (x_sym + y_sym) + 1
    x = jnp.array([[3.0, 4.0]])
    residual = Residual(f, x)
    result = residual.target
    expected = -(2 * (x[:, 0] + x[:, 1]) + 1)
    assert jnp.allclose(result, expected)


def test_lsqr_initialization_single_equation():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(5)
    lsqr = LSQR((f, x))
    assert len(lsqr.residuals) == 1
    assert lsqr.residuals[0].target == -5.0


def test_lsqr_initialization_multiple_equations():
    x1 = jnp.array([[1.0], [2.0]])
    x2 = jnp.array([[3.0], [4.0]])
    f1 = sp.Integer(5)
    f2 = sp.Integer(10)
    lsqr = LSQR((f1, x1), (f2, x2))
    assert len(lsqr.residuals) == 2
    assert lsqr.residuals[0].target == -5.0
    assert lsqr.residuals[1].target == -10.0
    assert id(lsqr.residuals[0].x) == id(x1)
    assert id(lsqr.residuals[1].x) == id(x2)


def test_lsqr_compute_residuals_simple():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(5)
    lsqr = LSQR((f, x))
    model = nnx.Module()
    residuals = lsqr.compute_residuals(model)
    assert residuals.shape == (1,)
    assert residuals[0] == 25.0


def test_get_fn_power_of_expression(base_scalars):
    x_sym = base_scalars[0]
    f = (2 * x_sym) ** 3
    fn = get_fn(f, base_scalars[:1])
    x = jnp.array([[2.0], [3.0]])
    result = fn(x, None)
    expected = (2 * x[:, 0]) ** 3
    assert jnp.allclose(result, expected)


def test_residual_with_zero_target():
    x = jnp.array([[1.0]])
    f = sp.Integer(5)
    residual = Residual(f, x, target=0)
    result = residual(residual.x, residual.target, None)
    assert jnp.allclose(result, 5.0)


def test_get_fn_no_free_symbols():
    f = sp.Integer(7)
    fn = get_fn(f, ())
    x = jnp.zeros((2, 0))
    result = fn(x, None)
    assert result == 7


def test_get_fn_symbolic_with_offset(base_scalars):
    x_sym = base_scalars[0]
    f = x_sym + 3
    fn = get_fn(f, base_scalars[:1])
    x = jnp.array([[5.0], [7.0]])
    result = fn(x, None)
    expected = x[:, 0] + 3
    assert np.allclose(result, expected)


if __name__ == "__main__":
    test_get_flaxfunction_args_with_flaxfunction(flax_func.__wrapped__())
    test_get_flaxfunction_args_with_simple_expr(base_scalars.__wrapped__())
    test_get_flaxfunctions_single(flax_func.__wrapped__())
    test_get_flaxfunctions_multiple(flax_func.__wrapped__())
    test_get_flaxfunctions_none(base_scalars.__wrapped__())
    test_evaluate(flax_func.__wrapped__())
    test_expand_with_constants_only(base_scalars.__wrapped__())
    test_expand_with_flaxfunctions(flax_func.__wrapped__())
    test_expand_mixed_terms(flax_func.__wrapped__())
    test_residual_initialization_simple()
    test_residual_initialization_with_weights()
    test_residual_call_no_flaxfunctions()
    test_residual_with_array_weights()
    test_residual_with_flaxfunction(flax_func.__wrapped__())
    test_residual_with_symbolic_expression(flax_func.__wrapped__())
    test_lsqr_vector_equation(flax_func.__wrapped__())
    test_lsqr_vector_equation_with_target()
    test_lsqr_compute_Li(flax_func.__wrapped__())
    test_lsqr_norm_grad_loss_i()
    test_lsqr_norm_grad_loss()
    test_lsqr_compute_global_weights(flax_func.__wrapped__())
    test_lsqr_update_global_weights(flax_func.__wrapped__())
    test_lsqr_call_with_weights(flax_func.__wrapped__())
    test_lsqr_compute_residuals_simple()
    test_lsqr_initialization_single_equation()
    test_lsqr_initialization_multiple_equations()
    test_lsqr_with_derivative_terms(flax_func.__wrapped__())
    test_get_fn_complex_number()
    test_get_fn_power_of_expression(base_scalars.__wrapped__())
    test_get_fn_no_free_symbols()
    test_get_fn_symbolic_with_offset(base_scalars.__wrapped__())
    test_residual_with_zero_target()
    print("All tests passed.")
