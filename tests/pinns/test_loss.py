from typing import no_type_check

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from flax import nnx

from jaxfun.coordinates import BaseScalar, CartCoordSys, x, y
from jaxfun.galerkin import FunctionSpace, Legendre, TestFunction
from jaxfun.operators import Grad
from jaxfun.pinns.loss import (
    Loss,
    Residual,
    ResidualVPINN,
    evaluate,
    expand,
    get_flaxfunction_args,
    get_flaxfunctions,
    get_fn,
    get_testfunction,
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
    x_sym = flax_func.functionspace.system.x
    expr = 2 * x_sym + flax_func + 3
    const_part, flax_parts = expand(expr)
    assert const_part == 2 * x_sym + 3
    assert len(flax_parts) == 1
    assert flax_func.doit() in flax_parts


def test_expand_mixed_terms(flax_func):
    x_sym = flax_func.functionspace.system.x
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
    x = flax_func.functionspace.system.x
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
    with pytest.raises(TypeError):
        residual = Residual(f, xj, target=x)


def test_lsqr_vector_equation(flax_func):
    grad_u = Grad(flax_func)
    pts = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = Loss((grad_u, pts))
    # Should create residuals for each component
    assert len(lsqr.residuals) == 2  # 2D vector = 2 components


def test_lsqr_vector_equation_with_target():
    # Test Loss with vector equations
    mlp = MLPSpace([4, 4], dims=2, rank=1, name="MLP")
    u = FlaxFunction(mlp, "u")
    pts = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = Loss((u, pts, u(pts), jnp.array([1.0, 1.0])))
    assert jnp.allclose(lsqr.residuals[0].target, u(pts)[:, 0])
    assert jnp.allclose(lsqr.residuals[1].target, u(pts)[:, 1])
    assert len(lsqr.residuals) == 2
    lsqr = Loss((u, pts, 0, jnp.array([1.0, 1.0])))
    assert len(lsqr.residuals) == 2


def test_lsqr_loss_i(flax_func):
    x = jnp.array([[1.0, 2.0]])
    lsqr = Loss((flax_func - 1, x))
    xs, targets = lsqr.args
    loss_i = lsqr.loss_i(flax_func.module, xs, targets, 0)
    assert jnp.allclose(xs[0], x)
    assert targets[0] == 1.0
    assert isinstance(loss_i, jnp.ndarray)
    assert loss_i.shape == ()  # scalar loss
    assert jnp.allclose((flax_func(xs[0]) - 1.0) ** 2, loss_i)


def test_lsqr_norm_grad_loss_i():
    V = FunctionSpace(2, Legendre.Legendre, name="V")
    u = FlaxFunction(V, "u", kernel_init=nnx.initializers.ones)
    x = jnp.array([[0.0], [1.0]])
    lsqr = Loss((u, x, u(x)))
    xs, targets = lsqr.args
    norm = lsqr.norm_grad_loss_i(u.module, xs, targets, 0)
    assert norm == 0


def test_lsqr_norm_grad_loss():
    V = FunctionSpace(2, Legendre.Legendre, name="V")
    u = FlaxFunction(V, "u", kernel_init=nnx.initializers.ones)
    x1 = jnp.array([[0.2, 0.5]])
    x2 = jnp.array([[0.3, 0.4]])
    lsqr = Loss((u, x1, u(x1)), (u - 1, x2, u(x2) - 1))
    norms = lsqr.norm_grad_loss(u.module, *lsqr.args)
    assert norms.shape == (2,)  # Two residuals
    assert jnp.all(norms == 0)


def test_lsqr_compute_global_weights(flax_func):
    x1 = jnp.array([[1.0, 2.0]])
    x2 = jnp.array([[3.0, 4.0]])
    lsqr = Loss((flax_func, x1), (flax_func - 1, x2))
    weights = lsqr.compute_global_weights(flax_func.module, *lsqr.args)
    assert weights.shape == (2,)
    assert jnp.all(weights > 0)


def test_lsqr_update_global_weights(flax_func):
    x = jnp.array([[1.0, 2.0]])
    lsqr = Loss((flax_func, x), (flax_func - 1, x, -1))
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
    lsqr = Loss((flax_func, x, 0, weights))
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
    lsqr = Loss((f, x))
    assert len(lsqr.residuals) == 1
    assert lsqr.residuals[0].target == -5.0


def test_lsqr_initialization_multiple_equations():
    x1 = jnp.array([[1.0], [2.0]])
    x2 = jnp.array([[3.0], [4.0]])
    f1 = sp.Integer(5)
    f2 = sp.Integer(10)
    lsqr = Loss((f1, x1), (f2, x2))
    assert len(lsqr.residuals) == 2
    assert lsqr.residuals[0].target == -5.0
    assert lsqr.residuals[1].target == -10.0
    assert id(lsqr.residuals[0].x) == id(x1)
    assert id(lsqr.residuals[1].x) == id(x2)


def test_lsqr_compute_residuals_simple():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(5)
    lsqr = Loss((f, x))
    model = nnx.Module()
    residuals = lsqr.compute_residuals(model)
    assert residuals.shape == (1,)
    assert residuals[0] == 12.5


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


def test_evaluate_with_2d_input(flax_func):
    x_sym = flax_func.functionspace.system.x
    expr = flax_func.diff(x_sym)
    xj = jnp.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]).T
    result = evaluate(expr, xj)
    assert result.shape[0] == xj.shape[0]


def test_evaluate_raises_without_flaxfunction(base_scalars):
    x_sym = base_scalars[0]
    expr = x_sym + 3
    xj = jnp.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="does not contain any FlaxFunctions"):
        evaluate(expr, xj)


def test_process_input_arrays_invalid_shape():
    x = jnp.array([[1.0], [2.0], [3.0]])
    target = jnp.array([1.0, 2.0])  # Wrong size
    with pytest.raises(ValueError, match="does not match number of collocation points"):
        from jaxfun.pinns.loss import _process_input_arrays

        _process_input_arrays((target,), x)


def test_process_input_arrays_2d_squeeze():
    from jaxfun.pinns.loss import _process_input_arrays

    x = jnp.array([[1.0], [2.0]])
    target = jnp.array([[1.0], [2.0]])
    result = _process_input_arrays((target,), x)
    assert len(result[0].shape) == 1
    assert result[0].shape[0] == 2


def test_process_input_arrays_invalid_dimensions():
    from jaxfun.pinns.loss import _process_input_arrays

    x = jnp.array([[1.0], [2.0]])
    target = jnp.array([[[1.0], [2.0]]])  # 3D array
    with pytest.raises(ValueError):
        _process_input_arrays((target,), x)


def test_process_input_arrays_with_none():
    from jaxfun.pinns.loss import _process_input_arrays

    x = jnp.array([[1.0], [2.0], [3.0]])
    result = _process_input_arrays((None,), x)
    expected = 1.0 / x.shape[0]
    assert jnp.allclose(result[0], expected)


def test_process_input_arrays_with_number():
    from jaxfun.pinns.loss import _process_input_arrays

    x = jnp.array([[1.0], [2.0]])
    result = _process_input_arrays((5.0,), x)
    assert jnp.allclose(result[0], 5.0)


def test_residual_eval_compute_grad(flax_func):
    x = jnp.array([[1.0, 2.0]])
    residual = Residual(flax_func, x)
    result = residual.eval_compute_grad(x, residual.target, flax_func.module, id(x))
    assert result.shape[0] == x.shape[0]


def test_residual_loss_compute_grad(flax_func):
    x = jnp.array([[1.0, 2.0]])
    residual = Residual(flax_func, x)
    result = residual.loss_compute_grad(x, residual.target, flax_func.module, id(x))
    assert isinstance(result, jnp.ndarray)
    assert result.shape == ()


def test_loss_compute_residual_i(flax_func):
    x1 = jnp.array([[1.0, 2.0]])
    x2 = jnp.array([[3.0, 4.0]])
    lsqr = Loss((flax_func, x1), (flax_func - 1, x2))
    result = lsqr.compute_residual_i(flax_func.module, 0)
    assert result.shape[0] == x1.shape[0]


def test_loss_value_and_grad_and_JTJ(flax_func):
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = Loss((flax_func, x))
    gw = jnp.array([1.0])
    loss, grad, JTJ = lsqr.value_and_grad_and_JTJ(flax_func.module, gw, *lsqr.args)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()
    assert isinstance(grad, nnx.State)
    assert isinstance(JTJ, jnp.ndarray)


def test_loss_JTJ(flax_func):
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = Loss((flax_func, x))
    gw = jnp.array([1.0])
    JTJ = lsqr.JTJ(flax_func.module, gw, *lsqr.args)
    assert isinstance(JTJ, jnp.ndarray)
    assert JTJ.ndim == 2


def test_get_fn_with_pow_expression(base_scalars):
    x_sym = base_scalars[0]
    f = x_sym**2
    fn = get_fn(f, base_scalars[:1])
    x = jnp.array([[2.0], [3.0]])
    result = fn(x, None)
    expected = x[:, 0] ** 2
    assert jnp.allclose(result, expected)


def test_get_fn_with_mul_and_flaxfunction(flax_func):
    x_sym = flax_func.functionspace.system.base_scalars()[0]
    f = 2 * x_sym * flax_func
    s = flax_func.functionspace.system.base_scalars()
    fn = get_fn(f.doit(), s)
    x = jnp.array([[1.0, 2.0]])
    result = fn(x, flax_func.module)
    assert result.shape[0] == x.shape[0]


def test_loss_with_vector_equation_and_array_target():
    mlp = MLPSpace([4, 4], dims=2, rank=1, name="MLP")
    u = FlaxFunction(mlp, "u")
    pts = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = Loss((u, pts, target, jnp.array([1.0, 1.0])))
    assert len(lsqr.residuals) == 2


def test_loss_with_scalar_target():
    mlp = MLPSpace([4, 4], dims=2, rank=1, name="MLP")
    u = FlaxFunction(mlp, "u")
    pts = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    lsqr = Loss((u, pts, 5.0))
    assert len(lsqr.residuals) == 2
    for res in lsqr.residuals:
        assert jnp.allclose(res.target, 5.0)


def test_get_testfunction():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    v = TestFunction(V)
    x_sym = V.system.base_scalars()[0]
    expr = v * x_sym
    result = get_testfunction(expr)
    assert result is not None
    assert result.functionspace == V


def test_get_testfunction_none(base_scalars):
    x_sym = base_scalars[0]
    expr = x_sym + 5
    result = get_testfunction(expr)
    assert result is None


def test_residual_vpinn_initialization():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    x_sym = V.system.x
    xj = jnp.linspace(-1, 1, 10)[:, None]
    weights = jnp.ones(10) / 10
    expr = u.diff(x_sym, 2) * v + u * v
    residual = ResidualVPINN(expr.doit(), xj, weights=weights)
    assert residual.V == V
    assert len(residual.eqs) > 0
    assert residual.target.shape[0] == xj.shape[0]


def test_residual_vpinn_raises_without_testfunction(flax_func):
    xj = jnp.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="No TestFunction found"):
        ResidualVPINN(flax_func, xj)


def test_residual_vpinn_loss():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    xj = jnp.linspace(-1, 1, 10)[:, None]
    weights = jnp.ones(10) / 10
    expr = u * v
    residual = ResidualVPINN(expr, xj, weights=weights)
    loss = residual.loss(xj, residual.target, u.module)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()


def test_residual_vpinn_call():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    xj = jnp.linspace(-1, 1, 10)[:, None]
    weights = jnp.ones(10) / 10
    expr = u * v
    residual = ResidualVPINN(expr, xj, weights=weights)
    result = residual(xj, residual.target, u.module)
    assert result.shape[0] == xj.shape[0]
    assert result.shape[1] == V.num_dofs


def test_loss_with_vpinn():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    x_sym = V.system.x
    xj = jnp.linspace(-1, 1, 20)[:, None]
    xb = jnp.array([[-1.0], [1.0]])
    weights = jnp.ones(20) / 20
    expr = u.diff(x_sym, 2) * v
    lsqr = Loss((expr, xj, 0, weights), (u, xb, 0))
    loss = lsqr(u.module)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()


def test_residual_with_no_flaxfunctions_in_equation():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x_sym = V.system.x
    xj = jnp.linspace(-1, 1, 10)[:, None]
    # Create an expression where one term has no FlaxFunction
    expr = u + x_sym  # x_sym has no FlaxFunction
    residual = Residual(expr, xj)
    # The x_sym should be in the target
    assert jnp.allclose(residual.target, -(xj[:, 0]))


def test_evaluate_with_testfunction_raises():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    xj = jnp.linspace(-1, 1, 10)[:, None]
    expr = u * v
    with pytest.raises(AssertionError, match="Expression contains TestFunction"):
        evaluate(expr, xj)


def test_residual_vpinn_with_target_expr():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    x_sym = V.system.x
    xj = jnp.linspace(-1, 1, 10)[:, None]
    weights = jnp.ones(10) / 10
    expr = u * v - x_sym * v
    residual = ResidualVPINN(expr, xj, weights=weights)
    assert residual.target.shape[0] == xj.shape[0]


def test_residual_vpinn_target_dict():
    V = FunctionSpace(5, Legendre.Legendre, name="V")
    mlp = MLPSpace(4, dims=1, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    v = TestFunction(V)
    x_sym = V.system.x
    xj = jnp.linspace(-1, 1, 10)[:, None]
    weights = jnp.ones(10) / 10
    expr = u.diff(x_sym) * v.diff(x_sym) + u * v - v
    residual = ResidualVPINN(expr, xj, weights=weights)
    assert len(residual.target_dict) > 0


@no_type_check
def main():
    test_get_flaxfunction_args_with_flaxfunction(flax_func.__wrapped__())
    test_get_flaxfunction_args_with_simple_expr(base_scalars.__wrapped__())
    test_get_flaxfunctions_single(flax_func.__wrapped__())
    test_get_flaxfunctions_multiple(flax_func.__wrapped__())
    test_get_flaxfunctions_none(base_scalars.__wrapped__())
    test_evaluate(flax_func.__wrapped__())
    test_evaluate_with_2d_input(flax_func.__wrapped__())
    test_evaluate_raises_without_flaxfunction(base_scalars.__wrapped__())
    test_expand_with_constants_only(base_scalars.__wrapped__())
    test_expand_with_flaxfunctions(flax_func.__wrapped__())
    test_expand_mixed_terms(flax_func.__wrapped__())
    test_residual_initialization_simple()
    test_residual_initialization_with_weights()
    test_residual_call_no_flaxfunctions()
    test_residual_with_array_weights()
    test_residual_with_flaxfunction(flax_func.__wrapped__())
    test_residual_with_symbolic_expression(flax_func.__wrapped__())
    test_residual_eval_compute_grad(flax_func.__wrapped__())
    test_residual_loss_compute_grad(flax_func.__wrapped__())
    test_lsqr_vector_equation(flax_func.__wrapped__())
    test_lsqr_vector_equation_with_target()
    test_loss_with_vector_equation_and_array_target()
    test_loss_with_scalar_target()
    test_lsqr_loss_i(flax_func.__wrapped__())
    test_lsqr_norm_grad_loss_i()
    test_lsqr_norm_grad_loss()
    test_lsqr_compute_global_weights(flax_func.__wrapped__())
    test_lsqr_update_global_weights(flax_func.__wrapped__())
    test_lsqr_call_with_weights(flax_func.__wrapped__())
    test_lsqr_compute_residuals_simple()
    test_loss_compute_residual_i(flax_func.__wrapped__())
    test_loss_value_and_grad_and_JTJ(flax_func.__wrapped__())
    test_loss_JTJ(flax_func.__wrapped__())
    test_lsqr_initialization_single_equation()
    test_lsqr_initialization_multiple_equations()
    test_get_fn_complex_number()
    test_get_fn_power_of_expression(base_scalars.__wrapped__())
    test_get_fn_with_pow_expression(base_scalars.__wrapped__())
    test_get_fn_with_mul_and_flaxfunction(flax_func.__wrapped__())
    test_get_fn_no_free_symbols()
    test_get_fn_symbolic_with_offset(base_scalars.__wrapped__())
    test_residual_with_zero_target()
    test_process_input_arrays_invalid_shape()
    test_process_input_arrays_2d_squeeze()
    test_process_input_arrays_invalid_dimensions()
    test_process_input_arrays_with_none()
    test_process_input_arrays_with_number()
    test_get_testfunction()
    test_get_testfunction_none(base_scalars.__wrapped__())
    test_residual_vpinn_initialization()
    test_residual_vpinn_raises_without_testfunction(flax_func.__wrapped__())
    test_residual_vpinn_loss()
    test_residual_vpinn_call()
    test_loss_with_vpinn()
    test_residual_with_no_flaxfunctions_in_equation()
    test_evaluate_with_testfunction_raises()
    test_residual_vpinn_with_target_expr()
    test_residual_vpinn_target_dict()
    print("All tests passed.")


if __name__ == "__main__":
    main()
