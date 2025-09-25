import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from flax import nnx

from jaxfun.coordinates import CartCoordSys, x, y
from jaxfun.galerkin import FunctionSpace, Legendre
from jaxfun.operators import Grad
from jaxfun.pinns.loss import (
    LSQR,
    Residual,
    eval_flaxfunction,
    expand,
    get_flaxfunction_args,
    get_flaxfunctions,
    get_fn,
    lookup_array,
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
    assert args is not None
    assert len(args) > 0


def test_get_flaxfunction_args_with_simple_expr(base_scalars):
    x_sym = base_scalars[0]
    expr = x_sym + 1
    args = get_flaxfunction_args(expr)
    # Should return None or empty for expressions without flax functions
    assert args is None or len(args) == 0


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


def test_eval_flaxfunction():
    # Test eval_flaxfunction function
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x = u.functionspace.system.base_scalars()[0]
    expr = u.diff(x)  # derivative with respect to first coordinate
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = eval_flaxfunction(expr, x)
    assert result.shape[0] == x.shape[0]


def test_residual_with_flaxfunction():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    xj = jnp.array([[1.0, 2.0]])
    f = u
    residual = Residual(f, xj)
    assert len(residual.eqs) == 1
    assert jnp.allclose(residual.target, 0.0)
    x = u.functionspace.system.base_scalars()[0]
    with pytest.raises(AssertionError):
        residual = Residual(f, xj, target=x)


def test_lsqr_vector_equation():
    # Test LSQR with vector equations
    mlp = MLPSpace([4, 4], dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    grad_u = Grad(u)
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
    # Should create residuals for each component
    assert len(lsqr.residuals) == 2
    lsqr = LSQR((u, pts, 0, jnp.array([1.0, 1.0])))
    assert len(lsqr.residuals) == 2


def test_lsqr_update_arrays():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((u, x))
    # Test update_arrays method
    key = list(lsqr.Js.keys())[0]
    old_value = lsqr.Js[key]
    lsqr.update_arrays(u.module, lsqr.Js)
    # Should update with fresh computation
    assert jnp.array_equal(lsqr.Js[key], old_value)


def test_lsqr_compute_Li():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((u - 1, x))  # residual should be u - 1
    loss_i = lsqr.compute_Li(u.module, 0)
    assert isinstance(loss_i, jnp.ndarray)
    assert loss_i.shape == ()  # scalar loss


def test_lsqr_norm_grad_loss_i():
    V = FunctionSpace(2, Legendre.Legendre, name="V")
    u = FlaxFunction(V, "u", kernel_init=nnx.initializers.ones)
    x = jnp.array([[0.0, 1.0]])
    lsqr = LSQR((u, x, u(x)))
    norm = lsqr.norm_grad_loss_i(u.module, 0)
    assert norm == 0


def test_lsqr_norm_grad_loss():
    V = FunctionSpace(2, Legendre.Legendre, name="V")
    u = FlaxFunction(V, "u", kernel_init=nnx.initializers.ones)
    x1 = jnp.array([[0.2, 0.5]])
    x2 = jnp.array([[0.3, 0.4]])
    lsqr = LSQR((u, x1, u(x1)), (u - 1, x2, u(x2)-1))
    norms = lsqr.norm_grad_loss(u.module)
    assert norms.shape == (2,)  # Two residuals
    assert jnp.all(norms == 0)


def test_lsqr_compute_global_weights():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x1 = jnp.array([[1.0, 2.0]])
    x2 = jnp.array([[3.0, 4.0]])
    lsqr = LSQR((u, x1), (u - 1, x2))
    weights = lsqr.compute_global_weights(u.module)
    assert weights.shape == (2,)
    assert jnp.all(weights > 0)


def test_lsqr_update_global_weights():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((u, x), (u-1, x, -1), alpha=0.5)
    old_weights = lsqr.global_weights.copy()
    lsqr.update_global_weights(u.module)
    # Weights should have been updated
    assert not jnp.array_equal(old_weights, lsqr.global_weights)


def test_lsqr_call_with_weights():
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    x = jnp.array([[1.0, 2.0]])
    weights = jnp.array([2.0])
    lsqr = LSQR((u, x, 0, weights))
    loss = lsqr(u.module)
    assert isinstance(loss, jnp.ndarray)
    assert loss.shape == ()


def test_get_fn_multiplication_without_flaxfunction(base_scalars):
    # Test multiplication where some terms don't contain FlaxFunctions
    x_sym = base_scalars[0]
    f = 2 * x_sym * 3  # Should simplify to 6 * x_sym
    fn = get_fn(f, base_scalars[:1])
    x = jnp.array([[4.0]])
    Js = {}
    result = fn(x, Js)
    expected = 6 * x[:, 0]
    assert jnp.allclose(result, expected)


def test_get_fn_complex_number():
    # Test with complex expression
    f = 1 + 2j
    fn = get_fn(sp.sympify(f), ())
    x = jnp.array([])
    Js = {}
    result = fn(x, Js)
    assert result == 1 + 2j


def test_lookup_array_with_multiple_variables():
    x = jnp.array([[1.0, 2.0, 3.0]])
    key = (id(x), 789, 1)
    mock_array = jnp.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    Js = {key: mock_array}

    class MockVar:
        def __init__(self, id_val):
            self._id = (id_val,)

    variables = (MockVar(0), MockVar(1))
    result = lookup_array(x, Js, mod=789, i=1, k=1, variables=variables)
    expected = mock_array[slice(None), 1, 0, 1]
    assert jnp.allclose(result, expected)


def test_residual_with_array_weights():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(0)
    weights = jnp.array([0.5, 1.5])
    residual = Residual(f, x, weights=weights)
    assert jnp.allclose(residual.weights, weights)


def test_lsqr_with_derivative_terms():
    # Test LSQR with derivative expressions
    mlp = MLPSpace(4, dims=2, rank=0, name="MLP")
    u = FlaxFunction(mlp, "u")
    du = Grad(u, 0)  # First derivative
    x = jnp.array([[1.0, 2.0]])
    lsqr = LSQR((du, x))
    # Should have derivative terms in Js
    derivative_keys = [k for k in lsqr.Js.keys() if k[2] > 0]
    assert len(derivative_keys) > 0


def test_get_flaxfunction_args_none():
    # Test case where get_flaxfunction_args returns None
    expr = sp.sympify("x + 1")
    args = get_flaxfunction_args(expr)
    assert args is None


def test_expand_with_complex_expression(flax_func):
    # Test expand with more complex mixed expressions
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
    Js = {}
    result = residual(Js)
    assert jnp.allclose(result, 5.0)  # sum([]) - (-5) = 5


def test_residual_with_symbolic_expression(flax_func):
    x_sym = flax_func.functionspace.system.base_scalars()[0]
    f = flax_func + 2 * x_sym + 1
    x = jnp.array([[3.0, 4.0]])
    residual = Residual(f, x)
    result = residual.target
    expected = -(2 * x[:, 0] + 1)  # -target since no FlaxFunctions
    assert jnp.allclose(result, expected)


def test_lsqr_initialization_single_equation():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(5)
    lsqr = LSQR((f, x))
    assert len(lsqr.residuals) == 1
    assert len(lsqr.global_weights) == 1


def test_lsqr_initialization_multiple_equations():
    x1 = jnp.array([[1.0], [2.0]])
    x2 = jnp.array([[3.0], [4.0]])
    f1 = sp.Integer(5)
    f2 = sp.Integer(10)
    lsqr = LSQR((f1, x1), (f2, x2))
    assert len(lsqr.residuals) == 2
    assert len(lsqr.global_weights) == 2


def test_lsqr_compute_residuals_simple():
    x = jnp.array([[1.0], [2.0]])
    f = sp.Integer(5)
    lsqr = LSQR((f, x))

    # Create a dummy model
    mlp = MLPSpace(4, dims=1, rank=0, name="dummy")
    model = nnx.Module()

    residuals = lsqr.compute_residuals(model)
    assert residuals.shape == (1,)
    assert residuals[0] > 0  # Should be positive since it's a squared residual


def test_lookup_array():
    x = jnp.array([[1.0, 2.0]])
    # Create mock Js dict
    key = (id(x), 123, 0)
    mock_array = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    Js = {key: mock_array}

    result = lookup_array(x, Js, mod=123, i=0, k=0, variables=())
    expected = mock_array[slice(None), 0]
    assert jnp.allclose(result, expected)


def test_lookup_array_with_variables():
    x = jnp.array([[1.0, 2.0]])
    key = (id(x), 456, 1)
    mock_array = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    Js = {key: mock_array}

    # Mock variables with _id attribute
    class MockVar:
        def __init__(self, id_val):
            self._id = (id_val,)

    variables = (MockVar(1),)
    result = lookup_array(x, Js, mod=456, i=1, k=1, variables=variables)
    expected = mock_array[slice(None), 1, 1]
    assert jnp.allclose(result, expected)


def test_get_fn_complex_multiplication(flax_func):
    base_scalars = flax_func.functionspace.system.base_scalars()
    x_sym = base_scalars[0]
    f = 3 * x_sym * flax_func
    fn = get_fn(f.doit(), base_scalars)
    x = jnp.array([[2.0, 1.0]])
    key = (id(x), id(flax_func.module), 0)
    mock_result = jnp.array([[5.0, 6.0]])
    Js = {key: mock_result}
    result = fn(x, Js)
    # Should be 3 * x[0] * flax_result[0, 0]
    expected = 3 * x[0, 0] * mock_result[0, 0]
    assert jnp.allclose(result, expected)


def test_get_fn_power_of_expression(base_scalars):
    x_sym = base_scalars[0]
    f = (2 * x_sym) ** 3
    fn = get_fn(f, base_scalars[:1])
    x = jnp.array([[2.0], [3.0]])
    Js = {}
    result = fn(x, Js)
    expected = (2 * x[:, 0]) ** 3
    assert jnp.allclose(result, expected)


def test_lsqr_alpha_parameter():
    x = jnp.array([[1.0]])
    f = sp.Integer(1)
    alpha = 0.5
    lsqr = LSQR((f, x), alpha=alpha)
    assert lsqr.alpha == alpha


def test_residual_with_zero_target():
    x = jnp.array([[1.0]])
    f = sp.Integer(5)
    residual = Residual(f, x, target=0)
    Js = {}
    result = residual(Js)
    assert jnp.allclose(result, 5.0)


def test_expand_with_multiple_flax_terms(flax_func):
    mlp2 = MLPSpace(4, dims=2, rank=0, name="MLP2")
    v = FlaxFunction(mlp2, "v").doit()

    expr = flax_func + v + 5
    const_part, flax_parts = expand(expr)
    assert const_part == 5
    assert len(flax_parts) == 2
    f = flax_func * v
    fn = get_fn(f.doit(), base_scalars)
    x = jnp.array([[1.0, 2.0]])
    key1 = (id(x), id(flax_func.module), 0)
    key2 = (id(x), id(v.module), 0)
    Js = {key1: flax_func.module(x), key2: v.module(x)}
    result = fn(x, Js)
    expected = Js[key1][(slice(None), 0)] * Js[key2][(slice(None), 0)]
    assert np.allclose(result, expected)


def test_get_fn_pow_flaxfunction(flax_func):
    # f is a FlaxFunction to a power
    base_scalars = flax_func.functionspace.system.base_scalars()
    f = flax_func**2
    fn = get_fn(f.doit(), base_scalars)
    x = jnp.array([[1.0, 2.0]])
    key = (id(x), id(flax_func.module), 0)
    Js = {key: flax_func.module(x)}
    result = fn(x, Js)
    expected = Js[key][(slice(None), 0)] ** 2
    assert np.allclose(result, expected)


def test_get_fn_no_free_symbols():
    # f is a constant with no free symbols
    f = sp.Integer(7)
    fn = get_fn(f, ())
    x = jnp.zeros((2, 0))
    Js = {}
    result = fn(x, Js)
    assert result == 7


def test_get_fn_symbolic_with_offset(base_scalars):
    # f is a symbolic variable plus a constant
    x_sym = base_scalars[0]
    f = x_sym + 3
    fn = get_fn(f, base_scalars[:1])
    x = jnp.array([[5.0], [7.0]])
    Js = {}
    result = fn(x, Js)
    expected = x[:, 0] + 3
    assert np.allclose(result, expected)
