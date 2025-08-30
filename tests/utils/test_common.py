from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from jaxfun.coordinates import x as _x, y as _y
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.typing import Array, ArrayLike
from jaxfun.utils import common


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 1e-10])
def test_ulp(x: float) -> None:
    result = common.ulp(x)
    assert np.isclose(result, jnp.nextafter(x, x + 1) - x)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_diff_simple(k: int) -> None:
    def fun(x: Array, p: ArrayLike) -> Array:
        return x**2 + p

    diff_fun = common.diff(fun, k=k)
    x = jnp.array([1.0, 2.0, 3.0])
    p = 1.0
    result = diff_fun(x, p)
    # Analytical derivatives: k=1: 2x, k=2: 2, k=3: 0
    expected = {1: 2 * x, 2: jnp.full_like(x, 2.0), 3: jnp.zeros_like(x)}
    assert jnp.allclose(result, expected[k])


@pytest.mark.parametrize(
    "k, expected_fn",
    [
        (1, lambda x: 3 * x**2),
        (2, lambda x: 6 * x),
    ],
)
def test_diffx_simple(k: int, expected_fn: Callable[[Array], Array]) -> None:
    def fun(x: Array, p: ArrayLike) -> Array:
        return x**3 + p

    diffx_fun = common.diffx(fun, k=k)
    x = jnp.array([1.0, 2.0, 3.0])
    p = 2
    result = diffx_fun(x, p)

    assert jnp.allclose(result, expected_fn(x))


def test_jacn() -> None:
    def fun(x: Array) -> Array:
        return jnp.array([x**2, x**3])

    jac_fun = common.jacn(fun, k=1)
    x = jnp.array([1.0, 2.0])
    result = jac_fun(x)
    # Jacobian of [x^2, x^3] w.r.t x: [[2x, 3x^2]]
    expected = jnp.stack([2 * x, 3 * x**2], axis=1)

    assert jnp.allclose(result, expected)


def test_matmat_dense() -> None:
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[2, 0], [1, 2]])
    result = common.matmat(a, b)
    expected = a @ b
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("tol", [1, 100])
def test_eliminate_near_zeros(tol: float) -> None:
    a = jnp.array([1e-16, 1.0, 0.0, -1e-16])
    result = common.eliminate_near_zeros(a, tol=tol)
    # All values close to zero should be set to zero
    assert jnp.all((result == 0) | (jnp.abs(result) >= 1.0))


def test_fromdense_and_tosparse() -> None:
    a = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    bcoo_fromdense = common.fromdense(a)
    bcoo_tosparse = common.tosparse(a)
    # Check type and values
    assert isinstance(bcoo_fromdense, BCOO)
    assert isinstance(bcoo_tosparse, BCOO)

    assert jnp.allclose(bcoo_fromdense.todense(), a)
    assert jnp.allclose(bcoo_tosparse.todense(), a)


def test_Domain_namedtuple() -> None:
    d = common.Domain(0, 1)
    assert d.lower == 0
    assert d.upper == 1


def test_lambdify_basic() -> None:
    x, y = _x, _y
    expr = x**2 + y**2
    M = 4
    Dx = FunctionSpace(
        M,
        Chebyshev,
        scaling=common.n + 1,
        # bcs=bcsx,
        name="Dx",
        fun_str="psi",
    )
    Dy = FunctionSpace(
        M,
        Chebyshev,
        scaling=common.n + 1,
        # bcs=bcsy,
        name="Dy",
        fun_str="phi",
    )
    T = TensorProduct(Dx, Dy, name="T")
    expr = T.system.expr_psi_to_base_scalar(expr)

    f = common.lambdify((x, y), expr)
    result = f(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(result, jnp.array([2.0, 8.0]))
