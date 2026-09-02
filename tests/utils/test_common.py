from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import dct as scipy_dct, dst as scipy_dst, idct as scipy_idct

from jaxfun.coordinates import CartCoordSys, x, y
from jaxfun.galerkin import FunctionSpace
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.la import DiaMatrix
from jaxfun.typing import Array, ArrayLike
from jaxfun.utils import common
from jaxfun.utils.common import ulp


@pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 1e-10])
def test_ulp(x: float) -> None:
    result = common.ulp(x)
    assert np.isclose(result, jnp.nextafter(x, x + 1) - x)


class _Cached:
    """Minimal stand-in for a space with `cache_static`-decorated methods."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    @property
    def _cache_key(self) -> tuple[int, ...]:
        return (self.n,)

    @common.cache_static
    def values(self, m: int | None = None) -> Array:
        self.calls += 1
        return jnp.arange(self.n if m is None else m, dtype=float)


def test_cache_static_evaluates_once() -> None:
    c = _Cached(4)
    first = c.values()
    assert c.calls == 1
    assert jnp.array_equal(c.values(), first)
    assert c.calls == 1


def test_cache_static_keys_on_arguments() -> None:
    c = _Cached(4)
    assert c.values().shape == (4,)
    assert c.values(6).shape == (6,)
    assert c.calls == 2


def test_cache_static_keys_on_instance_state() -> None:
    # BCGeneric mutates the N of its orthogonal basis after construction, so a
    # cache keyed on the arguments alone would go stale.
    c = _Cached(4)
    assert c.values().shape == (4,)
    c.n = 6
    assert c.values().shape == (6,)


def test_cache_static_body_runs_once_across_traces() -> None:
    # The body must not be re-executed (and so re-staged) on every trace, and no
    # tracer may end up in the cache.
    c = _Cached(4)

    @jax.jit
    def f(a: Array) -> Array:
        return a * c.values()

    f(jnp.ones(4))
    f(jnp.ones((2, 4)))  # a second trace, different shape
    assert c.calls == 1
    assert all(isinstance(v, np.ndarray) for v in c.__dict__["_static_cache"].values())


def test_cache_static_stages_a_constant_not_the_computation() -> None:
    # The point of the cache: the jaxpr holds the finished array, not the loop
    # that built it.
    space = FunctionSpace(8, Legendre, name="D")
    jaxpr = jax.make_jaxpr(space.backward)(jnp.ones(8))
    assert not any(eqn.primitive.name == "scan" for eqn in jaxpr.eqns), jaxpr


def test_clear_static_cache() -> None:
    space = FunctionSpace(8, Legendre, name="D")
    space.vandermonde(None)
    assert space.__dict__.get("_static_cache")
    space.clear_static_cache()
    assert "_static_cache" not in space.__dict__
    # and it repopulates on demand
    assert space.vandermonde(None).shape == (8, 8)


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
    dia_tosparse = common.tosparse(a)
    # Check type and values
    assert isinstance(dia_tosparse, DiaMatrix)
    assert jnp.allclose(dia_tosparse.todense(), a)


def test_Domain_namedtuple() -> None:
    d = common.Domain(0, 1)
    assert d.lower == 0
    assert d.upper == 1


def test_lambdify_basic() -> None:
    expr = x**2 + y**2
    N = CartCoordSys("N", (x, y))
    expr = N.expr_psi_to_base_scalar(expr)
    f = common.lambdify((x, y), expr)
    result = f(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(result, jnp.array([2.0, 8.0]))


def test_dst_type2_vs_scipy() -> None:
    x = jnp.linspace(-1.0, 1.0, 16)
    expected = scipy_dst(np.asarray(x), type=2, norm=None)
    result = common.dst(x, type=2)
    assert jnp.allclose(result, jnp.asarray(expected), rtol=ulp(1000), atol=ulp(1000))


def test_dst_type1_vs_scipy() -> None:
    x = jnp.linspace(0.25, 2.0, 16)
    expected = scipy_dst(np.asarray(x), type=1, norm=None)
    result = common.dst(x, type=1)
    assert jnp.allclose(result, jnp.asarray(expected), rtol=ulp(1000), atol=ulp(1000))


@pytest.mark.parametrize("transform", ["dct", "idct"])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("n", [15, 16, 17, 32])
def test_dct_vs_scipy(transform: str, complex_input: bool, n: int) -> None:
    """Both directions must match scipy for real and complex input alike.

    Complex input is the case these exist for: `jax.scipy.fft` splits it into two
    real transforms, and these do it in one.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    if complex_input:
        x = x + 1j * rng.standard_normal(n)
    ours = getattr(common, transform)(jnp.asarray(x))
    expected = (scipy_dct if transform == "dct" else scipy_idct)(x, type=2, norm=None)
    # Kind, not width: the suite runs in float32 unless --float64 is given.
    assert jnp.iscomplexobj(ours) == np.iscomplexobj(expected)
    assert jnp.allclose(ours, jnp.asarray(expected), rtol=ulp(1000), atol=ulp(1000))


def test_dct_axis_and_padding() -> None:
    """`axis` and a longer `n` must behave as scipy's do."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 16))
    along_0 = common.dct(jnp.asarray(x), axis=0)
    assert jnp.allclose(
        along_0,
        jnp.asarray(scipy_dct(x, type=2, axis=0, norm=None)),
        rtol=ulp(1000),
        atol=ulp(1000),
    )
    padded = common.dct(jnp.asarray(x), n=24)
    assert jnp.allclose(
        padded,
        jnp.asarray(scipy_dct(x, type=2, n=24, norm=None)),
        rtol=ulp(1000),
        atol=ulp(1000),
    )


def test_dct_rejects_other_types() -> None:
    """Only type 2 is implemented, as for `dst`."""
    x = jnp.zeros(8)
    for f in (common.dct, common.idct):
        with pytest.raises(ValueError, match="type"):
            f(x, type=1)


@pytest.mark.parametrize("type", [1, 2])
@pytest.mark.parametrize("complex_input", [False, True])
def test_dst_vs_scipy_complex_axis_and_padding(type: int, complex_input: bool) -> None:
    """`dst` must match scipy for complex input, off-axis and padded alike.

    Complex input goes through one FFT rather than two: the odd extension of
    z = a + ib is linear, so both Hermitian halves ride in a single transform.
    """
    rng = np.random.default_rng(0)
    for shape, kwargs in (
        ((32,), {}),
        ((8, 32), {}),
        ((8, 32), {"axis": 0}),
        ((32,), {"n": 40}),
    ):
        x = rng.standard_normal(shape)
        if complex_input:
            x = x + 1j * rng.standard_normal(shape)
        ours = common.dst(jnp.asarray(x), type=type, **kwargs)
        expected = scipy_dst(x, type=type, norm=None, **kwargs)
        assert jnp.iscomplexobj(ours) == np.iscomplexobj(expected)
        assert jnp.allclose(
            ours, jnp.asarray(expected), rtol=ulp(1000), atol=ulp(1000)
        ), f"{shape} {kwargs}"


def test_dst_rejects_other_types() -> None:
    """Only types 1 and 2 are implemented."""
    with pytest.raises(ValueError, match="type"):
        common.dst(jnp.zeros(8), type=3)


@pytest.mark.parametrize("transform", ["dct", "idct"])
@pytest.mark.parametrize("n", [4, 16, 31, 32, 40])
def test_dct_resizes_like_scipy(transform: str, n: int) -> None:
    """`n` is the length of the transform, not a lower bound on the input.

    scipy crops a longer input rather than transforming all of it, so
    `dct(x, n=k)` must equal `dct(x[:k])`. Getting this wrong is silent for the
    DSTs, whose odd extension would otherwise be built at one length and sliced
    at another.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    f = getattr(common, transform)
    scipy_f = scipy_dct if transform == "dct" else scipy_idct
    expected = scipy_f(x, type=2, n=n, norm=None)
    assert jnp.allclose(
        f(jnp.asarray(x), n=n), jnp.asarray(expected), rtol=ulp(1000), atol=ulp(1000)
    )
    # The crop is the whole of it: no other resizing may be going on.
    if n <= x.shape[0]:
        assert jnp.allclose(
            f(jnp.asarray(x), n=n),
            f(jnp.asarray(x[:n])),
            rtol=ulp(1000),
            atol=ulp(1000),
        )


@pytest.mark.parametrize("type", [1, 2])
@pytest.mark.parametrize("n", [4, 16, 31, 32, 40])
def test_dst_resizes_like_scipy(type: int, n: int) -> None:
    """The same for `dst`, where a shortened `n` used to be silently wrong.

    The odd extension was built from the full input while `_dst_modes` sliced it
    with the shorter length, so the mode ranges addressed the wrong entries.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    expected = scipy_dst(x, type=type, n=n, norm=None)
    assert jnp.allclose(
        common.dst(jnp.asarray(x), type=type, n=n),
        jnp.asarray(expected),
        rtol=ulp(1000),
        atol=ulp(1000),
    )
    if n <= x.shape[0]:
        assert jnp.allclose(
            common.dst(jnp.asarray(x), type=type, n=n),
            common.dst(jnp.asarray(x[:n]), type=type),
            rtol=ulp(1000),
            atol=ulp(1000),
        )
