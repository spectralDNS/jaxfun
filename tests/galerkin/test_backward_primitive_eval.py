import jax
import jax.numpy as jnp

from jaxfun import Domain
from jaxfun.galerkin import TensorProductSpace
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.utils.common import ulp


def _complex_coeffs(key: jax.Array, n: int) -> jax.Array:
    kr, ki = jax.random.split(key)
    return jax.random.normal(kr, (n,)) + 1j * jax.random.normal(ki, (n,))


def test_fourier_backward_primitive_matches_evaluate_derivative() -> None:
    N = 32
    V = Fourier(N, domain=Domain(-1, 1))
    c = _complex_coeffs(jax.random.PRNGKey(1), N)
    xj = V.mesh()

    for k in (0, 1, 3):
        got = V.backward_primitive(c, k=k)
        expected = V.evaluate_derivative(xj, c, k=k)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 20 * ulp(1.0)


def test_chebyshev_backward_primitive_matches_evaluate_derivative() -> None:
    N = 33
    V = Chebyshev(N, domain=Domain(0, 2))
    c = jax.random.normal(jax.random.PRNGKey(2), (N,))
    xj = V.mesh()

    for k in (0, 1, 2):
        got = V.backward_primitive(c, k=k)
        expected = V.evaluate_derivative(xj, c, k=k)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 70 * ulp(1.0)


def test_directsum_backward_primitive_includes_boundary_lift() -> None:
    N = 24
    V = FunctionSpace(
        N,
        Chebyshev,
        bcs={"left": {"D": 1.0}, "right": {"D": -0.5}},
        domain=Domain(-1, 1),
    )
    c = jax.random.normal(jax.random.PRNGKey(3), (V.num_dofs,))
    xj = V.mesh()

    for k in (0, 1):
        got = V.backward_primitive(c, k=k)
        expected = V.evaluate_derivative(xj, c, k=k)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 70 * ulp(1.0)


def test_tensorproduct_fourier_backward_primitive_matches_derivative() -> None:
    N = 24
    F = Fourier(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F))
    c = _complex_coeffs(jax.random.PRNGKey(4), N * N).reshape((N, N))
    xj = list(V.mesh())

    orders = ((0, 0), (1, 0), (0, 2), (1, 1))
    for k in orders:
        got = V.backward_primitive(c, k=k)
        expected = V.evaluate_derivative(xj, c, k=k)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < 30 * ulp(1.0)


def test_tensorproduct_fourier_backward_primitive_uses_fft_fast_path() -> None:
    N = 24
    F = Fourier(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F))
    c = _complex_coeffs(jax.random.PRNGKey(5), N * N).reshape((N, N))
    jaxpr = jax.make_jaxpr(lambda uhat: V.backward_primitive(uhat, k=(1, 0)))(c).jaxpr

    def count_primitive(jpr, primitive: str) -> int:
        total = 0
        for eqn in jpr.eqns:
            if str(eqn.primitive) == primitive:
                total += 1
            for val in eqn.params.values():
                if hasattr(val, "jaxpr") and hasattr(val, "consts"):
                    total += count_primitive(val.jaxpr, primitive)
                elif isinstance(val, tuple | list):
                    for item in val:
                        if hasattr(item, "jaxpr") and hasattr(item, "consts"):
                            total += count_primitive(item.jaxpr, primitive)
        return total

    assert count_primitive(jaxpr, "fft") >= 2
    assert count_primitive(jaxpr, "dot_general") == 0
