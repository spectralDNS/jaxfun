import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import TensorProductSpace
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import project, project1D
from jaxfun.utils.common import lambdify, ulp


def _complex_coeffs(key: jax.Array, n: int) -> jax.Array:
    kr, ki = jax.random.split(key)
    return jax.random.normal(kr, (n,)) + 1j * jax.random.normal(ki, (n,))


def test_fourier_backward_primitive_matches_analytical() -> None:
    N = 32
    V = Fourier(N, domain=Domain(-1, 1))
    x = V.system.base_scalars()[0]
    ue = sp.sin(sp.pi * x)
    uh = project1D(ue, V)
    xj = V.mesh()

    for k, deriv in [(0, ue), (1, sp.diff(ue, x))]:
        got = V.backward_primitive(uh, k=k)
        expected = lambdify(x, deriv)(xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(100), f"k={k}: rel={float(rel)}"
    if jax.config.jax_enable_x64:  # ty: ignore[unresolved-attribute]
        got = V.backward_primitive(uh, k=3)
        expected = lambdify(x, sp.diff(ue, x, 3))(xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(10000), f"k=3: rel={float(rel)}"


def test_chebyshev_backward_primitive_matches_analytical() -> None:
    N = 36
    V = Chebyshev(N, domain=Domain(0, 2))
    x = V.system.base_scalars()[0]
    ue = x**4 - 3 * x**2 + x
    uh = project1D(ue, V)
    xj = V.mesh()

    for k, deriv in [(0, ue), (1, sp.diff(ue, x))]:
        got = V.backward_primitive(uh, k=k)
        expected = lambdify(x, deriv)(xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(1000), f"k={k}: rel={float(rel)}"
    if jax.config.jax_enable_x64:  # ty: ignore[unresolved-attribute]
        got = V.backward_primitive(uh, k=2)
        expected = lambdify(x, sp.diff(ue, x, 2))(xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(10000), f"k=2: rel={float(rel)}"


def test_directsum_backward_primitive_includes_boundary_lift() -> None:
    N = 28
    V = FunctionSpace(
        N,
        ChebyshevU,
        bcs={"left": {"D": 1.0}, "right": {"D": -0.5}},
        domain=Domain(-1, 1),
    )
    x = V.system.base_scalars()[0]
    # Quadratic satisfying BCs u(-1)=1.0, u(1)=-0.5;
    # homogeneous part 7(1-x²)/10 is non-trivial.
    ue = -sp.Rational(7, 10) * x**2 - sp.Rational(3, 4) * x + sp.Rational(19, 20)
    uh = project1D(ue, V)
    xj = V.mesh()

    for k, deriv in [(0, ue), (1, sp.diff(ue, x))]:
        got = V.backward_primitive(uh, k=k)
        expected = lambdify(x, deriv)(xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(1000), f"k={k}: rel={float(rel)}"


def test_tensorproduct_fourier_backward_primitive_matches_analytical() -> None:
    N = 24
    F = Fourier(N, domain=Domain(-1, 1))
    V = TensorProductSpace((F, F))
    x, y = V.system.base_scalars()
    ue = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    uh = project(ue, V)
    xj = list(V.mesh())

    for k in [(0, 0), (1, 0), (1, 1)]:
        got = V.backward_primitive(uh, k=k)
        deriv = sp.diff(sp.diff(ue, x, k[0]), y, k[1])
        expected = lambdify((x, y), deriv)(*xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(100), f"k={k}: rel={float(rel)}"
    if jax.config.jax_enable_x64:  # ty: ignore[unresolved-attribute]
        k2 = (0, 2)
        got = V.backward_primitive(uh, k=k2)
        deriv = sp.diff(sp.diff(ue, x, k2[0]), y, k2[1])
        expected = lambdify((x, y), deriv)(*xj)
        rel = jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)
        assert float(rel) < ulp(10000), f"k=(0,2): rel={float(rel)}"


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
