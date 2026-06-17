"""Forward/backward transform tests for TensorProductSpace with 2 local devices.

Exercises the SPMD code paths (sharded forward/backward) that activate when
``jax.device_count() > 1``.  Only Chebyshev, Legendre, and Fourier are
included because these are the most relevant bases for the SPMD use-cases.

All tests are marked ``spmd`` and are **skipped by default**.  Run with
``--num-devices=2`` to enable them::

    pytest tests/galerkin/test_forward_backward_spmd.py --num-devices=2
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    JAXFunction,
    Legendre,
    TensorProduct,
)
from jaxfun.galerkin.inner import project
from jaxfun.sharding import physical_sharding, spectral_sharding
from jaxfun.utils.common import ulp

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)

# ---------------------------------------------------------------------------
# 2-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space0, space1",
    [
        (Chebyshev.Chebyshev, Chebyshev.Chebyshev),
        (Legendre.Legendre, Legendre.Legendre),
        (Fourier.Fourier, Fourier.Fourier),
        (Fourier.Fourier, Chebyshev.Chebyshev),
        (Fourier.Fourier, Legendre.Legendre),
    ],
    ids=["ChexChe", "LegxLeg", "FxF", "FxChe", "FxLeg"],
)
def test_forward_backward_2d_spmd(space0, space1) -> None:
    T = TensorProduct(space0(8), space1(8))
    x, y = T.system.base_scalars()
    ue = project(sp.sin(x) * sp.sin(y), T)
    uj = T.backward(ue)
    uh = T.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize(
    "space0, space1",
    [
        (Chebyshev.Chebyshev, Chebyshev.Chebyshev),
        (Legendre.Legendre, Legendre.Legendre),
        (Fourier.Fourier, Fourier.Fourier),
        (Fourier.Fourier, Chebyshev.Chebyshev),
        (Fourier.Fourier, Legendre.Legendre),
    ],
    ids=["ChexChe", "LegxLeg", "FxF", "FxChe", "FxLeg"],
)
def test_scalar_product_2d_spmd(space0, space1) -> None:
    T = TensorProduct(space0(8), space1(8))
    u = jax.device_put(jnp.ones(T.shape), physical_sharding)
    uh = T.scalar_product(u)
    assert uh.sharding == spectral_sharding


# ---------------------------------------------------------------------------
# 3-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space0, space1, space2",
    [
        (Chebyshev.Chebyshev, Chebyshev.Chebyshev, Chebyshev.Chebyshev),
        (Fourier.Fourier, Chebyshev.Chebyshev, Legendre.Legendre),
        (Fourier.Fourier, Fourier.Fourier, Legendre.Legendre),
    ],
    ids=["ChexChexChe", "FxChexLeg", "FxFxLeg"],
)
def test_forward_backward_3d_spmd(space0, space1, space2) -> None:
    T = TensorProduct(space0(8), space1(8), space2(8))
    x, y, z = T.system.base_scalars()
    ue = project(sp.cos(x) * sp.sin(y) * sp.cos(z), T)
    uj = T.backward(ue)
    uh = T.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize(
    "space0, space1, space2",
    [
        (Chebyshev.Chebyshev, Chebyshev.Chebyshev, Chebyshev.Chebyshev),
        (Fourier.Fourier, Chebyshev.Chebyshev, Legendre.Legendre),
        (Fourier.Fourier, Fourier.Fourier, Legendre.Legendre),
    ],
    ids=["ChexChexChe", "FxChexLeg", "FxFxLeg"],
)
def test_scalar_product_3d_spmd(space0, space1, space2) -> None:
    T = TensorProduct(space0(8), space1(8), space2(8))
    u = jax.device_put(jnp.ones(T.shape), physical_sharding)
    uh = T.scalar_product(u)
    assert uh.sharding == spectral_sharding


@pytest.mark.parametrize("domain", [(-1, 1), (0, 2), (-2, 2)])
def test_backward_primitive_tps_2d(domain):
    N = 16
    D = FunctionSpace(N, Legendre.Legendre, domain=domain)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    f = sp.sin(x) * sp.cos(y)
    uf = JAXFunction(f, T)
    du = JAXFunction(sp.diff(f, x, y), T)
    df = T.backward_primitive(uf.get_array(), (1, 1))
    error = jnp.linalg.norm(df - du.backward())

    assert error < jnp.sqrt(ulp(100))
    if jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        du = JAXFunction(sp.diff(f, x, 2, y, 1), T)
        df = T.backward_primitive(uf.get_array(), (2, 1))
        error = jnp.linalg.norm(df - du.backward())
        assert error < jnp.sqrt(ulp(100)), error


@pytest.mark.parametrize("domain", [(-1, 1), (0, 2), (-2, 2)])
def test_backward_primitive_tps_3d(domain):
    if jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        N = 16
        D = FunctionSpace(N, Legendre.Legendre, domain=domain)
        T = TensorProduct(D, D, D)
        x, y, z = T.system.base_scalars()
        f = sp.sin(x) * sp.cos(y) * sp.sin(z)
        uf = JAXFunction(f, T)
        du = JAXFunction(sp.diff(f, x, y, z), T)
        df = T.backward_primitive(uf.get_array(), (1, 1, 1))
        error = jnp.linalg.norm(df - du.backward())

        assert error < jnp.sqrt(ulp(100))
        du = JAXFunction(sp.diff(f, x, 2, y, 1, z, 1), T)
        df = T.backward_primitive(uf.get_array(), (2, 1, 1))
        error = jnp.linalg.norm(df - du.backward())
        assert error < jnp.sqrt(ulp(100)), error
