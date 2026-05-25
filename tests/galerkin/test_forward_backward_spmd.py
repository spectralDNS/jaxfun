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
    Legendre,
    TensorProduct,
)
from jaxfun.galerkin.inner import project
from jaxfun.utils.common import ulp

pytestmark = pytest.mark.spmd


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
    u = jax.device_put(jnp.ones(T.shape()), T._physical_sharding)
    uh = T.scalar_product(u)
    assert uh.sharding == T._spectral_sharding


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
    u = jax.device_put(jnp.ones(T.shape()), T._physical_sharding)
    uh = T.scalar_product(u)
    assert uh.sharding == T._spectral_sharding
