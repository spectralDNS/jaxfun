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
from jaxfun.utils.common import lambdify, ulp

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
    if jax.config.jax_enable_x64:
        du = JAXFunction(sp.diff(f, x, 2, y, 1), T)
        df = T.backward_primitive(uf.get_array(), (2, 1))
        error = jnp.linalg.norm(df - du.backward())
        assert error < jnp.sqrt(ulp(100)), error


@pytest.mark.parametrize("domain", [(-1, 1), (0, 2), (-2, 2)])
def test_backward_primitive_tps_3d(domain):
    if jax.config.jax_enable_x64:
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


def test_cached_basis_survives_shard_map() -> None:
    """A cached array must not carry the shard_map mesh out with it.

    `cache_static` evaluates eagerly, so a value first computed inside a
    `shard_map` would otherwise be tagged with that mesh (`axis_types=Manual`)
    and clash with ordinary arrays everywhere it was reused afterwards.
    """
    N = 8
    F = FunctionSpace(N, Fourier.Fourier, name="F")
    D = FunctionSpace(N, Legendre.Legendre, {"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D, name="T")
    x, y = T.system.base_scalars()
    uh = project(sp.sin(x) * (1 - y**2), T)

    # Sharded, so the transform goes through shard_map and populates the caches
    # for the padded point count from inside it.
    pad = (2 * N, 2 * N)
    uj = T.backward(jax.device_put(uh, spectral_sharding), N=pad)

    # Now reuse the same cached quadrature points outside the shard_map, and mix
    # the two. Before the fix this raised "Mesh for all inputs should be equal".
    xj = T.mesh(N=pad, broadcast=True)
    ue = lambdify((x, y), sp.sin(x) * (1 - y**2))(*xj)
    assert jnp.linalg.norm(uj.real - ue) < ulp(100)


def test_backward_batch_refuses_sharded() -> None:
    """`backward_batch` must refuse sharded input rather than fail obscurely.

    `_apply_separable_spmd_shard_map` identifies each axis's role by position
    against a fixed-rank spec, which a batch axis shifts. Without the check the
    vmap would instead die on `c.devices()` being called on a traced array.
    """
    N = 8
    F = FunctionSpace(N, Fourier.Fourier, name="F")
    D = FunctionSpace(N, Legendre.Legendre, {"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D, name="T")
    x, y = T.system.base_scalars()
    uh = jax.device_put(project(sp.sin(x) * (1 - y**2), T), spectral_sharding)

    # Unbatched is unaffected: it still goes down the sharded path.
    assert T.backward(uh).shape == T.mesh(broadcast=False)[0].shape[:1] + (N,)
    with pytest.raises(NotImplementedError, match="sharded coefficients"):
        T.backward_batch(jnp.stack([uh, uh]))


@pytest.mark.parametrize(
    "method", ("backward", "backward_primitive", "forward", "scalar_product")
)
def test_batch_refuses_sharded(method: str) -> None:
    """Every batched transform must refuse sharded input, not fail obscurely."""
    N = 8
    F = FunctionSpace(N, Fourier.Fourier, name="F")
    D = FunctionSpace(N, Legendre.Legendre, {"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D, name="T")
    x, y = T.system.base_scalars()
    uh = jax.device_put(project(sp.sin(x) * (1 - y**2), T), spectral_sharding)
    spectral = method.startswith("backward")
    arg = uh if spectral else jax.device_put(T.backward(uh), physical_sharding)
    kwargs = {"k": (1, 0)} if method == "backward_primitive" else {}
    with pytest.raises(NotImplementedError, match="does not handle sharded"):
        getattr(T, method + "_batch")(jnp.stack([arg, arg]), **kwargs)


def test_direct_sum_batch_needs_single_device() -> None:
    """A DirectSum cannot batch at all while sharding is active -- say so clearly."""
    N = 8
    F = FunctionSpace(N, Fourier.Fourier, name="F")
    Tb = FunctionSpace(N, Legendre.Legendre, {"left": {"D": 1}, "right": {"D": 0}})
    VT = TensorProduct(F, Tb, name="VT")
    x, y = VT.system.base_scalars()
    c = project(sp.sin(x) * (1 - y**2), VT)
    with pytest.raises(NotImplementedError, match="single-device host"):
        VT.backward_batch(jnp.stack([c, c]))
    with pytest.raises(NotImplementedError, match="single-device host"):
        VT.backward_primitive_batch(jnp.stack([c, c]), k=(1, 0))
