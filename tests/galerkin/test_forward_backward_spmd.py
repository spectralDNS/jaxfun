"""Forward/backward transform tests for TensorProductSpace with 2 local devices.

Exercises the SPMD code paths (sharded forward/backward) that activate when
``jax.device_count() > 1``.  Only Chebyshev, Legendre, and Fourier are
included because these are the most relevant bases for the SPMD use-cases.

All tests are marked ``spmd`` and are **skipped by default**.  Run with
``--num-devices=2`` to enable them::

    pytest tests/galerkin/test_forward_backward_spmd.py --num-devices=2
"""

import re

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


def _batch_case(N: int = 8, bcs: bool = False) -> tuple:
    """A sharded coefficient array and the space it belongs to.

    Without `bcs` every extent is `N`, so the sharded path applies at any device
    count that divides it. Two Dirichlet conditions make the polynomial axis
    carry `N - 2` coefficients against `N` quadrature points, and `_use_spmd`
    wants both divisible -- which only ever holds for two devices. That case is
    worth transforming anyway; it just may take the local path.
    """
    F = FunctionSpace(N, Fourier.Fourier, name="F")
    kw = {"bcs": {"left": {"D": 0}, "right": {"D": 0}}} if bcs else {}
    D = FunctionSpace(N, Legendre.Legendre, name="D", **kw)
    T = TensorProduct(F, D, name="T")
    x, y = T.system.base_scalars()
    uh = jax.device_put(project(sp.sin(x) * (1 - y**2), T), spectral_sharding)
    return T, uh


@pytest.mark.parametrize("bcs", (False, True), ids=["orthogonal", "composite"])
@pytest.mark.parametrize(
    "method", ("backward", "backward_primitive", "forward", "scalar_product")
)
def test_batch_matches_one_at_a_time_sharded(method: str, bcs: bool) -> None:
    """A batched transform of sharded input must equal the per-field result.

    The batch axis rides along replicated while the space axes keep their
    sharding, so batching changes only how the arithmetic is issued -- exactly
    the guarantee the unsharded batch already makes.
    """
    T, uh = _batch_case(bcs=bcs)
    spectral = method.startswith("backward")
    arg = uh if spectral else jax.device_put(T.backward(uh), physical_sharding)
    kwargs = {"k": (1, 0)} if method == "backward_primitive" else {}

    fields = jnp.stack([arg, 2.0 * arg, -arg])
    batched = getattr(T, method + "_batch")(fields, **kwargs)
    one_at_a_time = jnp.stack(
        [getattr(T, method)(fields[i], **kwargs) for i in range(fields.shape[0])]
    )
    assert batched.shape == one_at_a_time.shape
    assert jnp.linalg.norm(batched - one_at_a_time) < ulp(100)


def _split_axes(x) -> tuple[int, ...]:
    """The axes of `x` that are actually spread over the mesh.

    Read off the spec rather than compared to one: JAX drops trailing `None`s,
    so `P(None, "k")` and `P(None, "k", None)` are the same placement of a
    rank-3 array and only one of them is what comes back.
    """
    return tuple(ax for ax, part in enumerate(x.sharding.spec) if part is not None)


def test_batch_transposes_the_sharding() -> None:
    """The batch axis stays whole; the space axes swap which one is split."""
    T, uh = _batch_case()
    assert T._use_spmd(T._spectral_sharding, uh.shape, T.num_quad_points)
    fields = jnp.stack([uh, uh])

    uj = T.backward_batch(fields)
    assert _split_axes(uj) == (2,)  # physical: last space axis

    back = T.forward_batch(uj)
    assert _split_axes(back) == (1,)  # spectral: first space axis, never the batch
    assert jnp.linalg.norm(back - fields) < ulp(100)


def test_batch_communicates_once_for_the_whole_batch() -> None:
    """One `all_to_all` per batched transform, not one per field, and no gather.

    A gather on the split axis would mean the fields are not distributed at all,
    which is the failure the histogram catches and a correctness check does not.
    """
    T, uh = _batch_case()
    assert T._use_spmd(T._spectral_sharding, uh.shape, T.num_quad_points)
    fields = jnp.stack([uh, uh, uh])
    hlo = jax.jit(T.backward_batch).lower(fields).compile().as_text()
    assert len(re.findall(r"\ball-to-all\(", hlo)) == 1, hlo
    assert "all-gather(" not in hlo


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
