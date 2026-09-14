"""A source term that varies in time, under SPMD.

`tests/integrators/test_transient_bcs_spmd.py` covers the same hazard for a
moving lifting, and its header explains why the step time has to be carried as a
shape-`(1,)` array. The source is the other thing that time reaches.

The placement risk is its own, though. A steady source is assembled once and
replicated at construction, and stays put; a moving one is rebuilt from `t`
inside the jitted step, so every rebuild is a fresh chance to come back on the
assembly sharding that `_advance`'s closure cannot carry across devices.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/integrators/test_transient_sources_spmd.py --num-devices=2
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import R
from jaxfun.galerkin import Fourier, FunctionSpace, Legendre, TensorProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.integrators import ARS443, IMEXRungeKutta
from jaxfun.operators import Constant, Div, Grad

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)

# The leading axis has to divide across the mesh for assembly to shard anything,
# which is the case these exist for.
N = 4 * jax.device_count()
NU = 0.1


def _diffusion_with_a_moving_source():
    """Heat equation on Fourier x Legendre driven by a source that decays."""
    R2 = R(2)
    x, y = R2.base_scalars()
    t = R2.base_time()
    ue = sp.cos(x) * sp.sin(sp.pi * y) * sp.exp(-3 * t)
    f = sp.diff(ue, t) - NU * Div(Grad(ue))
    T = TensorProduct(
        Fourier.Fourier(N, name="Fsrc"),
        FunctionSpace(
            N + 2,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            name="Dsrc",
        ),
        name="Tsrc",
        system=R2,
    )
    v = TestFunction(T, name="vs")
    u = TrialFunction(T, name="us", transient=True)
    return IMEXRungeKutta(
        v * (u.diff(t) - Constant("nu", NU) * Div(Grad(u)) - f),
        tableau=ARS443,
        time=(0.0, 0.2),
        initial=ue.subs(t, 0),
        sparse=True,
    )


def test_a_moving_source_is_replicated_at_every_time() -> None:
    """Rebuilt per stage, so each rebuild has to come back replicated."""
    integrator = _diffusion_with_a_moving_source()
    assert integrator._transient_source
    assert not integrator._transient_boundary, "the walls are homogeneous"
    assert integrator._source is not None
    early, late = integrator.forcing_at(0.0), integrator.forcing_at(0.2)
    assert early is not None and late is not None
    # Without this the placement assertions would hold for a frozen source too.
    assert not jnp.allclose(early, late)
    for forcing in (early, late, integrator.forcing_at(0.1)):
        assert jnp.asarray(forcing).is_fully_replicated
    for vector in integrator._source._vectors:
        assert jnp.asarray(vector).is_fully_replicated


def test_a_moving_source_integrates_under_spmd() -> None:
    """A sharded solve whose right-hand side moves as it steps."""
    out = _diffusion_with_a_moving_source().solve(
        dt=0.01, steps=20, n_batches=2, progress=False
    )
    assert jnp.all(jnp.isfinite(out))


def test_a_moving_source_batches_agree_with_one_long_batch() -> None:
    """Per-batch start times must reach the source, not just the stepping.

    Several batches hand `_advance` a different start time each; one batch hands
    it a single one. The source is rebuilt from that time inside the step, so a
    start time that failed to reach it would show up here as a disagreement.
    """
    batched = _diffusion_with_a_moving_source().solve(
        dt=0.01, steps=20, n_batches=4, progress=False
    )
    single = _diffusion_with_a_moving_source().solve(
        dt=0.01, steps=20, n_batches=1, progress=False
    )
    assert jnp.allclose(batched, single, atol=1e-6)
