"""Threading the current time through a step, under SPMD.

`dt` had to be made a *static* argument of `_advance` because, left traced as a
rank-0 parameter, GSPMD propagated `P("k")` onto it from the sharded arrays it
is multiplied into and JAX then raised an `IndexError` out of
`_to_xla_hlo_sharding`. The step time is structurally the same kind of value and
reaches the same arithmetic, but it cannot be static: `solve` uses a different
start time for every batch, so a static one would recompile once per batch.

It is carried as a shape-`(1,)` array instead, and constrained to a replicated
sharding on the way in. These tests are the canary for that: they exist to fail
here, in isolation, rather than inside a time-dependent boundary condition.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/integrators/test_transient_bcs_spmd.py --num-devices=2
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import Fourier, FunctionSpace, Legendre, TensorProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.integrators import ARS443, IMEXRungeKutta
from jaxfun.integrators.base import _as_start_time, _step_time
from jaxfun.operators import Constant, Div, Grad

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)

# The leading axis has to divide across the mesh for assembly to shard anything,
# which is the case these exist for.
N = 4 * jax.device_count()


def _diffusion_with_a_lifted_wall():
    """Heat equation on Fourier x Legendre with one wall held at 1."""
    F = Fourier.Fourier(N, name="Fs")
    D = FunctionSpace(
        N + 2, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 1}}, name="Ds"
    )
    T = TensorProduct(F, D, name="Ts")
    v = TestFunction(T, name="v")
    u = TrialFunction(T, name="u", transient=True)
    t = T.system.base_time()
    _, y = T.system.base_scalars()
    return IMEXRungeKutta(
        v * (u.diff(t) - Constant("nu", 0.1) * Div(Grad(u))),
        tableau=ARS443,
        time=(0.0, 0.1),
        initial=sp.sin(sp.pi * y),
        sparse=True,
    )


def test_the_step_time_is_carried_as_a_rank_one_array() -> None:
    """Never a rank-0 parameter -- that is the shape GSPMD cannot place."""
    t0 = _as_start_time(0.25)
    assert t0.ndim == 1 and t0.shape == (1,)
    assert float(_step_time(t0, 4, 0.1)) == pytest.approx(0.65)


def test_a_traced_start_time_survives_a_sharded_solve() -> None:
    """The canary: batches pass different start times through one compilation."""
    integrator = _diffusion_with_a_lifted_wall()
    out = integrator.solve(dt=0.01, steps=10, n_batches=2, progress=False)
    assert jnp.all(jnp.isfinite(out))


def test_batched_and_single_shot_solves_agree() -> None:
    """Several batches, each with its own start time, match one long batch.

    Both go through the same compiled `_advance`; if the start time were being
    mishandled the two would drift apart.
    """
    batched = _diffusion_with_a_lifted_wall().solve(
        dt=0.01, steps=10, n_batches=5, progress=False
    )
    single = _diffusion_with_a_lifted_wall().solve(
        dt=0.01, steps=10, n_batches=1, progress=False
    )
    assert jnp.allclose(batched, single, atol=1e-6)


def test_the_boundary_forcing_stays_replicated() -> None:
    """A forcing reached through `_advance` must not be left sharded."""
    integrator = _diffusion_with_a_lifted_wall()
    forcing = integrator.forcing_at(0.0)
    if forcing is None:
        pytest.skip("this lifting contributes no forcing")
    assert jnp.asarray(forcing).is_fully_replicated
