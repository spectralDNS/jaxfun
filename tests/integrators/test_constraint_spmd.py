"""Constraint solvers under SPMD, with 2 or 4 local devices.

The stepper reaches `_advance` as a *traced* pytree, so its arrays arrive as jit
arguments and may carry any sharding -- which is what lets the wavenumber solver
keep per-device factors. It was static once, and then every array it held had to
be replicated, because JAX refuses to close over one spanning devices the process
cannot address.

What tracing does *not* extend to is anything used as a Python value. An index, a
shape or a flag stored under `nnx.data` arrives as a tracer where a concrete value
is needed, which is why coupling slots live in a separate `nnx.static` attribute.

The failure these guard against is multi-*process*: within one process every
device is addressable, so what is checkable here is the placement and the
static/traced split rather than the crash.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/integrators/test_constraint_spmd.py --num-devices=2
"""

import jax
import jax.numpy as jnp
import pytest

from tests.integrators.test_constraint_equations import signal_integrator

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)

# `signal_spaces` puts an inhomogeneous Dirichlet value on one wall, so the
# constraint assembles a boundary-lifting forcing. N-2 dofs per direction, and
# the leading axis has to divide across the mesh for assembly to shard it at all
# -- which is the case this exists for.
N = 2 + 4 * jax.device_count()


def test_inhomogeneous_constraint_forcing_is_replicated() -> None:
    """The boundary lifting a constraint assembles must not stay sharded."""
    V, _U, integrator = signal_integrator(N=N)
    (constraint,) = integrator.constraints
    assert constraint.forcing is not None, "the wall value should force the constraint"
    # Divisible and multidimensional, so assembly would have sharded it.
    assert V.num_dofs[0] % jax.device_count() == 0
    assert len(V.num_dofs) >= 2
    assert constraint.forcing.is_fully_replicated


def test_coupling_slots_stay_concrete_under_tracing() -> None:
    """Slots index the state tuple, so tracing them makes them unusable.

    The regression a traced stepper introduces: `_couplings` used to carry
    `(slot, operator, forcing)` triples under `nnx.data`, and once the stepper
    stopped being static the slot arrived as a tracer and every constrained
    solve died in `apply_field_couplings`. `split_couplings` keeps the slots
    static; this is what says they stayed that way.
    """
    _V, _U, integrator = signal_integrator(N=N)
    (constraint,) = integrator.constraints
    if not constraint._coupling_slots:
        pytest.skip("this integrator assembles no field couplings")

    seen: list = []

    @jax.jit
    def _trace(c) -> jax.Array:
        seen.append(c._coupling_slots)
        return jnp.zeros(())

    _trace(constraint)
    (slots,) = seen
    assert all(isinstance(s, int) for s in slots), (
        f"coupling slots became tracers under jit: {slots}"
    )


def test_a_sharded_operator_survives_the_trace() -> None:
    """The freedom the traced stepper buys, which the static one forbade.

    A sharded array reached through a static stepper's closure raises
    `RuntimeError` on more than one process. Passed as an argument it is fine,
    and that is precisely what lets `TPMatricesWavenumberSolver` hold factors
    for one device's wavenumbers instead of every device holding all of them.
    """
    from jaxfun.sharding import spectral_sharding

    sharded = jax.device_put(
        jnp.arange(4 * jax.device_count(), dtype=float).reshape(jax.device_count(), 4),
        spectral_sharding,
    )
    assert not sharded.is_fully_replicated

    _V, _U, integrator = signal_integrator(N=N)
    integrator._probe_sharded = sharded  # noqa: SLF001

    @jax.jit
    def _step(obj, x: jax.Array) -> jax.Array:
        return x * obj._probe_sharded.sum()

    out = _step(integrator, jnp.ones(()))
    assert jnp.isfinite(out)


def test_inhomogeneous_constraint_integrates_under_spmd() -> None:
    """The system must still step, and settle, with the constraint distributed."""
    _V, U, integrator = signal_integrator(N=N)
    u_hats, v_hats = integrator.solve(
        dt=0.02, steps=40, n_batches=4, return_batch_snapshots=True, progress=False
    )
    assert bool(jnp.isfinite(u_hats).all()) and bool(jnp.isfinite(v_hats).all())
    peak = jnp.max(jnp.abs(jax.vmap(U.backward)(u_hats)), axis=(1, 2))
    # Both terms of the u equation are dissipative: a monotone approach from rest.
    assert bool(jnp.all(jnp.diff(peak) > 0))
    assert float(peak[-1]) < 2.0
