"""Threading the current time through a step, under SPMD.

`dt` had to be made a *static* argument of `_advance` because, left traced as a
rank-0 parameter, GSPMD propagated `P("k")` onto it from the sharded arrays it
is multiplied into and JAX then raised an `IndexError` out of
`_to_xla_hlo_sharding`. The step time is structurally the same kind of value and
reaches the same arithmetic, but it cannot be static: `solve` uses a different
start time for every batch, so a static one would recompile once per batch.

It is carried as a shape-`(1,)` array instead, and constrained to a replicated
sharding on the way in. `_diffusion_with_a_lifted_wall` is the canary for that
mechanism alone: its wall is deliberately *steady*, so a failure there points at
the time threading rather than at anything the lifting does.

`_diffusion_with_a_moving_wall` is the other half, and the one that exercises
what the threading is for. Its lifting is rebuilt at every stage from inside the
jitted step, so the arrays that come out have to be replicated where the steady
ones were merely constant -- a distinct hazard, and not one the canary can see.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/integrators/test_transient_bcs_spmd.py --num-devices=2
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import R
from jaxfun.galerkin import Fourier, FunctionSpace, Legendre, TensorProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.integrators import ARS443, IMEXRungeKutta, SystemIMEXRungeKutta
from jaxfun.integrators._utils import apply_field_couplings
from jaxfun.integrators.base import _as_start_time, _step_time
from jaxfun.operators import Constant, Div, Grad

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)

# The leading axis has to divide across the mesh for assembly to shard anything,
# which is the case these exist for.
N = 4 * jax.device_count()

# A Dirichlet pair costs two dofs, so this is what makes the *constrained*
# space's leading axis divide across the mesh.
Nc = 2 + 4 * jax.device_count()


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


def _diffusion_with_a_moving_wall():
    """The same problem with a wall that moves in time and along the sharded axis."""
    R2 = R(2)
    x, y = R2.base_scalars()
    t = R2.base_time()
    wall = sp.cos(x) * sp.exp(-t)
    T = TensorProduct(
        Fourier.Fourier(N, name="Fm"),
        FunctionSpace(
            N + 2,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": wall}},
            name="Dm",
        ),
        name="Tm",
    )
    v = TestFunction(T, name="vm")
    u = TrialFunction(T, name="um", transient=True)
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
    assert not integrator._transient_boundary, "this fixture is the steady canary"
    forcing = integrator.forcing_at(0.0)
    if forcing is None:
        pytest.skip("this lifting contributes no forcing")
    assert jnp.asarray(forcing).is_fully_replicated


def test_a_moving_lifting_is_replicated_at_every_time() -> None:
    """The arrays a moving lifting produces are rebuilt inside the jitted step.

    A steady forcing is replicated once at construction and stays that way. A
    moving one is recomputed per stage from `t`, so each rebuild is a fresh
    chance to come back on the assembly sharding -- which `_advance`'s closure
    cannot carry across devices.
    """
    integrator = _diffusion_with_a_moving_wall()
    assert integrator._transient_boundary
    assert integrator._linear_boundary is not None
    assert integrator._mass_boundary is not None
    early, late = integrator.forcing_at(0.0), integrator.forcing_at(0.1)
    assert early is not None and late is not None
    # Without this the assertions below would hold for a frozen lifting too.
    assert not jnp.allclose(early, late)
    for forcing in (early, late):
        assert jnp.asarray(forcing).is_fully_replicated
    for block in (integrator._linear_boundary, integrator._mass_boundary):
        assert jnp.asarray(block(0.05)).is_fully_replicated
        assert jnp.asarray(block.rate(0.05)).is_fully_replicated


def test_a_moving_wall_integrates_under_spmd() -> None:
    """The whole point: a sharded solve whose boundary data moves as it steps."""
    out = _diffusion_with_a_moving_wall().solve(
        dt=0.01, steps=10, n_batches=2, progress=False
    )
    assert jnp.all(jnp.isfinite(out))


def test_a_moving_wall_batches_agree_with_one_long_batch() -> None:
    """Per-batch start times must reach the lifting, not just the stepping.

    Several batches hand `_advance` a different start time each; one batch hands
    it a single one. The lifting is rebuilt from that time inside the step, so a
    start time that failed to reach it would show up here as a disagreement.
    """
    batched = _diffusion_with_a_moving_wall().solve(
        dt=0.01, steps=10, n_batches=5, progress=False
    )
    single = _diffusion_with_a_moving_wall().solve(
        dt=0.01, steps=10, n_batches=1, progress=False
    )
    assert jnp.allclose(batched, single, atol=1e-6)


def _coupled_system_with_a_moving_wall():
    """A field with no boundary data of its own, coupled to one with a wall.

    `u` is orthogonal -- nothing about its equation looks transient. All the
    motion reaches it through the coupling to `v`, whose wall moves. The
    coupling's boundary block is therefore rebuilt from `t` inside the jitted
    step, where a steady one would have been a constant.
    """
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    R2 = R(2)
    x, _ = R2.base_scalars()
    t = R2.base_time()
    wall = (1 - x**2) * sp.exp(-t)
    V = TensorProduct(
        FunctionSpace(Nc, Legendre.Legendre, bcs=hom, name="Vcx", fun_str="Lcx"),
        FunctionSpace(
            Nc,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": wall}},
            name="Vcy",
            fun_str="Lcy",
        ),
        name="Vc",
    )
    U = V.get_orthogonal()
    v = TrialFunction(V, name="vc")
    q = TestFunction(V, name="qc")
    u = TrialFunction(U, name="uc", transient=True)
    w = TestFunction(U, name="wc")
    return (
        SystemIMEXRungeKutta(
            ((u.diff(t) + u + v) * w, (Div(Grad(v)) - v + u) * q),
            tableau=ARS443,
            time=(0.0, 6.0),
            initial=(sp.Integer(0), None),
            sparse=True,
        ),
        V,
    )


def test_a_moving_coupling_block_is_replicated_at_every_time() -> None:
    """A coupling's lifting is rebuilt per stage, so it can come back sharded.

    `inner` folds a coupled foreign field's boundary block into a load vector,
    and the deferred replacement re-derives it from `t` inside the step. Each
    rebuild is a fresh chance to pick up the assembly sharding, which
    `_advance`'s closure cannot carry across devices -- and the equation's own
    space has no boundary conditions, so nothing else about it is suspicious.

    Reached through `apply_field_couplings` rather than by calling the block
    directly: that is the path the step takes, and a block that is correct but
    read at the wrong time would be invisible to a direct call.
    """
    integrator, V = _coupled_system_with_a_moving_wall()
    equation = integrator.integrators[0]
    assert not equation._transient_boundary, "the motion is all in the coupling"
    ((_operator, _forcing, boundary),) = equation._couplings
    assert boundary is not None and boundary.is_transient
    # Divisible and multidimensional, so assembly would have sharded it.
    assert V.num_dofs[0] % jax.device_count() == 0
    assert len(V.num_dofs) >= 2

    state = tuple(jnp.zeros_like(c) for c in integrator.initial_coefficients())
    slots = equation._coupling_slots
    blocks = []
    for ti in (0.0, 0.1, 0.2):
        block = apply_field_couplings(slots, equation._couplings, state, ti)
        assert block is not None
        blocks.append(block)
    # Without this the placement assertions would hold for a frozen block too.
    assert not jnp.allclose(blocks[0], blocks[-1])
    for block in blocks:
        assert jnp.asarray(block).is_fully_replicated


def test_a_moving_coupling_decays_under_spmd() -> None:
    """The sharded run of the physics the non-SPMD suite pins.

    `u` is driven only through `v`, and `v` only by a wall decaying like
    `exp(-t)`, so `u` has to decay. Held at the wall's initial value the
    coupling never switches off and `u` climbs to a steady state instead. Run
    here with the operators distributed, which is what says the deferred block
    survives being assembled across devices.
    """
    integrator, _ = _coupled_system_with_a_moving_wall()
    u_hats, v_hats = integrator.solve(
        dt=0.05, steps=120, n_batches=6, return_batch_snapshots=True, progress=False
    )
    assert jnp.all(jnp.isfinite(u_hats)) and jnp.all(jnp.isfinite(v_hats))
    U = integrator.integrators[0].trialspace
    peak = float(jnp.abs(U.backward(u_hats[1])).max())
    final = float(jnp.abs(U.backward(u_hats[-1])).max())
    assert final < 0.2 * peak, (peak, final)
