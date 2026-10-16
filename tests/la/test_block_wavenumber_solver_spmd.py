"""Block wavenumber solves under SPMD, with 2 or 4 local devices.

The wavenumbers are split over the devices as the state is, and each device
assembles, factorizes and solves only its own. So the factors are born split,
and the compiled solve has nothing to communicate.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/la/test_block_wavenumber_solver_spmd.py --num-devices=2
"""

import re

import jax
import jax.numpy as jnp
import pytest

from jaxfun.la.blockwavenumber import (
    BlockFourierSolver,
    BlockWavenumberSolver,
    block_lu_factor,
)
from jaxfun.sharding import state_sharding
from jaxfun.utils.common import ulp
from tests.la.test_block_wavenumber_solver import (
    CHEB_PINS,
    boussinesq_operator,
    manufactured,
    stokes_operator,
)

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)

COLLECTIVE = re.compile(
    r"\b(all-gather|all-to-all|all-reduce|collective-permute|reduce-scatter)\("
)


def factors(solver) -> tuple[jax.Array, ...]:
    if isinstance(solver, BlockFourierSolver):
        return (solver.inverse,)
    if isinstance(solver, BlockWavenumberSolver):
        return solver.lu, solver.perm
    return solver.multipliers, solver.pivots, solver.U


@pytest.mark.parametrize("method", ["dense", "rcm"])
@pytest.mark.parametrize("kind", ["fourier_poly", "fourier_cheb", "fourier"])
def test_each_device_solves_its_own_wavenumbers(kind, method) -> None:
    mats, test_space, trial_space = stokes_operator(kind, M=8)
    pins = CHEB_PINS if kind == "fourier_cheb" else []
    solver = block_lu_factor(mats, test_space, trial_space, pins, method=method)
    for f in factors(solver):
        n_local = f.shape[0] // jax.device_count()
        assert {s.data.shape[0] for s in f.addressable_shards} == {n_local}

    x, b = manufactured(mats, test_space, trial_space, True, pins)
    b = tuple(jax.device_put(r, state_sharding(r.shape)) for r in b)
    hlo = jax.jit(lambda s, r: s.solve(r)).lower(solver, b).compile().as_text()
    assert hlo is not None and not COLLECTIVE.search(hlo)
    for xc, sc in zip(x, solver.solve(b), strict=True):
        assert sc.sharding.is_equivalent_to(state_sharding(xc.shape), xc.ndim)
        assert jnp.allclose(sc, xc, atol=float(1000 * ulp(1.0)))


def test_one_way_coupling_solves_without_communication() -> None:
    """Group by group, each group on the wavenumbers of its own device."""
    mats, test_space, trial_space = boussinesq_operator(M=8)
    pins = [(2, (0, 0))]
    solver = block_lu_factor(mats, test_space, trial_space, pins)
    x, b = manufactured(mats, test_space, trial_space, True, pins)
    b = tuple(jax.device_put(r, state_sharding(r.shape)) for r in b)
    hlo = jax.jit(lambda s, r: s.solve(r)).lower(solver, b).compile().as_text()
    assert hlo is not None and not COLLECTIVE.search(hlo)
    for xc, sc in zip(x, solver.solve(b), strict=True):
        assert sc.sharding.is_equivalent_to(state_sharding(xc.shape), xc.ndim)
        assert jnp.allclose(sc, xc, atol=float(1000 * ulp(1.0)))
