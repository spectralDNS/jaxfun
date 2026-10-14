"""CoupledIMEXRungeKutta under SPMD, with 2 or 4 local devices.

The state is split over the wavenumbers, and so are the stage solvers' factors:
each device factorizes and solves the block systems of its own wavenumbers.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/integrators/test_coupled_imex_rk_spmd.py --num-devices=2
"""

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import FunctionSpace, Legendre, TensorProduct
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import project
from jaxfun.integrators import ARS443, CoupledIMEXRungeKutta
from jaxfun.la.blockwavenumber import (
    BlockBandedWavenumberSolver,
    BlockWavenumberSolver,
)
from jaxfun.sharding import state_sharding
from tests.integrators.test_coupled_imex_rk import HOM, navier_stokes

pytestmark = pytest.mark.spmd

if jax.device_count() not in (1, 2, 4):
    pytest.skip("SPMD tests require 1, 2 or 4 devices", allow_module_level=True)


def assert_split(stepper: CoupledIMEXRungeKutta, state: tuple[jax.Array, ...]) -> None:
    """The state and every stage solver's factors are split over the devices."""
    for a in state:
        assert a.sharding.is_equivalent_to(state_sharding(a.shape), a.ndim)
        assert not a.is_fully_replicated
    for solver in stepper._stage_solvers:
        assert isinstance(solver, BlockWavenumberSolver | BlockBandedWavenumberSolver)
        f = solver.lu if isinstance(solver, BlockWavenumberSolver) else solver.U
        n_local = f.shape[0] // jax.device_count()
        assert {s.data.shape[0] for s in f.addressable_shards} == {n_local}


def test_taylor_green_vortex() -> None:
    nu = 0.1
    F = FunctionSpace(8, Fourier, name="F")
    VD = TensorProduct(F, F, name="VD")
    Q = TensorProduct(F, F, name="Q", system=VD.system)
    equations, (x, y), (i, j), _ = navier_stokes(VD, Q, nu)
    u0 = sp.sin(x) * sp.cos(y) * i - sp.cos(x) * sp.sin(y) * j
    tg = CoupledIMEXRungeKutta(
        equations, tableau=ARS443, initial=(u0, None), sparse=True
    )
    ux, uy, p = tg.solve(0.02, 10, progress=False)
    assert_split(tg, (ux, uy, p))
    decay = sp.exp(-2 * nu * 0.2)
    assert jnp.allclose(ux, project(sp.sin(x) * sp.cos(y) * decay, VD), atol=1e-5)
    assert jnp.allclose(uy, project(-sp.cos(x) * sp.sin(y) * decay, VD), atol=1e-5)


@pytest.mark.parametrize("method", ["dense", "rcm"])
def test_channel_keeps_continuity(method) -> None:
    F = FunctionSpace(4, Fourier, name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs=HOM, name="D")
    P = FunctionSpace(8, Legendre.Legendre, name="P")
    VD = TensorProduct(F, D, name="VD")
    Q = TensorProduct(F, P, name="Q", system=VD.system)
    equations, (x, y), (i, j), _ = navier_stokes(VD, Q, 1.0, 1.0)
    u0 = -4 * y * (1 - y**2) * sp.sin(x) * i - (1 - y**2) ** 2 * sp.cos(x) * j
    channel = CoupledIMEXRungeKutta(
        equations,
        tableau=ARS443,
        initial=(u0, None),
        sparse=True,
        solver_options={"method": method},
    )
    state = channel.solve(0.05, 4, progress=False)
    assert_split(channel, state)
    continuity = channel.apply_linear(state)[2]
    eps = jnp.finfo(state[0].dtype).eps
    assert jnp.abs(continuity).max() < 1000 * eps * jnp.abs(state[0]).max()
