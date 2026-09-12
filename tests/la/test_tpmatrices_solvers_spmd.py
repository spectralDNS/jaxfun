"""`TPMatricesWavenumberSolver` with 2 or 4 local devices.

Each Fourier wavenumber owns an independent banded system, so the solver is
embarrassingly parallel and should need no communication at all. Getting there
means the *factors* are built per device rather than assembled whole and then
replicated: these check that they are, that the solve stays local, and that the
answers are unchanged.

Correctness is measured against the dense Kronecker solve, which knows nothing
about sharding, so it is a reference in every device configuration.

All tests are marked ``spmd`` and are skipped by default. Run with::

    pytest tests/la/test_tpmatrices_solvers_spmd.py -m spmd --num-devices=2
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import Chebyshev, FunctionSpace, Legendre, TensorProduct
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import inner
from jaxfun.la import TPMatrices
from jaxfun.la.tpmatrix import tpmats_to_kron, tpmats_wavenumber_factor
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import ulp
from tests.la.test_tpmatrices_solvers import _poisson_fourier_poly_2d

pytestmark = pytest.mark.spmd

POLY_SPACES = pytest.mark.parametrize(
    "poly", [Legendre.Legendre, Chebyshev.Chebyshev], ids=["legendre", "chebyshev"]
)

# Divisible by 1, 2 and 4 so the leading Fourier axis splits over any mesh the
# suite runs on -- `_check_shardable` rejects anything that does not.
N = 16

_BIHARMONIC = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}


def _biharmonic_fourier_poly_2d(n: int, poly):
    """Fourier x poly biharmonic: a wider band than Poisson, same generic path."""
    F = FunctionSpace(n, Fourier)
    D = FunctionSpace(n, poly, _BIHARMONIC)
    T = TensorProduct(F, D)
    v, u = TestFunction(T), TrialFunction(T)
    x, y = T.system.base_scalars()
    ue = sp.cos(2 * x) * (1 - y**2) ** 2
    A, b = inner(
        v * Div(Grad(Div(Grad(u)))) - v * Div(Grad(Div(Grad(ue)))),
        sparse=True,
        kind="system",
    )
    return T, cast(TPMatrices, A), b, ue


def _collectives(wn, b) -> dict[str, int]:
    """True cross-device ops in the compiled solve; `dynamic-slice` is local."""
    hlo = wn._solve_jit.lower(wn.L, wn.U, b).compile().as_text()
    ops = ("all-gather", "all-reduce", "collective-permute", "all-to-all")
    return {op: hlo.count(f" {op}(") for op in ops if hlo.count(f" {op}(")}


@POLY_SPACES
def test_sharded_solve_matches_kron_reference(poly) -> None:
    """The distributed answer is the undistributed answer."""
    _, A, b, _ = _poisson_fourier_poly_2d(N, poly)
    wn = tpmats_wavenumber_factor(A)
    ref = tpmats_to_kron(A.tpmats).solve(b.flatten()).reshape(b.shape)
    uh = wn.solve(b)
    assert uh.shape == b.shape
    assert float(jnp.max(jnp.abs(uh - ref))) < ulp(10)


@POLY_SPACES
def test_biharmonic_sharded_solve_is_consistent(poly) -> None:
    """A wider band than Poisson, to keep the generic banded path honest.

    The Poisson operator happens to produce one sub- and one super-diagonal at
    even offsets. Nothing may depend on that: Chebyshev bands and biharmonic
    problems go through the same code, and this is the case that would notice a
    solver quietly specialised to the narrow one.

    Checked by residual rather than against `tpmats_to_kron`. A fourth-order
    operator is ill-conditioned enough that the banded and dense factorisations
    part company in the suite's default float32 -- by 1.5e-2 here, on *one*
    device as much as on four, so it says nothing about sharding. The residual
    stays well-conditioned, and it still catches the failure that matters: a
    device solving with another device's factors leaves one behind.
    """
    _, A, b, _ = _biharmonic_fourier_poly_2d(N, poly)
    wn = tpmats_wavenumber_factor(A)
    uh = wn.solve(b)
    assert uh.shape == b.shape
    residual = float(jnp.max(jnp.abs(A @ uh - b))) / float(jnp.max(jnp.abs(b)))
    assert residual < jnp.sqrt(ulp(10)), f"relative residual {residual:.2e}"


def test_factors_are_sharded_like_the_rhs() -> None:
    """The mismatch that used to cost a reshard on every single solve.

    Factors built whole carry `SingleDeviceSharding` while the right-hand side
    arrives split, so JAX reshards them before each call -- outside the module,
    where grepping the HLO for collectives does not show it. Matching shardings
    is what removes it.
    """
    _, A, b, _ = _poisson_fourier_poly_2d(N, Legendre.Legendre)
    wn = tpmats_wavenumber_factor(A)
    assert wn.L.sharding == b.sharding
    assert wn.U.sharding == b.sharding

    wants = wn._solve_jit.lower(wn.L, wn.U, b).compile().input_shardings[0]
    assert wants[0] == wn.L.sharding, "executable would reshard L on every call"
    assert wants[1] == wn.U.sharding, "executable would reshard U on every call"


def test_factor_memory_scales_with_device_count() -> None:
    """The objective: each device holds its own wavenumbers, not everyone's.

    Timing cannot show this on CPU devices that share memory, so assert the
    block sizes directly -- it is the property the whole change exists for.
    """
    n_dev = jax.device_count()
    _, A, b, _ = _poisson_fourier_poly_2d(N, Legendre.Legendre)
    wn = tpmats_wavenumber_factor(A)
    for name, fac in (("L", wn.L), ("U", wn.U)):
        n_F = fac.shape[0]
        local = fac.addressable_shards[0].data.shape[0]
        assert local == n_F // n_dev, (
            f"{name} block is {local}, expected {n_F}/{n_dev} = {n_F // n_dev}"
        )


@POLY_SPACES
def test_solve_needs_no_communication(poly) -> None:
    """Wavenumbers are independent, so a correct solve moves nothing."""
    _, A, b, _ = _poisson_fourier_poly_2d(N, poly)
    wn = tpmats_wavenumber_factor(A)
    assert _collectives(wn, b) == {}
