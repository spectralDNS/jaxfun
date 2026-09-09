"""Boundary blocks kept apart from their lifting.

`inner` contracts the boundary block of a bilinear form into a load vector while
it assembles, which pins it to whatever lifting the space held at the time. That
is right for steady boundary data and wrong for data that moves: a time
integrator needs the block again at every stage.

`inner_boundary` returns the same blocks uncontracted. The two share one
definition of each contraction, so what these tests check is not that the
numbers happen to agree but that nothing else crept in between them.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import x
from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    Legendre,
    TensorProduct,
)
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.inner import inner, inner_boundary
from jaxfun.galerkin.tensorproductspace import DirectSumTPS
from jaxfun.operators import Div, Grad
from tests.galerkin.test_boundary_lifting import CASES

pytestmark = pytest.mark.integration


def _helmholtz(V):
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u")
    return v * (Div(Grad(u)) + u)


@pytest.mark.parametrize("case", list(CASES))
def test_deferred_boundary_equals_the_collapsed_one(case: str) -> None:
    """Same operators, same contraction, same numbers -- to the last bit."""
    T, _ = CASES[case]()
    form = _helmholtz(T)

    _, collapsed = inner(form, sparse=True, kind="system")
    deferred = inner_boundary(form)

    assert isinstance(collapsed, jax.Array)  # scalar, so not a block system
    assert deferred is not None
    assert not deferred.is_transient
    # The form carries no source term, so the whole right-hand side is boundary.
    assert jnp.array_equal(deferred(), collapsed)


def test_deferred_boundary_equals_the_collapsed_one_in_1d() -> None:
    """One dimension reaches the boundary block by a separate path inside `inner`."""
    D = FunctionSpace(
        20, Chebyshev.Chebyshev, bcs={"left": {"D": 1.0}, "right": {"D": -0.5}}
    )
    form = _helmholtz(D)

    _, collapsed = inner(form, sparse=True, kind="system")
    deferred = inner_boundary(form)

    assert isinstance(collapsed, jax.Array)  # scalar, so not a block system
    assert deferred is not None
    assert jnp.array_equal(deferred(), collapsed)


def test_a_form_without_boundary_data_has_no_blocks() -> None:
    V = FunctionSpace(12, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    assert inner_boundary(_helmholtz(V)) is None


def _transient_2d():
    F = Fourier.Fourier(16, name="Fb")
    t = F.system.base_time()
    g = sp.cos(2 * x) * sp.sin(3 * t)
    D = FunctionSpace(
        14, Legendre.Legendre, bcs={"left": {"D": g}, "right": {"D": 2 * g}}, name="Db"
    )
    return cast(DirectSumTPS, TensorProduct(F, D, name="Tb"))


def test_transient_boundary_forcing_moves_and_traces() -> None:
    T = _transient_2d()
    bf = inner_boundary(_helmholtz(T))

    assert bf is not None and bf.is_transient
    assert not jnp.allclose(bf(0.0), bf(0.9))
    assert jnp.allclose(bf(0.37), jax.jit(bf)(0.37))


def test_rate_matches_forward_mode_ad() -> None:
    """The symbolic derivative is the real one, in 2D and in 1D."""
    bf = inner_boundary(_helmholtz(_transient_2d()))
    assert bf is not None
    ad = jax.jacfwd(bf)(0.37)
    assert jnp.abs(ad - bf.rate(0.37)).max() < 1e-5 * jnp.abs(ad).max()

    t = FunctionSpace(20, Legendre.Legendre).system.base_time()
    W = FunctionSpace(
        20, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": sp.sin(t)}}
    )
    v, u = TestFunction(W, name="vm"), TrialFunction(W, name="um")
    mass = inner_boundary(v * u)
    assert mass is not None
    ad1 = jax.jacfwd(mass)(0.4)
    assert jnp.abs(ad1 - mass.rate(0.4)).max() < 1e-5 * jnp.abs(ad1).max()


@pytest.mark.parametrize("case", list(CASES))
def test_static_boundary_forcing_has_an_exactly_zero_rate(case: str) -> None:
    """What lets a steady lifting drop out of d/dt as an identity."""
    bf = inner_boundary(_helmholtz(CASES[case]()[0]))
    assert bf is not None
    assert float(jnp.abs(bf.rate(0.0)).max()) == 0.0


def test_a_linear_dirichlet_lifting_contributes_no_stiffness_forcing() -> None:
    """Physics worth pinning: a two-point Dirichlet lifting is linear in x.

    Its Laplacian vanishes identically, so a moving wall shows up in the mass
    term and nowhere in the stiffness term. `examples/RayleighBenard.py` relies
    on the same fact.
    """
    t = FunctionSpace(20, Legendre.Legendre).system.base_time()
    W = FunctionSpace(
        20, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": sp.sin(t)}}
    )
    v, u = TestFunction(W, name="vs"), TrialFunction(W, name="us")

    stiffness = inner_boundary(v * Div(Grad(u)))
    assert stiffness is not None
    assert float(jnp.abs(stiffness(1.0)).max()) == 0.0

    mass = inner_boundary(v * u)
    assert mass is not None
    assert float(jnp.abs(mass(1.0)).max()) > 1e-3
