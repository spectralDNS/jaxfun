from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as npcheb, legendre as nplegendre

from jaxfun.galerkin import (
    Chebyshev,
    ChebyshevU,
    FunctionSpace,
    JAXFunction,
    Legendre,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.composite import (
    BCGeneric,
    BoundaryConditions,
    Composite,
    DirectSum,
    get_bc_basis,
    get_stencil_matrix,
)
from jaxfun.utils.common import n


def test_boundary_conditions_basic():
    bc = BoundaryConditions({"left": {"D": 1, "N": 0}, "right": {"D": 0}})
    assert bc.orderednames() == ["LD", "LN", "RD"]
    assert bc.num_bcs() == 3
    assert bc.num_derivatives() == 1  # One N contributes one derivative order
    assert not bc.is_homogeneous()
    h = bc.get_homogeneous()
    assert h.is_homogeneous()


def test_get_stencil_matrix_special_cases():
    # Special case LDRD for Chebyshev and Legendre
    bcs_dd = BoundaryConditions({"left": {"D": 0}, "right": {"D": 0}})
    C = Chebyshev.Chebyshev(8)
    L = Legendre.Legendre(8)
    stC = get_stencil_matrix(bcs_dd, C)
    stL = get_stencil_matrix(bcs_dd, L)
    assert stC[0] == 1 and stC[2] == -1
    assert stL[0] == 1 and stL[2] == -1
    # Special case LDLNRDRN
    bcs_dn = BoundaryConditions({"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}})
    stC2 = get_stencil_matrix(bcs_dn, C)
    stL2 = get_stencil_matrix(bcs_dn, L)
    # Should have keys 0,2,4
    assert set(stC2.keys()) == {0, 2, 4}
    assert set(stL2.keys()) == {0, 2, 4}


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, ChebyshevU.ChebyshevU)
)
def test_composite_and_mass_matrix(space):
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    C = Composite(10, space, bcs)
    # Mass matrix should be SPD and same shape as dim
    M = C.mass_matrix().todense()
    assert M.shape == (C.dim, C.dim)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, ChebyshevU.ChebyshevU)
)
def test_bcgeneric_space(space):
    bcs = {"left": {"D": 0, "N": 1}, "right": {"D": 2}}
    B = BCGeneric(3, space, bcs)
    assert B.dim == 3
    # quad_points_and_weights should use M when N==0
    xw = B.quad_points_and_weights()
    assert xw[0].shape[0] == B.orthogonal.num_quad_points


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, ChebyshevU.ChebyshevU)
)
def test_direct_sum_evaluate_backward(space):
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    FS = FunctionSpace(6, space, bcs=bcs)
    assert isinstance(FS, DirectSum)
    C, _ = FS[0], FS[1]
    c = jnp.ones(C.dim)
    val = FS.evaluate(0.1, c)
    # Evaluate should add base solution and boundary lift
    assert jnp.isfinite(val)
    u = jnp.ones(C.dim)
    uh = FS.backward(u)
    assert uh.shape[0] == C.num_quad_points
    u = JAXFunction(jnp.zeros(4), FS)
    assert u(1) == 2.0
    assert u(-1) == 1.0


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, ChebyshevU.ChebyshevU)
)
def test_get_bc_basis(space):
    bcs = {"left": {"D": 0}, "right": {"N": 0}}
    L = space(6)
    B = get_bc_basis(BoundaryConditions(bcs), L)
    assert B.shape[0] == 2  # number of bcs


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, ChebyshevU.ChebyshevU)
)
def test_get_homogeneous(space):
    bcs = {"left": {"D": 1}, "right": {"N": 1}}
    C = Composite(8, space, bcs)
    H = C.get_homogeneous()
    assert H.bcs.is_homogeneous()


@pytest.mark.parametrize(
    "space,alpha,side,row",
    [
        # T_1 satisfies u(-1) + u'(-1) = 0 by itself, T_2 does u(-1) + u'(-1)/4.
        (Chebyshev.Chebyshev, 1, "left", 0),
        (Chebyshev.Chebyshev, 0.25, "left", 1),
        (Chebyshev.Chebyshev, -1, "right", 0),
        (Legendre.Legendre, 1, "left", 0),
        (Legendre.Legendre, 1 / 3, "left", 1),
        (Legendre.Legendre, 2, "left", None),
    ],
)
def test_single_robin_condition(space, alpha, side, row):
    """Where one orthogonal function meets the condition alone, the two-term
    rule is singular in the row before it, which skips one function instead."""
    N = 12
    B = cast(Composite, FunctionSpace(N, space, bcs={side: {"R": (alpha, 0)}}))
    val, der = (
        (npcheb.chebval, npcheb.chebder)
        if space is Chebyshev.Chebyshev
        else (nplegendre.legval, nplegendre.legder)
    )
    S = np.asarray(B.S.todense())
    x = -1 if side == "left" else 1
    assert B.dim == N - 1 and np.linalg.matrix_rank(S) == N - 1
    bc = [val(x, r) + alpha * val(x, der(r)) for r in S]
    # The derivative term reaches about alpha N^2.
    assert np.abs(bc).max() < 10 * np.finfo(S.dtype).eps * N**2 * max(1, abs(alpha))
    assert sorted(B.stencil.overrides) == ([] if row is None else [row])


def _dense(V) -> np.ndarray:
    return np.asarray(V.S.todense())


@pytest.mark.parametrize("space", [Chebyshev.Chebyshev, Legendre.Legendre])
def test_scaling_is_applied_once(space):
    N = 10
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    U = cast(Composite, FunctionSpace(N, space, bcs=bcs))
    B = cast(Composite, FunctionSpace(N, space, bcs=bcs, scaling=n + 1))
    k = np.arange(U.dim)[:, None]
    assert np.allclose(_dense(B), _dense(U) / (k + 1))
    # A copy keeps the scaling, applied once.
    assert B.get_homogeneous() is B
    C = Composite(N, space, {"left": {"D": 1}, "right": {"D": 2}}, scaling=n + 1)
    H = C.get_homogeneous()
    assert H.scaling == n + 1 and np.allclose(_dense(H), _dense(B))
    # A test space does not inherit it: unscaled unless asked, then scaled once.
    assert U.get_testspace("G") is U
    V = B.get_testspace("G")
    assert V is not B and V.scaling == 1 and np.allclose(_dense(V), _dense(U))
    assert np.allclose(_dense(B.get_testspace("G", name="v")), _dense(U))
    W = B.get_testspace("G", scaling=n + 2)
    assert np.allclose(_dense(W), _dense(U) / (k + 2))
    # The precomputed matrices, Legendre's from scaled derivative stencils, too.
    x = B.system.x
    u, v = TrialFunction(B), TestFunction(W)
    for form in (u.diff(x, 2) * v, u.diff(x, 1) * v, u * v):
        A = np.asarray(inner(form, sparse=True, kind="bilinear").todense())
        ref = inner(form, use_precomputed_matrices=False, kind="bilinear")
        ref = np.asarray(ref.todense())
        assert np.abs(A - ref).max() < 1e3 * np.finfo(A.dtype).eps * np.abs(ref).max()
