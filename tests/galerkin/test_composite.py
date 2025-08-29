import jax.numpy as jnp

from jaxfun.galerkin import Chebyshev, FunctionSpace, Legendre
from jaxfun.galerkin.composite import (
    BCGeneric,
    BoundaryConditions,
    Composite,
    DirectSum,
    get_bc_basis,
    get_stencil_matrix,
)


def test_boundary_conditions_basic():
    bc = BoundaryConditions({"left": {"D": 0, "N": 0}, "right": {"D": 0}})
    assert bc.orderednames() == ["LD", "LN", "RD"]
    assert bc.num_bcs() == 3
    assert bc.num_derivatives() == 1  # One N contributes one derivative order
    assert bc.is_homogeneous()
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


def test_composite_and_mass_matrix():
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    C = Composite(10, Chebyshev.Chebyshev, bcs)
    # Mass matrix should be SPD and same shape as dim
    M = C.mass_matrix().todense()
    assert M.shape == (C.dim, C.dim)
    # Stencil consistency with scipy diag construction
    scipyM = C.stencil_to_scipy_sparse().toarray()
    assert scipyM.shape[1] == C.N


def test_bcgeneric_space():
    bcs = {"left": {"D": 0, "N": 1}, "right": {"D": 2}}
    B = BCGeneric(5, Legendre.Legendre, bcs)
    assert B.dim == 5
    # quad_points_and_weights should use M when N==0
    xw = B.quad_points_and_weights()
    assert xw.shape[1] == B.orthogonal.M


def test_direct_sum_evaluate_backward():
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    FS = FunctionSpace(6, Legendre.Legendre, bcs=bcs)
    assert isinstance(FS, DirectSum)
    C, _ = FS[0], FS[1]
    c = jnp.ones(C.dim)
    val = FS.evaluate(0.1, c)
    # Evaluate should add base solution and boundary lift
    assert jnp.isfinite(val)
    u = jnp.ones(C.dim)
    uh = FS.backward(u)
    assert uh.shape[0] == C.M


def test_get_bc_basis():
    bcs = {"left": {"D": 0}, "right": {"N": 0}}
    L = Legendre.Legendre(6)
    B = get_bc_basis(BoundaryConditions(bcs), L)
    assert B.shape[0] == 2  # number of bcs


def test_get_homogeneous_and_padding():
    bcs = {"left": {"D": 0}, "right": {"N": 0}}
    C = Composite(8, Chebyshev.Chebyshev, bcs)
    H = C.get_homogeneous()
    assert H.bcs.is_homogeneous()
    P = C.get_padded(12)
    assert P.N == 12
