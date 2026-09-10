"""Characterization of the boundary lifting, branch by branch.

`DirectSumTPS` splits an inhomogeneous space into a homogeneous tensor product
plus one lifting block per combination of boundary factors, and stores each
block's coefficients in `bndvals`. The construction loop has four branches,
selected by how many factors of a key are `BCGeneric`, and they differ in what
they project onto and how they stack the result. Two of them -- the 3D ones --
have no other coverage.

These tests pin the contract in terms of observable behaviour rather than the
construction loop: which keys carry a lifting and what shape each holds, that a
zero homogeneous coefficient array evaluates to exactly the prescribed boundary
data, and that the lifting is what `backward` adds on top of the homogeneous
part. They are the oracle for moving the lifting out of `__init__`, so none of
them may reference where or when it is computed.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import R
from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    Legendre,
    TensorProduct,
)
from jaxfun.galerkin.composite import BCGeneric, DirectSum
from jaxfun.galerkin.inner import project1D
from jaxfun.galerkin.tensorproductspace import DirectSumTPS
from jaxfun.utils.common import lambdify

pytestmark = pytest.mark.integration


def _dirichlet(expr: sp.Expr, s: sp.Symbol, lo: float, up: float) -> dict:
    return {"left": {"D": expr.subs(s, lo)}, "right": {"D": expr.subs(s, up)}}


def _one_inhomogeneous_2d() -> tuple[DirectSumTPS, sp.Expr]:
    """Fourier x Legendre, inhomogeneous in y only -> `1 other + 1 bc`."""
    R2 = R(2)
    x, y = R2.base_scalars()
    ue = sp.cos(2 * x) * (2 + y)
    D = FunctionSpace(14, Legendre.Legendre, bcs=_dirichlet(ue, y, -1, 1), name="D")
    F = Fourier.Fourier(16, name="F")
    return cast(DirectSumTPS, TensorProduct(F, D, name="T1")), ue


def _two_inhomogeneous_2d() -> tuple[DirectSumTPS, sp.Expr]:
    """Legendre x Legendre, inhomogeneous in both -> adds the corner block."""
    R2 = R(2)
    x, y = R2.base_scalars()
    ue = sp.exp(-(x**2 + y**2))
    Dx = FunctionSpace(18, Legendre.Legendre, bcs=_dirichlet(ue, x, -1, 1), name="Dx")
    Dy = FunctionSpace(19, Legendre.Legendre, bcs=_dirichlet(ue, y, -1, 1), name="Dy")
    return cast(DirectSumTPS, TensorProduct(Dx, Dy, name="T2")), ue


def _one_inhomogeneous_3d() -> tuple[DirectSumTPS, sp.Expr]:
    """Fourier x Legendre x Chebyshev, inhomogeneous in z -> `2 other + 1 bc`."""
    R3 = R(3)
    x, y, z = R3.base_scalars()
    ue = sp.cos(2 * x) * (2 + y) * (3 + z)
    F = Fourier.Fourier(12, name="F3")
    L = FunctionSpace(10, Legendre.Legendre, name="L3")
    Dz = FunctionSpace(9, Chebyshev.Chebyshev, bcs=_dirichlet(ue, z, -1, 1), name="Dz")
    return cast(DirectSumTPS, TensorProduct(F, L, Dz, name="T3")), ue


def _two_inhomogeneous_3d() -> tuple[DirectSumTPS, sp.Expr]:
    """Fourier x Chebyshev x Chebyshev, inhomogeneous in y and z -> `1 other + 2 bc`.

    The y and z dependence must not be linear: an exactly-linear profile is
    reproduced by the corner block alone, leaving the two single-axis blocks at
    round-off and testing nothing.
    """
    R3 = R(3)
    x, y, z = R3.base_scalars()
    ue = sp.cos(2 * x) * sp.exp(y / 2) * sp.cos(z)
    F = Fourier.Fourier(12, name="F4")
    Dy = FunctionSpace(
        10, Chebyshev.Chebyshev, bcs=_dirichlet(ue, y, -1, 1), name="Dy4"
    )
    Dz = FunctionSpace(9, Chebyshev.Chebyshev, bcs=_dirichlet(ue, z, -1, 1), name="Dz4")
    return cast(DirectSumTPS, TensorProduct(F, Dy, Dz, name="T4")), ue


CASES = {
    "2d_one": _one_inhomogeneous_2d,
    "2d_two": _two_inhomogeneous_2d,
    "3d_one": _one_inhomogeneous_3d,
    "3d_two": _two_inhomogeneous_3d,
}

# Which axes of each lifting key are boundary factors. One entry per branch of
# the construction loop: a single boundary axis beside one or two others, and --
# where two axes are inhomogeneous -- the block where they meet.
EXPECTED_BC_AXES = {
    "2d_one": {(1,)},
    "2d_two": {(0,), (1,), (0, 1)},
    "3d_one": {(2,)},
    "3d_two": {(1,), (2,), (1, 2)},
}


def _bc_axes(key: tuple) -> tuple[int, ...]:
    return tuple(i for i, s in enumerate(key) if isinstance(s, BCGeneric))


@pytest.mark.parametrize("case", list(CASES))
def test_bndvals_keys_and_shapes(case: str) -> None:
    """Every tpspace but the homogeneous one carries exactly one lifting block.

    A block is shaped like the space it belongs to, with each boundary axis
    contracted to that axis' number of boundary conditions.
    """
    T, _ = CASES[case]()

    assert {_bc_axes(k) for k in T.bndvals} == EXPECTED_BC_AXES[case]

    for key, block in T.bndvals.items():
        expected = tuple(
            s.bcs.num_bcs() if isinstance(s, BCGeneric) else s.dim for s in key
        )
        assert tuple(block.shape) == expected, key

    hom = T.get_homogeneous()
    hom_keys = [k for k, v in T.tpspaces.items() if v is hom]
    assert len(hom_keys) == 1
    assert hom_keys[0] not in T.bndvals
    assert len(T.tpspaces) == len(T.bndvals) + 1


@pytest.mark.parametrize("case", list(CASES))
def test_lifting_attains_the_prescribed_boundary_values(case: str) -> None:
    """With no homogeneous part, the expansion *is* the boundary data.

    The functional statement of what the lifting is for, and the property that
    has to survive any change to how the coefficients are computed. Probed with
    `evaluate` because the quadrature mesh has no boundary points.
    """
    T, ue = CASES[case]()
    c = jnp.zeros(tuple(s.dim for s in T.get_homogeneous().basespaces))

    dims = len(T.basespaces)
    interior = jnp.linspace(-0.7, 0.7, 5)
    exact = lambdify(T.system.base_scalars(), ue)

    probed = 0
    for axis, space in enumerate(T.basespaces):
        bcs = getattr(space, "bcs", None)
        if bcs is None or bcs.is_homogeneous():
            continue
        for wall in (-1.0, 1.0):
            pts = jnp.stack(
                [
                    jnp.full_like(interior, wall) if i == axis else interior
                    for i in range(dims)
                ],
                axis=-1,
            )
            got = T.evaluate(pts, c)
            expected = exact(*[pts[:, i] for i in range(dims)])
            scale = float(jnp.abs(jnp.asarray(expected)).max())
            assert jnp.allclose(got, expected, atol=1e-5 * max(scale, 1.0)), (
                f"{case}: axis {axis} wall {wall}"
            )
            probed += 1

    inhomogeneous_axes = {a for axes in EXPECTED_BC_AXES[case] for a in axes}
    assert probed == 2 * len(inhomogeneous_axes)


@pytest.mark.parametrize("case", list(CASES))
def test_backward_decomposes_over_the_lifting_blocks(case: str) -> None:
    """`backward` is the homogeneous transform plus every lifting block's.

    Every block is transformed onto one explicitly given mesh: the blocks carry
    different numbers of quadrature points, and `to_orthogonal` pads them up
    before transforming.
    """
    T, _ = CASES[case]()
    hom = T.get_homogeneous()
    c = jnp.ones(tuple(s.dim for s in hom.basespaces))
    N = tuple(s.num_quad_points for s in T.basespaces)

    blocks = [v.backward(T.bndvals.get(f, c), N=N) for f, v in T.tpspaces.items()]
    got, expected = jnp.sum(jnp.array(blocks), axis=0), T.backward(c, N=N)
    assert jnp.linalg.norm(got - expected) < 1e-6 * jnp.linalg.norm(expected)


def test_one_bc_axis_block_is_the_projection_of_each_wall_value() -> None:
    """The `1 other + 1 bc` block is column-wise `project1D` of the wall data.

    An independent derivation of that branch: the lifting coefficients along the
    remaining axis are the expansion of each boundary expression in that axis'
    basis, in `orderedvals` order.
    """
    T, _ = _one_inhomogeneous_2d()
    (key,) = T.bndvals
    F, bcspace = key[0], cast(BCGeneric, key[1])

    expected = jnp.array([project1D(val, F) for val in bcspace.bcs.orderedvals()]).T
    assert jnp.allclose(T.bndvals[key], expected, atol=1e-12)


# --- the lifting as a function of time --------------------------------------


def _transient_2d() -> tuple[DirectSumTPS, sp.Expr, sp.Symbol]:
    """Fourier x Legendre with a wall value varying in both space and time."""
    R2 = R(2)
    x, y = R2.base_scalars()
    t = R2.base_time()
    F = Fourier.Fourier(16, name="Ft")
    g = sp.cos(2 * x) * sp.sin(3 * t)
    D = FunctionSpace(
        14, Legendre.Legendre, bcs={"left": {"D": g}, "right": {"D": 2 * g}}, name="Dt"
    )
    return cast(DirectSumTPS, TensorProduct(F, D, name="Tt")), g, t


def _transient_2d_both_axes() -> tuple[DirectSumTPS, sp.Symbol]:
    """Both axes inhomogeneous and time-dependent -> the corner block moves too."""
    R2 = R(2)
    x, y = R2.base_scalars()
    t = R2.base_time()
    ue = sp.exp(-(x**2 + y**2)) * (1 + sp.sin(t))
    Dx = FunctionSpace(12, Legendre.Legendre, bcs=_dirichlet(ue, x, -1, 1), name="Dxt")
    Dy = FunctionSpace(13, Legendre.Legendre, bcs=_dirichlet(ue, y, -1, 1), name="Dyt")
    return cast(DirectSumTPS, TensorProduct(Dx, Dy, name="Tt2")), t


@pytest.mark.parametrize("case", list(CASES))
def test_lifting_reproduces_the_cached_bndvals(case: str) -> None:
    """The cached dict is the plan evaluated once, not a separate computation."""
    T, _ = CASES[case]()
    assert not T.lifting.is_transient
    fresh = T.lifting()
    assert set(fresh) == set(T.bndvals)
    for key, block in fresh.items():
        assert jnp.array_equal(block, T.bndvals[key])


@pytest.mark.parametrize("case", list(CASES))
def test_static_boundary_data_has_an_exactly_zero_rate(case: str) -> None:
    """Constant data differentiates to zero symbolically, not to round-off.

    This is what lets the mass-term lifting drop out of `d/dt` as an identity
    rather than as an assertion about magnitudes.
    """
    T, _ = CASES[case]()
    for block in T.lifting.rate(0.0).values():
        assert float(jnp.abs(block).max()) == 0.0


def test_transient_lifting_traces_and_matches_forward_mode_ad() -> None:
    """`lifting` is jittable in `t`, and its symbolic rate is the true derivative.

    The cross-check that justifies differentiating the boundary values rather
    than the projection: the two routes are mathematically the same because the
    lifting is linear in those values, and this pins that they agree numerically.
    """
    T, _, _ = _transient_2d()
    L = T.lifting
    assert L.is_transient
    (key,) = L(0.0)
    t0 = 0.37

    assert jnp.allclose(L(t0)[key], jax.jit(lambda s: L(s)[key])(t0))

    ad = jax.jacfwd(lambda s: L(s)[key])(t0)
    symbolic = L.rate(t0)[key]
    assert jnp.abs(ad - symbolic).max() < 1e-5 * jnp.abs(ad).max()


def test_transient_lifting_attains_the_moving_wall() -> None:
    T, g, t = _transient_2d()
    c = jnp.zeros(tuple(s.dim for s in T.get_homogeneous().basespaces))
    xs = jnp.linspace(0.0, 5.0, 4)
    t0 = 0.37

    for wall, factor in ((-1.0, 1), (1.0, 2)):
        pts = jnp.stack([xs, jnp.full_like(xs, wall)], axis=-1)
        want = factor * jnp.cos(2 * xs) * jnp.sin(3 * t0)
        got = T.evaluate(pts, c, t=t0)
        assert jnp.abs(got - want).max() < 1e-5 * jnp.abs(want).max()


def test_transient_lifting_on_both_axes_moves_the_corner_block() -> None:
    """The corner-consistency values inherit the time dependence.

    `projected_bcs` is built by substituting and differentiating in *space*,
    which commutes with d/dt -- so the corner block, which is pure boundary
    data with both coordinates already evaluated at a wall, has to move in time
    like everything else.
    """
    T, _ = _transient_2d_both_axes()
    assert T.lifting.is_transient

    corner = [k for k in T.bndvals if len(_bc_axes(k)) == 2]
    assert len(corner) == 1
    early, late = T.lifting(0.0)[corner[0]], T.lifting(1.1)[corner[0]]
    assert not jnp.allclose(early, late)

    rate = T.lifting.rate(0.4)[corner[0]]
    ad = jax.jacfwd(lambda s: T.lifting(s)[corner[0]])(0.4)
    assert jnp.abs(ad - rate).max() < 1e-5 * max(float(jnp.abs(ad).max()), 1e-30)


def test_a_time_argument_leaves_the_cache_alone() -> None:
    """`t=` rebuilds; only `update_bndvals` rebinds the shared dict.

    A transform is reachable from inside a jitted step, where assigning to a
    dict held on a shared space would capture tracers.
    """
    T, _, _ = _transient_2d()
    (key,) = T.bndvals
    before = T.bndvals[key]

    T.backward(jnp.zeros(tuple(s.dim for s in T.get_homogeneous().basespaces)), t=0.9)
    assert jnp.array_equal(before, T.bndvals[key])

    T.update_bndvals(0.9)
    assert not jnp.array_equal(before, T.bndvals[key])
    assert jnp.array_equal(T.bndvals[key], T.lifting(0.9)[key])


def test_changing_boundary_conditions_is_visible_through_the_jit_cache() -> None:
    """A mutated lifting must not be hidden by an identity-keyed jit cache.

    `DirectSum`'s transforms are cached on the space's identity, so a lifting
    read from inside the trace would be baked into the first executable
    compiled. It crosses as an argument instead.
    """
    V = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 2}})
    assert isinstance(V, DirectSum)  # nonzero values, so a lifting is carried
    ends = jnp.array([-1.0, 1.0])
    c = jnp.zeros(V.num_dofs)

    assert jnp.allclose(V.evaluate(ends, c), jnp.array([1.0, 2.0]), atol=1e-5)
    V[1].bcs["right"]["D"] = 5
    assert jnp.allclose(V.evaluate(ends, c), jnp.array([1.0, 5.0]), atol=1e-5)


def test_transient_boundary_values_need_a_time() -> None:
    R1 = R(1)
    t = R1.base_time()
    W = FunctionSpace(
        8, Legendre.Legendre, bcs={"left": {"D": sp.sin(t)}, "right": {"D": 0}}
    )
    assert isinstance(W, DirectSum)
    assert jnp.allclose(W.bnd_vals(float(jnp.pi) / 2), jnp.array([1.0, 0.0]), atol=1e-6)
    with pytest.raises(ValueError, match="depend on time"):
        W.bnd_vals()


def test_a_plain_time_symbol_binds_like_a_plain_coordinate() -> None:
    """`sp.Symbol("t")` means time, the way `sp.Symbol("x")` already means x.

    Boundary values bind to coordinates by name -- that is how a plain `x`
    works. `BaseTime` compares unequal to a plain `Symbol("t")` because of its
    assumptions, so without canonicalizing by name the value would look constant
    in time and its lifting would silently freeze.
    """
    R2 = R(2)
    x, _ = R2.base_scalars()
    t = R2.base_time()
    D = FunctionSpace(
        8,
        Legendre.Legendre,
        bcs={
            "left": {"D": sp.cos(2 * x) * t},
            "right": {"D": 0},
        },
        name="Dplain",
    )
    T = cast(
        DirectSumTPS, TensorProduct(Fourier.Fourier(8, name="Fplain"), D, name="Tp")
    )
    assert T.lifting.is_transient
    (key,) = T.bndvals
    assert not jnp.allclose(T.lifting(0.0)[key], T.lifting(0.9)[key])


def test_an_unknown_symbol_in_a_boundary_value_is_rejected() -> None:
    """A name that is neither a coordinate nor time cannot be evaluated."""
    R2 = R(2)
    x, y = R2.base_scalars()
    D = FunctionSpace(
        8,
        Legendre.Legendre,
        bcs={"left": {"D": sp.cos(2 * x) * sp.Symbol("alpha")}, "right": {"D": 0}},
        name="Dbad",
    )
    with pytest.raises(ValueError, match="which the space cannot evaluate"):
        TensorProduct(Fourier.Fourier(8, name="Fbad"), D, name="Tbad")


def test_a_time_dependent_robin_coefficient_is_rejected() -> None:
    """Alpha enters the boundary basis itself, not the data applied to it."""
    from jaxfun.galerkin.composite import BoundaryConditions

    V = FunctionSpace(8, Legendre.Legendre)
    t = V.system.base_time()
    bcs = BoundaryConditions({"left": {"R": (sp.sin(t), 1)}, "right": {"D": 0}})
    with pytest.raises(NotImplementedError, match="Robin"):
        bcs.has_time(t)
