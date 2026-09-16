import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import BaseVector, get_CoordSys
from jaxfun.galerkin import (
    Chebyshev,
    ChebyshevU,
    Fourier,
    Legendre,
    TensorProduct,
    TensorProductSpace,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.inner import inner, project
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.operators import Div, Grad, dot
from jaxfun.typing import ProjectionKind
from jaxfun.utils.common import Domain, lambdify, ulp

x, r, theta, z, phi = sp.symbols("x,r,theta,z,phi", real=True, positive=True)

polspaces: dict[str, type[OrthogonalSpace]] = {
    "Chebyshev": Chebyshev.Chebyshev,
    "Legendre": Legendre.Legendre,
    "ChebyshevU": ChebyshevU.ChebyshevU,
    "Fourier": Fourier.Fourier,
}


def get_polar():
    return get_CoordSys(
        "P", sp.Lambda((theta, r), (r * sp.cos(theta), r * sp.sin(theta)))
    )


def get_polar_space(space) -> TensorProductSpace:
    P = get_polar()
    F = Fourier.RFourier(8, name="F")
    L = polspaces[space](8, name="L", domain=Domain(0, 1))
    return TensorProduct(F, L, system=P, name="P")


def get_cylindrical():
    return get_CoordSys(
        "C", sp.Lambda((theta, r, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    )


def get_cylindrical_space(space) -> TensorProductSpace:
    P = get_cylindrical()
    F = Fourier.RFourier(8, name="F")
    L = polspaces[space](8, name="L", domain=Domain(0, 1))
    return TensorProduct(F, L, L, system=P, name="C")


def get_polar_spherical():
    r = 1
    return get_CoordSys(
        "T",
        sp.Lambda(
            (theta, phi),
            (
                r * sp.sin(theta) * sp.cos(phi),
                r * sp.sin(theta) * sp.sin(phi),
                r * sp.cos(theta),
            ),
        ),
        assumptions=sp.Q.positive(theta)
        & sp.Q.positive(phi)
        & sp.Q.positive(sp.sin(theta)),
    )


def get_polar_spherical_space(space) -> TensorProductSpace:
    P = get_polar_spherical()
    F1 = Fourier.RFourier(8, name="F", domain=Domain(0, 2 * sp.pi))
    F2 = polspaces[space](8, name="L", domain=Domain(0, sp.pi))
    return TensorProduct(F1, F2, system=P, name="S")


def get_spherical():
    return get_CoordSys(
        "S",
        sp.Lambda(
            (theta, phi, r),
            (
                r * sp.sin(theta) * sp.cos(phi),
                r * sp.sin(theta) * sp.sin(phi),
                r * sp.cos(theta),
            ),
        ),
        assumptions=sp.Q.positive(theta)
        & sp.Q.positive(phi)
        & sp.Q.positive(r)
        & sp.Q.positive(sp.sin(theta)),
    )


def get_spherical_space(space) -> TensorProductSpace:
    P = get_spherical()
    F1 = Fourier.RFourier(8, name="F", domain=Domain(0, 2 * sp.pi))
    L1 = polspaces[space](8, name="L1", domain=Domain(0, sp.pi))
    L2 = polspaces[space](8, name="L2", domain=Domain(0, 1))
    return TensorProduct(F1, L1, L2, system=P, name="S")


def get_clustering(beta: sp.Expr | float, x0: sp.Expr | float = 0):
    """Return the map ``xi -> x`` clustering by ``(1+beta)/(1-beta)`` at ``x0``.

    Args:
        beta: Strength, in ``[0, 1)``. Zero is the identity.
        x0: Where resolution is concentrated. Pass it exactly (``sp.pi``, a
            `Rational`) rather than as a float.
    """

    def _exact(value: sp.Expr | float) -> sp.Expr:
        if isinstance(value, sp.Basic):
            return value
        return sp.Rational(value).limit_denominator(10**6)

    b, c = _exact(beta), _exact(x0)
    if not 0 <= float(b) < 1:
        raise ValueError(f"beta must lie in [0, 1), got {float(b)}")
    m = sp.Lambda((x,), (c + 2 * sp.atan(((1 - b) / (1 + b)) * sp.tan((x - c) / 2)),))
    return get_CoordSys("C", m)


def get_clustering_space(
    space: str, beta: sp.Expr | float = 0.8, x0: sp.Expr | float = 0
) -> OrthogonalSpace:
    system = get_clustering(beta, x0)
    C = polspaces[space]
    return C(8, system=system, name="F")


spaces: dict[str, Callable[..., OrthogonalSpace | TensorProductSpace]] = {
    "polar": lambda space: get_polar_space(space),
    "cylindrical": lambda space: get_cylindrical_space(space),
    "polar_spherical": lambda space: get_polar_spherical_space(space),
    "sphere": lambda space: get_spherical_space(space),
    "clustering": lambda space: get_clustering_space(space, 0.8, 0),
}

args = list(
    itertools.product(
        (
            "polar",
            "cylindrical",
            "polar_spherical",
            "sphere",
            "clustering",
        ),
        ("Legendre", "Chebyshev", "ChebyshevU"),
    )
)


def get_curve():
    """A 1D system whose metric varies along it: sg = sqrt(1 + t**2)."""
    return get_CoordSys("V", sp.Lambda((x,), (x, x**2 / 2)))


def get_helix():
    """A 1D system with a constant, non-unit metric."""
    return get_CoordSys(
        "H", sp.Lambda((x,), (sp.sin(2 * sp.pi * x), sp.cos(2 * sp.pi * x), 2 * x))
    )


# 1D curvilinear spaces, where `sg` is the space's own metric rather than the
# trivial one a `SubCoordSys` hands each axis of a tensor product. `Fourier`
# only appears against the clustering map, which is the periodic one.
spaces1d: dict[str, Callable[..., OrthogonalSpace]] = {
    "clustering": lambda space: get_clustering_space(space, 0.8, 0),
    "curve": lambda space: polspaces[space](12, system=get_curve(), name="Cu"),
    "helix": lambda space: polspaces[space](12, system=get_helix(), name="He"),
}

args1d = [
    (system, space)
    for system in ("clustering", "curve", "helix")
    for space in ("Legendre", "Chebyshev", "ChebyshevU", "Fourier")
    if not (space == "Fourier" and system != "clustering")
]


@pytest.mark.parametrize("system,space", args1d)
def test_forward_backward_1d_curvilinear(system, space) -> None:
    """`forward` inverts `backward` whatever the metric."""
    V = spaces1d[system](space)
    assert V.system.sg != 1, "this test is pointless without a metric weight"
    rand = jax.random.normal(jax.random.PRNGKey(101), shape=(V.num_quad_points,))
    uj = V.backward(V.forward(rand))  # land in the space first
    assert jnp.allclose(V.backward(V.forward(uj)), uj, rtol=ulp(1000), atol=ulp(1000))


@pytest.mark.parametrize("system,space", args1d)
def test_scalar_product_carries_the_measure(system, space) -> None:
    """`scalar_product(a)` is `inner(a*v)`, measure included.

    The integrators assemble mass and linear operators with `inner` but build
    their explicit terms from `scalar_product`, so the two halves of an
    equation only agree if both carry `sg`.
    """
    V = spaces1d[system](space)
    assert V.system.sg != 1, "this test is pointless without a metric weight"
    s = V.system.base_scalars()[0]
    ue = sp.cos(2 * s) + sp.sin(s)
    uj = jnp.asarray(lambdify(s, ue, modules="jax")(V.mesh()))
    got = V.scalar_product(uj)
    ref = inner(TestFunction(V) * ue)
    assert isinstance(ref, jnp.ndarray)
    assert jnp.allclose(got, ref, rtol=ulp(1000), atol=ulp(1000))


@pytest.mark.parametrize("system,space", args1d)
def test_project_kinds_differ_only_past_resolution(system, space) -> None:
    """The two `ProjectionKind`s are each exact, in different inner products.

    `INTERPOLATION` matches `ue` at the quadrature points -- testing with
    ``v/sg`` cancels the measure, which is what makes the cheap transform a
    projection in its own right. `L2` is orthogonal under ``sg*dxi`` instead.
    They coincide whenever the space can represent `ue`, and only part company
    once it cannot, which is the whole reason the distinction exists.
    """
    V = spaces1d[system](space)
    assert V.system.sg != 1, "this test is pointless without a metric weight"
    s = V.system.base_scalars()[0]
    u, v = TrialFunction(V), TestFunction(V)

    # INTERPOLATION is the discrete transform, exactly.
    # Analytic everywhere and periodic, so it suits the Fourier spaces too, but
    # with a complex pole close enough to the domain that none of these spaces
    # resolves it -- which is the only regime where the two kinds differ.
    ue = 1 / (sp.Rational(11, 10) - sp.cos(s))
    uj = jnp.asarray(lambdify(s, ue, modules="jax")(V.mesh()))
    interp = project(ue, V)
    assert jnp.allclose(interp, V.forward(uj), rtol=ulp(1000), atol=ulp(1000))

    # L2 agrees with a heavily over-integrated Galerkin solve. The loop stops
    # once refinement stops paying, so it is held to that bar, not to eps.
    l2 = project(ue, V, kind=ProjectionKind.L2)
    M, b = inner(v * (u - ue), kind="system", num_quad_points=32 * V.num_quad_points)
    settled = jnp.sqrt(jnp.finfo(jnp.result_type(l2)).eps)
    assert jnp.abs(l2 - M.solve(b)).max() <= settled * jnp.abs(l2).max()

    # Past resolution the two are genuinely different functions.
    assert not jnp.allclose(interp, l2, rtol=ulp(1000), atol=ulp(1000))

    # Within resolution they agree; a constant is in every space here.
    for kind in ("interpolation", "l2"):
        assert jnp.allclose(
            project(sp.Integer(1) + 0 * s, V, kind=kind),
            project(sp.Integer(1) + 0 * s, V),
            rtol=ulp(1000),
            atol=ulp(1000),
        )


@pytest.mark.parametrize("system,space", args)
def test_forward_backward(system, space) -> None:
    """The forward and backward transforms are inverses, for every system."""
    T = spaces[system](space)
    shape = T.num_dofs if isinstance(T, TensorProductSpace) else (T.num_dofs,)
    u_hat = jax.random.normal(jax.random.PRNGKey(101), shape=shape)
    u = T.backward(u_hat)
    uh = T.forward(u)
    assert jnp.allclose(u_hat, uh, rtol=ulp(1000), atol=ulp(1000))


def test_laplace():
    P = get_polar()
    theta, r = P.base_scalars()
    f = (r * theta) ** 2
    Lf = Div(Grad(f)).doit()
    assert Lf == 4 * theta**2 + 2

    P = get_cylindrical()
    theta, r, z = P.base_scalars()
    f = (r * theta * z) ** 2
    Lf = Div(Grad(f)).doit()
    assert Lf == 4 * theta**2 * z**2 + 2 * z**2 + 2 * r**2 * theta**2


def test_projection():
    P = get_polar()
    theta, r = P.base_scalars()
    rv = P.position_vector(True)
    rp = dot(rv, P.b_r) * P.b_r + dot(rv, P.b_theta) * P.b_theta
    assert isinstance(rp.args[1], BaseVector)
    assert rp.args[1]._latex_form == "\\mathbf{b_{r}}"
    rs = P.simplify(rp)
    assert isinstance(rs.args[1], BaseVector)
    assert rs.args[1]._latex_form == "\\mathbf{b_{r}}"


def test_base_time_id_is_per_system_but_time_is_one_symbol() -> None:
    """`BaseTime` must not share one cached object between coordinate systems.

    `Symbol.__new__` memoizes on the name, and every time symbol is named "t",
    so the cached constructor hands back a single object for every system.
    `_id` is assigned after construction, so the last system to ask would own
    it -- retroactively, for every holder of an earlier one. That is
    observable: `_id[0]` is time's position among a system's variables, which
    `jaxfun.pinns.loss` uses as an axis index into a Jacobian, so a 1-D system
    asking after a 2-D one would index the wrong axis.

    What must *not* change is equality. The time axis is the same one whatever
    system is used for the spatial dimensions, so the two symbols stay equal,
    hash alike, collapse in a set and substitute for one another. Only identity
    separates them.
    """
    import copy
    import pickle

    from jaxfun.coordinates import CartCoordSys, x, y

    A = CartCoordSys("A", (x,))
    B = CartCoordSys("B", (x, y))

    tA = A.base_time()
    tB = B.base_time()  # must not reach back and overwrite tA

    assert tA is not tB
    assert tA._id == (1,)
    assert tB._id == (2,)
    assert tA._system is A
    assert tB._system is B

    assert tA == tB
    assert hash(tA) == hash(tB)
    assert len({tA, tB}) == 1
    assert (tA**2).subs(tB, 3) == 9

    # Reconstruction carries the owning system, not whichever asked last.
    # `deepcopy` preserves identity; `pickle` rebuilds by value, so that one is
    # an equality check.
    assert copy.deepcopy(tA)._id == (1,)
    assert copy.deepcopy(tA)._system is A
    assert pickle.loads(pickle.dumps(tA))._system == A
    assert tB._id == (2,)
