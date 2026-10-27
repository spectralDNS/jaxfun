import numpy as np
import pytest
import sympy as sp

from jaxfun.coordinates import CoordSys, MetricComponent, get_CoordSys
from jaxfun.utils.common import lambdify

xi, eta = sp.symbols("xi,eta", real=True)


def _quadrilateral(name: str, symbolic_metric: bool) -> CoordSys:
    """Bilinear map of [-1, 1]^2 onto a quadrilateral: a non-orthogonal system."""
    return get_CoordSys(
        name,
        sp.Lambda((xi, eta), ((1 + xi) * (7 - eta) / 4, (1 + eta) * (5 - xi) / 4)),
        assumptions=sp.Q.positive(xi + 1)
        & sp.Q.positive(1 - xi)
        & sp.Q.positive(eta + 1)
        & sp.Q.positive(1 - eta),
        symbolic_metric=symbolic_metric,
    )


@pytest.fixture(scope="module")
def systems() -> tuple[CoordSys, CoordSys]:
    # Different names: the flag is not part of a system's identity.
    return _quadrilateral("QE", False), _quadrilateral("QS", True)


def _explicit(E: CoordSys, S: CoordSys, expr: sp.Expr) -> sp.Expr:
    """Return S's expression with the metric made explicit, in E's coordinates."""
    return S.expand_metric(expr).xreplace(dict(zip(S.base_scalars(), E.base_scalars())))


def test_determinant_is_a_positive_symbol(systems) -> None:
    E, S = systems
    g = S.get_det_g()
    assert S.symbolic_metric and not E.symbolic_metric
    assert S.sg == sp.sqrt(g)
    assert S.sg**2 == g
    assert S.sg**3 / g == S.sg
    assert sp.expand(_explicit(E, S, S.sg) - E.sg) == 0
    assert sp.expand(_explicit(E, S, g) - E.get_det_g()) == 0


def test_metric_components_are_symbols(systems) -> None:
    E, S = systems
    G, Ge = S.get_covariant_metric_tensor(), E.get_covariant_metric_tensor()
    assert all(isinstance(c, MetricComponent) for c in G.flat)
    assert G[0, 1] is G[1, 0]
    for c, ce in zip(G.flat, Ge.flat, strict=True):
        assert sp.expand(_explicit(E, S, c) - ce) == 0
    # The inverse is the adjugate over g, in the symbols.
    g = S.get_det_g()
    expected = [[G[1, 1] / g, -G[0, 1] / g], [-G[0, 1] / g, G[0, 0] / g]]
    assert S.get_contravariant_metric_tensor().tolist() == expected


def test_vanishing_components_stay_zero() -> None:
    r, theta = sp.symbols("r,theta", real=True, positive=True)
    P = get_CoordSys(
        "PS",
        sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta))),
        symbolic_metric=True,
    )
    G = P.get_covariant_metric_tensor()
    assert G[0, 1] == 0 and isinstance(G[1, 1], MetricComponent)
    assert P.get_contravariant_metric_tensor()[0, 1] == 0
    assert P.expand_metric(P.sg) == P.base_scalars()[0]


def test_metric_and_christoffel_symbols_match_explicit(systems) -> None:
    E, S = systems
    pairs = (
        (S.get_contravariant_metric_tensor(), E.get_contravariant_metric_tensor()),
        (S.get_christoffel_second(), E.get_christoffel_second()),
    )
    for symbolic, explicit in pairs:
        for a, b in zip(np.ravel(symbolic), np.ravel(explicit), strict=True):
            # sp.cancel can leave equal base-scalar terms uncombined, so cancel
            # in the plain symbols, as CoordSys.simplify does.
            d = E.expr_base_scalar_to_psi(_explicit(E, S, a) - b)
            assert E.expr_psi_to_base_scalar(sp.cancel(d)) == 0


def test_derivatives_are_those_of_the_explicit_expressions(systems) -> None:
    E, S = systems
    symbols = (S.get_det_g(), *S.get_covariant_metric_tensor().flat)
    explicit = (E.get_det_g(), *E.get_covariant_metric_tensor().flat)
    for qs, qe in zip(S.base_scalars(), E.base_scalars(), strict=True):
        for a, b in zip(symbols, explicit, strict=True):
            assert sp.expand(_explicit(E, S, a.diff(qs)) - b.diff(qe)) == 0


def test_lambdify_evaluates_the_explicit_expressions(systems) -> None:
    E, S = systems
    x, y = np.array([0.2, -0.5]), np.array([-0.3, 0.7])
    for a, b in (
        (S.sg, E.sg),
        (S.get_covariant_metric_tensor()[0, 1], E.get_covariant_metric_tensor()[0, 1]),
    ):
        expected = lambdify(E.base_scalars(), b)(x, y)
        assert np.allclose(lambdify(S.base_scalars(), a)(x, y), expected)
