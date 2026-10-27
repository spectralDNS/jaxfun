from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.galerkin import Jacobi, Legendre
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Jacobi import _tridiagonal_functions
from jaxfun.galerkin.Ultraspherical import Ultraspherical
from jaxfun.utils.common import Domain, ulp


def test_legendre_short_series_and_matrices_branches():
    # Trigger nd==1 and nd==2 branches in evaluate
    L = Legendre.Legendre(3)
    c1 = jnp.array([2.0])
    v1 = L.evaluate(0.25, c1)
    assert v1 == c1[0]
    c2 = jnp.array([1.0, 0.5])
    v2 = L.evaluate(0.3, c2)
    expected = c2[0] + c2[1] * 0.3
    assert jnp.isclose(v2, expected, atol=ulp(2.0))
    # matrices lookup branches

    _ = L.matrices(0, (L, 0))
    _ = L.matrices(0, (L, 1))
    _ = L.matrices(1, (L, 0))
    _ = L.matrices(0, (L, 2))
    _ = L.matrices(2, (L, 0))
    assert L.matrices(3, (L, 3)) is None


def test_jacobi_edge_matrices_and_evaluate_short():
    J = Jacobi.Jacobi(3, alpha=1, beta=0, domain=Domain(-1, 1))
    # Short series evaluate path len=1,2
    c1 = jnp.array([1.0])
    _ = J.evaluate(0.2, c1)
    c2 = jnp.array([1.0, 0.3])
    _ = J.evaluate(0.4, c2)
    # Longer series evaluate for stability
    e1 = J.evaluate(0.4, jnp.array([1.0, 0.3, 0.2]))
    assert jnp.isfinite(e1)


@pytest.mark.parametrize(
    "space",
    [
        lambda: Jacobi.Jacobi(4, alpha=sp.Rational(1, 3), beta=sp.Rational(-1, 4)),
        lambda: Chebyshev(4),
        lambda: Ultraspherical(4, lambda_=sp.Rational(3, 2), normalization="Unity"),
    ],
)
def test_recurrence_matrices_match_symbolic_entries(space):
    """The lambdified diagonals equal the recurrences evaluated at integers,
    wherever those are finite (some vanish as 0/0 at n = 0)."""
    S = space()
    N = S.N
    for M, mat in (
        (S.A(N), S.a),
        (S.B(N), S.b),
        (S.A_(N, k=2), partial(S.a_, k=2)),
    ):
        D = np.asarray(M.todense())
        for i in range(N):
            for j in range(max(0, i - 1), min(N, i + 2)):
                e = sp.sympify(mat(sp.Integer(i), sp.Integer(j)))
                if e.is_finite:
                    assert abs(D[i, j] - float(e)) <= ulp(10), (i, j)


def test_recurrence_diagonals_built_once_per_family_and_parameters():
    L4, L9 = Legendre.Legendre(4), Legendre.Legendre(9)
    assert _tridiagonal_functions(L4, L4.a) is _tridiagonal_functions(L9, L9.a)
    J = Jacobi.Jacobi(4, alpha=1, beta=0)
    assert _tridiagonal_functions(J, J.a) is not _tridiagonal_functions(L4, L4.a)
    a1 = _tridiagonal_functions(L4, partial(L4.a_, k=1))
    assert a1 is not _tridiagonal_functions(L4, partial(L4.a_, k=2))
