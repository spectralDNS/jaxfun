import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jax import Array
from scipy.special import eval_gegenbauer

from jaxfun.galerkin import FunctionSpace, TestFunction, TrialFunction, inner
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.composite import (
    Composite,
    DirectSum,
    GalerkinRecombined,
    Stencil,
)
from jaxfun.galerkin.Ultraspherical import Ultraspherical
from jaxfun.typing import UltrasphericalNormalization
from jaxfun.utils import ulp

NORMALIZATIONS = ("Gegenbauer", "Unity")


def _scale(n: np.ndarray, lam: float, normalization: str) -> np.ndarray:
    """Return the factors that take scipy's Gegenbauer polynomials to ours."""
    if normalization == "Unity":
        return 1 / eval_gegenbauer(n, lam, 1.0)
    return np.ones(len(n))


def _assert_close(a, b, tol: Array | None = None) -> None:
    tol = ulp(1000) if tol is None else tol
    b = np.asarray(b)
    assert np.abs(np.asarray(a) - b).max() < tol * np.abs(b).max()


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("lam", (sp.Rational(3, 2), 2))
def test_basis_values(lam, normalization: str) -> None:
    N = 8
    V = Ultraspherical(N, lambda_=lam, normalization=normalization)
    x = np.linspace(-1, 1, 5)
    n = np.arange(N)
    lam = float(lam)
    ref = eval_gegenbauer(n, lam, x[:, None]) * _scale(n, lam, normalization)
    _assert_close(V.eval_basis_functions(jnp.asarray(x)), ref)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_derivative_coeffs(normalization: str) -> None:
    """d/dx C^(lam)_n = 2 lam C^(lam+1)_{n-1}."""
    lam, N = 2, 12
    V = Ultraspherical(N, lambda_=lam, normalization=normalization)
    c = np.random.default_rng(1).standard_normal(N)
    x = np.linspace(-1, 1, 7)
    n = np.arange(1, N)
    dQ = 2 * lam * eval_gegenbauer(n - 1, lam + 1, x[:, None])
    ref = dQ @ (c[1:] * _scale(n, lam, normalization))
    dc = V.derivative_coeffs(jnp.asarray(c), 1)
    _assert_close(V.evaluate(jnp.asarray(x), dc), ref)


def test_gegenbauer_lambda_one_is_chebyshevU() -> None:
    N = 8
    V, U = Ultraspherical(N), ChebyshevU(N)
    assert V.norm_squared().shape == (N,)
    _assert_close(V.norm_squared(), U.norm_squared())
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    S_V = Composite(N, Ultraspherical, bcs).S.todense()
    _assert_close(S_V, Composite(N, ChebyshevU, bcs).S.todense())


def test_lambda_zero_is_chebyshev() -> None:
    V = Ultraspherical(8, lambda_=0)
    assert V.normalization == UltrasphericalNormalization.UNITY
    x = jnp.linspace(-1, 1, 5)
    _assert_close(V.eval_basis_functions(x), Chebyshev(8).eval_basis_functions(x))


@pytest.mark.parametrize(
    "trial", ({"lambda_": 2}, {"normalization": "Unity"}), ids=("lambda", "scaling")
)
def test_mass_matrix_of_different_spaces(trial: dict) -> None:
    N = 6
    V, W = Ultraspherical(N), Ultraspherical(N, **trial)
    A = inner(TestFunction(V) * TrialFunction(W), kind="bilinear")
    x, w = V.quad_points_and_weights(2 * N)
    ref = (V.eval_basis_functions(x) * w[:, None]).T @ W.eval_basis_functions(x)
    _assert_close(A.todense(), ref)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_composite_keeps_parameters(normalization: str) -> None:
    N = 12
    bcs = {"left": {"D": 1}, "right": {"D": 2}}
    D = FunctionSpace(
        N, Ultraspherical, bcs=bcs, lambda_=2, normalization=normalization
    )
    assert isinstance(D, DirectSum)
    U = D[0].orthogonal
    assert isinstance(U, Ultraspherical)
    assert U.lambda_ == 2
    assert U.normalization == normalization
    H = Composite(
        N, Ultraspherical, bcs, lambda_=2, normalization=normalization
    ).get_homogeneous()
    for space in (D[1], D[0].get_testspace(name="w"), H):
        assert space.orthogonal.parameters == U.parameters
    # The composite functions vanish at the boundary, where the lifting meets bcs.
    u = D.evaluate(jnp.array([-1.0, 1.0]), jnp.ones(D.num_dofs))
    _assert_close(u, jnp.array([1.0, 2.0]), tol=jnp.sqrt(ulp(1)))


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("k", (1, 2, 3))
def test_pg_stencil(k: int, normalization: str) -> None:
    """(psi_j, d^k C_l)_w = delta_{j, l-k} for the Petrov-Galerkin functions."""
    N = 10
    V = Ultraspherical(N + 2 * k, lambda_=2, normalization=normalization)
    S = Stencil(V.pg_stencil(k)).matrix((N - k, N + 2 * k)).todense()
    x, w = V.quad_points_and_weights()
    dC = V.evaluate_basis_derivative(x, k)[:, :N]
    G = S @ (V.eval_basis_functions(x) * w[:, None]).T @ dC
    _assert_close(G, jnp.eye(N - k, N, k))


@pytest.mark.parametrize(
    "trial", ({"lambda_": 2}, {"normalization": "Unity"}), ids=("lambda", "scaling")
)
def test_pg_with_other_trial_space(trial: dict) -> None:
    """The PG recurrences hold only for their own family; others take quadrature."""
    N = 6
    P = Ultraspherical(N).get_testspace("PG")
    form = TestFunction(P) * TrialFunction(Ultraspherical(N, **trial)).diff(P.system.x)
    A = inner(form, kind="bilinear")
    ref = inner(form, kind="bilinear", use_precomputed_matrices=False)
    _assert_close(A.todense(), ref.todense())


def test_galerkin_recombined() -> None:
    """The recurrence-built matrices of the GR test space, replaced rows too."""
    N = 8
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    B = FunctionSpace(N, Ultraspherical, bcs=bcs, lambda_=2)
    G = B.get_testspace("GR")
    assert isinstance(G, GalerkinRecombined)
    x = B.system.x
    form = TrialFunction(B).diff(x, 2) * TestFunction(G)
    A = inner(form, kind="bilinear")
    ref = inner(form, kind="bilinear", use_precomputed_matrices=False)
    _assert_close(A.todense(), ref.todense())
