import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    Legendre,
    TensorProduct,
)
from jaxfun.galerkin.inner import project
from jaxfun.utils.common import lambdify, ulp

pytestmark = pytest.mark.spmd


# ---------------------------------------------------------------------------
# 2-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space0, space1",
    [
        (Chebyshev.Chebyshev, Chebyshev.Chebyshev),
        (Legendre.Legendre, Legendre.Legendre),
        (Fourier.Fourier, Fourier.Fourier),
        (Fourier.Fourier, Chebyshev.Chebyshev),
        (Fourier.Fourier, Legendre.Legendre),
    ],
    ids=["ChexChe", "LegxLeg", "FxF", "FxChe", "FxLeg"],
)
def test_forward_backward_2d_spmd(space0, space1) -> None:
    T = TensorProduct(space0(16), space1(16))
    x, y = T.system.base_scalars()
    ue = sp.sin(x) * sp.sin(y)
    uh = project(ue, T)
    xj = jnp.array([[0.5, 0.5], [0.6, 0.6]])
    uj = T.evaluate(xj, uh)
    uej = lambdify((x, y), ue)(*xj.T)
    assert jnp.linalg.norm(uj - uej) < ulp(100)


# ---------------------------------------------------------------------------
# 3-D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space0, space1, space2",
    [
        (Chebyshev.Chebyshev, Chebyshev.Chebyshev, Chebyshev.Chebyshev),
        (Fourier.Fourier, Chebyshev.Chebyshev, Legendre.Legendre),
        (Fourier.Fourier, Fourier.Fourier, Legendre.Legendre),
    ],
    ids=["ChexChexChe", "FxChexLeg", "FxFxLeg"],
)
def test_evaluate_3d_spmd(space0, space1, space2) -> None:
    T = TensorProduct(space0(16), space1(16), space2(16))
    x, y, z = T.system.base_scalars()
    ue = sp.cos(x) * sp.sin(y) * sp.cos(z)
    uh = project(ue, T)
    xj = jnp.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    uj = T.evaluate(xj, uh)
    uej = lambdify((x, y, z), ue)(*xj.T)
    assert jnp.linalg.norm(uj - uej) < ulp(100)
