import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin import (
    Chebyshev,
    Fourier,
    FunctionSpace,
    Jacobi,
    Legendre,
    TensorProduct,
)
from jaxfun.galerkin.inner import project, project1D
from jaxfun.utils.common import ulp


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_forward_backward(
    space: Legendre.Legendre | Chebyshev.Chebyshev | Fourier.Fourier | Chebyshev.Jacobi,
) -> None:
    D = space(8)
    x = D.system.x
    ue = project1D(sp.sin(x), D)
    uj = D.backward(ue)
    uh = D.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Jacobi.Jacobi)
)
def test_forward_backward_composite(
    space: Legendre.Legendre | Chebyshev.Chebyshev | Chebyshev.Jacobi,
) -> None:
    D = FunctionSpace(8, space, bcs={"left": {"D": 0}, "right": {"D": 0}})
    x = D.system.x
    ue = project1D(sp.sin(x * 2 * sp.pi), D)
    uj = D.backward(ue)
    uh = D.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize(
    "space", (Legendre.Legendre, Chebyshev.Chebyshev, Fourier.Fourier, Jacobi.Jacobi)
)
def test_forward_backward_2d(
    space: Legendre.Legendre | Chebyshev.Chebyshev | Fourier.Fourier | Chebyshev.Jacobi,
) -> None:
    D = space(8)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    ue = project(sp.sin(x) * sp.sin(y), T)
    uj = T.backward(ue)
    uh = T.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


if __name__ == "__main__":
    test_forward_backward_2d(Fourier.Fourier)
