import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.Chebyshev import Chebyshev
from jaxfun.composite import Composite
from jaxfun.inner import inner
from jaxfun.Legendre import Legendre
from jaxfun.utils.common import ulp


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner(space) -> None:
    N = 11
    V = space(N)
    x = V.system.x
    u = TrialFunction(V)
    v = TestFunction(V)
    M = inner(v * u, sparse=True, sparse_tol=1000)
    M0 = V.mass_matrix()
    assert jnp.allclose(M.data, M0.data)
    M = inner(x * v * u, sparse=True)
    a0 = answer1[space.__name__]
    assert jnp.allclose(M.todense().diagonal(1), a0)
    assert jnp.allclose(M.todense().diagonal(-1), a0)
    M = inner(x * v * u + sp.diff(u, x) * v, sparse=True)

    N = 11
    C = Composite(N, space, {"left": {"D": 0}, "right": {"D": 0}})
    u = TrialFunction(C)
    v = TestFunction(C)
    M = inner(x * v * u + sp.diff(u, x) * v, sparse=False)
    a0 = answer2[space.__name__]
    for d in (-3, -1, 1, 3):
        assert jnp.allclose(M.diagonal(d), a0[d], atol=1000 * ulp(1.0))


# Some data from shenfun
answer1 = {
    "Chebyshev": jnp.array(
        [
            1.570796326794896,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
            0.785398163397448,
        ]
    ),
    "Legendre": jnp.array(
        [
            0.6666666666666666,
            0.2666666666666666,
            0.1714285714285713,
            0.126984126984127,
            0.101010101010101,
            0.0839160839160839,
            0.0717948717948718,
            0.0627450980392157,
            0.0557275541795665,
            0.050125313283208,
        ]
    ),
}

answer2 = {
    "Chebyshev": {
        -3: jnp.array(
            [
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
            ]
        ),
        -1: jnp.array(
            [
                -4.71238898038469,
                -8.63937979737193,
                -11.780972450961723,
                -14.922565104551516,
                -18.06415775814131,
                -21.205750411731103,
                -24.347343065320896,
                -27.48893571891069,
            ]
        ),
        1: jnp.array(
            [
                4.71238898038469,
                7.0685834705770345,
                10.210176124166829,
                13.351768777756622,
                16.493361431346415,
                19.634954084936208,
                22.776546738526,
                25.918139392115794,
            ]
        ),
        3: jnp.array(
            [
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
                -0.7853981633974483,
            ]
        ),
    },
    "Legendre": {
        -3: jnp.array(
            [
                -0.1714285714285714,
                -0.126984126984127,
                -0.101010101010101,
                -0.0839160839160839,
                -0.0717948717948718,
                -0.0627450980392157,
            ]
        ),
        -1: jnp.array(
            [
                -1.4285714285714286,
                -1.7777777777777777,
                -1.8545454545454545,
                -1.89010989010989,
                -1.9111111111111112,
                -1.9251336898395721,
                -1.9352226720647774,
                -1.9428571428571428,
            ]
        ),
        1: jnp.array(
            [
                2.571428571428571,
                2.2222222222222223,
                2.1454545454545455,
                2.10989010989011,
                2.088888888888889,
                2.0748663101604277,
                2.064777327935223,
                2.057142857142857,
            ]
        ),
        3: jnp.array(
            [
                -0.1714285714285714,
                -0.126984126984127,
                -0.101010101010101,
                -0.0839160839160839,
                -0.0717948717948718,
                -0.0627450980392157,
            ]
        ),
    },
}

if __name__ == "__main__":
    test_inner(Chebyshev)
    test_inner(Legendre)
