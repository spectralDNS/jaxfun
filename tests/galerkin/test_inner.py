from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp
from jax.experimental.sparse import BCOO
from scipy.integrate import dblquad

from jaxfun.galerkin import (
    Composite,
    FunctionSpace,
    JAXFunction,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.utils import ulp


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner(space: type[Legendre] | type[Chebyshev]) -> None:
    N = 11
    V = space(N)
    x = V.system.x
    u = TrialFunction(V)
    v = TestFunction(V)
    M = inner(v * u, sparse=True, sparse_tol=1000)
    assert isinstance(M, BCOO)
    M0 = V.mass_matrix()
    assert jnp.allclose(M.data, M0.data)
    M = inner(x * v * u, sparse=True)
    assert isinstance(M, BCOO)
    a0 = answer1[space.__name__]
    assert jnp.allclose(M.todense().diagonal(1), a0)
    assert jnp.allclose(M.todense().diagonal(-1), a0)
    M = inner(x * v * u + sp.diff(u, x) * v, sparse=True)
    assert isinstance(M, BCOO)

    N = 11
    C = Composite(N, space, {"left": {"D": 0}, "right": {"D": 0}})
    u = TrialFunction(C)
    v = TestFunction(C)
    M = inner(x * v * u + sp.diff(u, x) * v, sparse=False)
    assert hasattr(M, "diagonal")
    M_arr = cast(jax.Array, M)
    a0 = answer2[space.__name__]
    for d in (-3, -1, 1, 3):
        assert jnp.allclose(M_arr.diagonal(d), a0[d], atol=1000 * ulp(1.0))


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


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_linear_inner(space):
    N = 11
    V = space(N)
    x = V.system.x
    v = TestFunction(V)
    a = JAXFunction(sp.sin(x), V, name="a")
    f = [sp.sin(x), a]
    l0 = []
    for fi in f:
        l0.append(inner(v * fi))
    assert jnp.allclose(l0[0], l0[1], atol=ulp(100))

    l1 = []
    for fi in f:
        l1.append(inner(v.diff(x, 1) * fi))
    assert jnp.allclose(l1[0], l1[1], atol=ulp(100))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_linear_inner_2d(space):
    N = 11
    V = space(N)
    T = TensorProduct(V, V)
    x, y = T.system.base_scalars()
    v = TestFunction(T)
    a = JAXFunction(sp.sin(x) * sp.cos(y), T, name="a")
    xj, yj = T.mesh()
    f = [sp.sin(x) * sp.cos(y), a]
    l0 = []
    for fi in f:
        l0.append(inner(v * fi))
    assert jnp.allclose(l0[0], l0[1], atol=ulp(100)), jnp.max(jnp.abs(l0[0] - l0[1]))

    l1 = []
    for fi in f:
        l1.append(inner(v.diff(x, 1) * fi))
    assert jnp.allclose(l1[0], l1[1], atol=ulp(10000)), jnp.max(jnp.abs(l1[0] - l1[1]))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_linear_inner_3d(space):
    N = 11
    V = space(N)
    T = TensorProduct(V, V, V)
    x, y, z = T.system.base_scalars()
    v = TestFunction(T)
    a = JAXFunction(sp.sin(x) * sp.cos(y) * sp.sin(z), T, name="a")
    xj, yj, zj = T.mesh()
    f = [sp.sin(x) * sp.cos(y) * sp.sin(z), a]
    l0 = []
    for fi in f:
        l0.append(inner(v * fi))
    assert jnp.allclose(l0[0], l0[1], atol=ulp(100)), jnp.max(jnp.abs(l0[0] - l0[1]))

    l1 = []
    for fi in f:
        l1.append(inner(v.diff(x, 1) * fi))
    assert jnp.allclose(l1[0], l1[1], atol=ulp(10000)), jnp.max(jnp.abs(l1[0] - l1[1]))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_nonlinear_inner_3d(space):
    N = 11
    V = space(N)
    T = TensorProduct(V, V, V)
    x, y, z = T.system.base_scalars()
    v = TestFunction(T)
    a = JAXFunction(sp.sin(x) * sp.cos(y) * sp.sin(z), T, name="a")
    xj, yj, zj = T.mesh()
    f = [(sp.sin(x) * sp.cos(y) * sp.sin(z)) ** 2, a**2]  # nonlinear jaxfunction
    l0 = []
    for fi in f:
        l0.append(inner(v * fi))
    assert jnp.allclose(l0[0], l0[1], atol=ulp(100))

    l1 = []
    for fi in f:
        l1.append(inner(v.diff(y, 1) * fi))
    assert jnp.allclose(l1[0], l1[1], atol=ulp(1000))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_bilinear_inner(space):
    N = 11
    V = space(N)
    x = V.system.x
    v = TestFunction(V)
    u = TrialFunction(V)
    a = JAXFunction(sp.sin(x), V, name="a")
    f = [sp.sin(x), a]
    l0 = []
    for fi in f:
        l0.append(inner(v * u * fi))

    assert jnp.allclose(l0[0], l0[1], atol=ulp(100))

    l1 = []
    for fi in f:
        l1.append(inner(v.diff(x, 1) * u * fi))
    assert jnp.allclose(l1[0], l1[1], atol=ulp(1000)), jnp.max(jnp.abs(l1[0] - l1[1]))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_resolved(
    space: type[Legendre | Chebyshev],
) -> None:
    D = space(36)
    x = D.system.x
    uf = JAXFunction(sp.sin(x * 2 * sp.pi), D)
    v = TestFunction(D)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=48)
    # Since the function is well-resolved with 36 modes, we expect no
    # aliasing and thus the same result with or without padding
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = D.forward(uf.backward() ** 2)
    y3 = D.forward(uf.backward(N=48) ** 2)
    assert jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_not_resolved(
    space: type[Legendre | Chebyshev],
) -> None:
    D = space(8)
    x = D.system.x
    uf = JAXFunction(sp.sin(x * 2 * sp.pi), D)
    v = TestFunction(D)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=12)
    # Since the function is not well-resolved with 8 modes, we expect
    # aliasing and thus different results with or without padding
    assert not jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = D.forward(uf.backward() ** 2)
    y3 = D.forward(uf.backward(N=12) ** 2)
    assert not jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))
    # No further change with more padding
    y4 = D.forward(uf.backward(N=16) ** 2)
    assert jnp.linalg.norm(y3 - y4) < jnp.sqrt(ulp(100))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_resolved_2d(
    space: type[Legendre | Chebyshev],
) -> None:
    D = space(36)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    f = sp.sin(x * 2 * sp.pi) * sp.sin(y * 2 * sp.pi)
    uf = JAXFunction(f, T)
    v = TestFunction(T)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=(48, 48))
    # Since the function is well-resolved with 36 modes, we expect no
    # aliasing and thus the same result with or without padding
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = T.forward(uf.backward() ** 2)
    y3 = T.forward(uf.backward(N=(48, 48)) ** 2)
    assert jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_not_resolved_2d(
    space: type[Legendre | Chebyshev],
) -> None:
    D = space(8)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    f = sp.sin(x * 2 * sp.pi) * sp.sin(y * 2 * sp.pi)
    uf = JAXFunction(f, T)
    v = TestFunction(T)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=(12, 12))
    # Since the function is not well-resolved with 8 modes, we expect
    # aliasing and thus different results with or without padding
    assert not jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = T.forward(uf.backward() ** 2)
    y3 = T.forward(uf.backward(N=(12, 12)) ** 2)
    assert not jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))
    # No further change with more padding
    y4 = T.forward(uf.backward(N=(16, 16)) ** 2)
    assert jnp.linalg.norm(y3 - y4) < jnp.sqrt(ulp(100))


def test_inner_padding_Fourier() -> None:
    F = Fourier(6)
    x = F.system.x
    uf = JAXFunction(sp.exp(4j * x), F)
    v = TestFunction(F)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=9)
    # uf**2 has wavenumber 8, which is above the Nyquist wavenumber of 6,
    # thus aliased to wavenumber 2. With padding, no aliasing occurs and
    # we correctly get zero when truncating to 6 wavenumbers.
    assert jnp.linalg.norm(y1) < jnp.sqrt(ulp(100))  # No aliasing - thus zero
    assert abs(y0[2] - 2 * jnp.pi) < jnp.sqrt(ulp(100))  # Aliasing error
    y2 = F.forward(uf.backward() ** 2)
    y3 = F.forward(uf.backward(N=9) ** 2)
    assert jnp.linalg.norm(y3) < jnp.sqrt(ulp(100))  # No aliasing - thus zero
    assert abs(y2[2] - 1) < jnp.sqrt(ulp(100))  # Aliasing error


def test_inner_padding_Fourier_2d() -> None:
    F = Fourier(6)
    T = TensorProduct(F, F)
    x, y = T.system.base_scalars()
    uf = JAXFunction(sp.exp(4j * x) * sp.exp(4j * y), T)
    v = TestFunction(T)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=(9, 9))
    assert jnp.linalg.norm(y1) < jnp.sqrt(ulp(100))  # No aliasing - thus zero
    assert abs(y0[2, 2] - (2 * jnp.pi) ** 2) < jnp.sqrt(ulp(100))  # Aliasing error
    y2 = T.forward(uf.backward() ** 2)
    y3 = T.forward(uf.backward(N=(9, 9)) ** 2)
    assert jnp.linalg.norm(y3) < jnp.sqrt(ulp(100))  # No aliasing - thus zero
    assert abs(y2[2, 2] - 1) < jnp.sqrt(ulp(100))  # Aliasing error


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_composite_resolved(
    space: type[Legendre | Chebyshev],
) -> None:
    bcs = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(36, space, bcs=bcs)
    x = D.system.x
    uf = JAXFunction(sp.sin(x * 2 * sp.pi), D)
    v = TestFunction(D)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=48)
    # Since the function is well-resolved with 36 modes, we expect no
    # aliasing and thus the same result with or without padding
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = D.forward(uf.backward() ** 2)
    y3 = D.forward(uf.backward(N=48) ** 2)
    assert jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_directsum_resolved(
    space: type[Legendre | Chebyshev],
) -> None:
    bcs = {"left": {"D": 1}, "right": {"D": 1}}
    D = FunctionSpace(36, space, bcs=bcs)
    x = D.system.x
    uf = JAXFunction(sp.cos(x * 2 * sp.pi), D)
    v = TestFunction(D)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=48)
    # Since the function is well-resolved with 36 modes, we expect no
    # aliasing and thus the same result with or without padding
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = D.forward(uf.backward() ** 2)
    y3 = D.forward(uf.backward(N=48) ** 2)
    assert jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))


@pytest.mark.parametrize("space", (Legendre, Chebyshev))
def test_inner_padding_directsumtps_resolved_2d(
    space: type[Legendre | Chebyshev],
) -> None:
    from jaxfun.coordinates import x, y

    f = sp.cos(x * 2 * sp.pi) * sp.cos(y * 2 * sp.pi)
    bcsx = {"left": {"D": f.subs(x, -1)}, "right": {"D": f.subs(x, 1)}}
    bcsy = {"left": {"D": f.subs(y, -1)}, "right": {"D": f.subs(y, 1)}}
    Dx = FunctionSpace(36, space, bcs=bcsx, name="Dx")
    Dy = FunctionSpace(36, space, bcs=bcsy, name="Dy")
    T = TensorProduct(Dx, Dy, name="T")
    x, y = T.system.base_scalars()
    f = T.system.expr_psi_to_base_scalar(f)
    uf = JAXFunction(f, T)
    v = TestFunction(T)
    y0 = inner(v * uf**2, num_quad_points=None)
    y1 = inner(v * uf**2, num_quad_points=(48, 48))
    # Since the function is well-resolved with 36 modes, we expect no
    # aliasing and thus the same result with or without padding
    assert jnp.linalg.norm(y0 - y1) < jnp.sqrt(ulp(100))
    y2 = T.forward(uf.backward() ** 2)
    y3 = T.forward(uf.backward(N=(48, 48)) ** 2)
    assert jnp.linalg.norm(y2 - y3) < jnp.sqrt(ulp(100))


def test_inner_exact_poly():
    N = 6
    L = Legendre(N, name="L")
    u = TrialFunction(L)
    v = TestFunction(L)
    x = L.system.x
    t = sp.integrate(
        x**2 * sp.legendre(N - 1, x) * sp.legendre(N - 1, x), (x, -1, 1)
    ).n()

    # Check that without padding we get wrong result
    A = inner(x**2 * v * u)
    assert not jnp.allclose(A[N - 1, N - 1], float(t), atol=jnp.sqrt(ulp(100)))
    # Check that with padding we get correct result
    A = inner(x**2 * v * u, num_quad_points=7)
    assert jnp.allclose(A[N - 1, N - 1], float(t), atol=jnp.sqrt(ulp(100)))


def test_inner_exact_poly_2d():
    N = 6
    L = Legendre(N, name="L")
    T = TensorProduct(L, L)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    x, y = T.system.base_scalars()

    t = sp.integrate(
        x**2 * sp.legendre(N - 1, x) * sp.legendre(N - 1, x), (x, -1, 1)
    ).n()

    # Check that without padding we get wrong result
    B = inner((x**2 + y**2) * v * u)
    A = B[0].mats[0]
    assert not jnp.allclose(A[N - 1, N - 1], float(t), atol=jnp.sqrt(ulp(100)))
    A = B[1].mats[1]
    assert not jnp.allclose(A[N - 1, N - 1], float(t), atol=jnp.sqrt(ulp(100)))
    # Check that with padding we get correct result
    B = inner((x**2 + y**2) * v * u, num_quad_points=(7, 7))
    A = B[0].mats[0]
    assert jnp.allclose(A[N - 1, N - 1], float(t), atol=jnp.sqrt(ulp(100)))
    A = B[1].mats[1]
    assert jnp.allclose(A[N - 1, N - 1], float(t), atol=jnp.sqrt(ulp(100)))


def test_inner_exact_jaxfunction_2d():
    N = 6
    L = Legendre(N, name="L")
    T = TensorProduct(L, L)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)
    x, y = T.system.base_scalars()

    t = sp.integrate(
        (x**2 + y**2)
        * sp.legendre(N - 1, x)
        * sp.legendre(N - 1, x)
        * sp.legendre(N - 1, y)
        * sp.legendre(N - 1, y),
        (x, -1, 1),
        (y, -1, 1),
    ).n()

    uf = JAXFunction(x**2 + y**2, T)

    # Check that without padding we get wrong result
    B = inner(uf * v * u)
    A = B[0].mat
    assert not jnp.allclose(A[-1, -1, -1, -1], float(t), atol=jnp.sqrt(ulp(100)))
    # Check that with padding we get correct result
    B = inner(uf * v * u, num_quad_points=(7, 7))
    A = B[0].mat
    assert jnp.allclose(A[-1, -1, -1, -1], float(t), atol=jnp.sqrt(ulp(100)))


def test_inner_exact_multivar_2d():
    N = 6
    L = Legendre(N, name="L")
    T = TensorProduct(L, L)
    x, y = T.system.base_scalars()
    u = TrialFunction(T)
    v = TestFunction(T)

    x, y = sp.symbols("x,y", real=True)
    t, err = dblquad(
        sp.lambdify(
            (x, y),
            sp.sqrt(x**2 + y**2)
            * sp.legendre(N - 1, x) ** 2
            * sp.legendre(N - 1, y) ** 2,
        ),
        -1,
        1,
        -1,
        1,
        epsabs=ulp(100),
    )

    x, y = T.system.base_scalars()

    # Check that without padding we get wrong result
    B = inner(sp.sqrt(x**2 + y**2) * v * u)
    A = B[0].mat
    assert not jnp.allclose(A[-1, -1, -1, -1], float(t), atol=jnp.sqrt(ulp(100)))
    # Check that with padding we can get correct result (much padding since not
    # polynomial coefficient)
    B = inner(sp.sqrt(x**2 + y**2) * v * u, num_quad_points=(32, 32))
    A = B[0].mat
    assert jnp.allclose(A[-1, -1, -1, -1], float(t), atol=jnp.sqrt(ulp(100))), abs(
        A[-1, -1, -1, -1] - float(t)
    )
