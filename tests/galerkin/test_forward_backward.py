import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jaxfun.coordinates import get_CoordSys
from jaxfun.galerkin import (
    Chebyshev,
    ChebyshevU,
    Fourier,
    FunctionSpace,
    Jacobi,
    Legendre,
    TensorProduct,
    Ultraspherical,
)
from jaxfun.galerkin.composite import Composite
from jaxfun.galerkin.inner import project, project1D
from jaxfun.utils.common import Domain, ulp


@pytest.mark.parametrize(
    "space",
    (
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Fourier.Fourier,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
)
def test_forward_backward(
    space: type[
        Legendre.Legendre
        | Chebyshev.Chebyshev
        | ChebyshevU.ChebyshevU
        | Fourier.Fourier
        | Jacobi.Jacobi
        | Ultraspherical.Ultraspherical
    ],
) -> None:
    D = space(8)
    x = D.system.x
    ue = project1D(sp.sin(x), D)
    uj = D.backward(ue)
    uh = D.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize(
    "space",
    (
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
)
def test_forward_backward_composite(
    space: type[
        Legendre.Legendre
        | Chebyshev.Chebyshev
        | ChebyshevU.ChebyshevU
        | Jacobi.Jacobi
        | Ultraspherical.Ultraspherical
    ],
) -> None:
    D = FunctionSpace(8, space, bcs={"left": {"D": 0}, "right": {"D": 0}})
    assert isinstance(D, Composite)
    x = D.system.x
    ue = project1D(sp.sin(x * 2 * sp.pi), D)
    uj = D.backward(ue)
    uh = D.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize(
    "space",
    (
        Legendre.Legendre,
        Chebyshev.Chebyshev,
        ChebyshevU.ChebyshevU,
        Fourier.Fourier,
        Jacobi.Jacobi,
        Ultraspherical.Ultraspherical,
    ),
)
def test_forward_backward_2d(
    space: type[
        Legendre.Legendre
        | Chebyshev.Chebyshev
        | ChebyshevU.ChebyshevU
        | Fourier.Fourier
        | Jacobi.Jacobi
        | Ultraspherical.Ultraspherical
    ],
) -> None:
    D = space(8)
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    ue = project(sp.sin(x) * sp.sin(y), T)
    uj = T.backward(ue)
    uh = T.forward(uj)
    assert jnp.linalg.norm(uh - ue) < ulp(100)


@pytest.mark.parametrize("pad", (None, (12, 8)))
def test_backward_batch_matches_per_field(pad: tuple[int, int] | None) -> None:
    """A leading batch axis must transform every field exactly as a lone call."""
    F = FunctionSpace(8, Fourier.Fourier, domain=Domain(0, 2 * sp.pi), name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D)
    x, y = T.system.base_scalars()
    fields = [
        project(sp.sin(x) * (1 - y**2), T),
        project(sp.cos(2 * x) * y * (1 - y**2), T),
        project(sp.sin(3 * x) * (1 - y**2) ** 2, T),
    ]
    batched = T.backward_batch(jnp.stack(fields), N=pad)
    assert batched.shape[0] == len(fields)
    for field, got in zip(fields, batched, strict=True):
        # Exactly equal, not merely close: vmap only changes how the same
        # arithmetic is issued.
        assert jnp.array_equal(got, T.backward(field, N=pad))


def test_backward_batch_3d() -> None:
    """Batching must work for a 3-D space, not just the 2-D case it was built for."""
    D = Legendre.Legendre(6)
    T = TensorProduct(D, D, D)
    x, y, z = T.system.base_scalars()
    fields = [project(sp.sin(x) * y * z, T), project(x * sp.cos(y) * z, T)]
    batched = T.backward_batch(jnp.stack(fields))
    for field, got in zip(fields, batched, strict=True):
        assert jnp.array_equal(got, T.backward(field))


@pytest.mark.parametrize("N", (None, (12, None), (None, 12), (None, None)))
def test_partial_quad_counts(N: tuple[int | None, int | None] | None) -> None:
    """A None entry means that axis's default, and must resolve to exactly that.

    The whole argument may be None, or any single entry of it; both stand for
    the same fully-specified tuple, so the transforms must agree with it.
    """
    F = FunctionSpace(8, Fourier.Fourier, domain=Domain(0, 2 * sp.pi), name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D)
    x, y = T.system.base_scalars()
    c = project(sp.sin(x) * (1 - y**2), T)
    default = tuple(s.num_quad_points for s in T.basespaces)
    full = (
        default
        if N is None
        else tuple(default[ax] if N[ax] is None else N[ax] for ax in range(2))
    )
    assert jnp.array_equal(T.backward(c, N=N), T.backward(c, N=full))
    assert jnp.array_equal(
        T.backward_primitive(c, k=(1, 0), N=N),
        T.backward_primitive(c, k=(1, 0), N=full),
    )
    assert jnp.array_equal(T.evaluate_mesh(c, N=N), T.evaluate_mesh(c, N=full))


@pytest.mark.parametrize("method", ("forward", "scalar_product"))
def test_forward_scalar_product_batch(method: str) -> None:
    """The physical-space transforms batch the same way, with the same result."""
    F = FunctionSpace(8, Fourier.Fourier, domain=Domain(0, 2 * sp.pi), name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D)
    x, y = T.system.base_scalars()
    fields = [
        T.backward(project(sp.sin(x) * (1 - y**2), T)),
        T.backward(project(sp.cos(2 * x) * y * (1 - y**2), T)),
        T.backward(project(sp.sin(3 * x) * (1 - y**2) ** 2, T)),
    ]
    one, many = getattr(T, method), getattr(T, method + "_batch")
    batched = many(jnp.stack(fields))
    assert batched.shape[0] == len(fields)
    for field, got in zip(fields, batched, strict=True):
        assert jnp.array_equal(got, one(field))


@pytest.mark.parametrize("k", ((0, 0), (1, 0), (0, 1), (2, 1)))
def test_backward_primitive_batch(k: tuple[int, int]) -> None:
    """Derivatives batch too, for every order the shared `k` can take."""
    F = FunctionSpace(8, Fourier.Fourier, domain=Domain(0, 2 * sp.pi), name="F")
    D = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(F, D)
    x, y = T.system.base_scalars()
    fields = [
        project(sp.sin(x) * (1 - y**2), T),
        project(sp.cos(2 * x) * y * (1 - y**2), T),
    ]
    batched = T.backward_primitive_batch(jnp.stack(fields), k=k, N=(12, 8))
    assert batched.shape[0] == len(fields)
    for field, got in zip(fields, batched, strict=True):
        assert jnp.array_equal(got, T.backward_primitive(field, k=k, N=(12, 8)))


def test_backward_primitive_batch_direct_sum() -> None:
    """The lifting has to reach the derivative too, batched or not."""
    F = FunctionSpace(8, Fourier.Fourier, domain=Domain(0, 2 * sp.pi), name="F")
    Tb = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 0}})
    VT = TensorProduct(F, Tb)
    x, y = VT.system.base_scalars()
    fields = [project(sp.sin(x) * (1 - y**2), VT), project(sp.cos(x) * y, VT)]
    batched = VT.backward_primitive_batch(jnp.stack(fields), k=(1, 1), N=(12, None))
    for field, got in zip(fields, batched, strict=True):
        assert jnp.array_equal(
            got, VT.backward_primitive(field, k=(1, 1), N=(12, None))
        )


def test_scalar_product_batch_keeps_metric() -> None:
    """A curvilinear system weights by sqrt(g); batching must not lose it."""
    r, theta = sp.symbols("r,theta", real=True, positive=True)
    polar = get_CoordSys(
        "polar", sp.Lambda((r, theta), (r * sp.cos(theta), r * sp.sin(theta)))
    )
    Dr = FunctionSpace(8, Legendre.Legendre, domain=Domain(sp.Rational(1, 2), 1))
    Ft = FunctionSpace(8, Fourier.Fourier, name="Ft")
    V = TensorProduct(Dr, Ft, system=polar, name="Vp")
    assert V.system.sg != 1, "this test is pointless without a metric weight"
    fields = [jnp.asarray(np.random.rand(*V.shape) + 0j) for _ in range(2)]
    for field, got in zip(
        fields, V.scalar_product_batch(jnp.stack(fields)), strict=True
    ):
        assert jnp.array_equal(got, V.scalar_product(field))


def test_backward_batch_direct_sum() -> None:
    """A DirectSum space lifts its boundary values, and must batch regardless."""
    F = FunctionSpace(8, Fourier.Fourier, domain=Domain(0, 2 * sp.pi), name="F")
    Tb = FunctionSpace(8, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 0}})
    VT = TensorProduct(F, Tb)
    x, y = VT.system.base_scalars()
    fields = [project(sp.sin(x) * (1 - y**2), VT), project(sp.cos(x) * y, VT)]
    batched = VT.backward_batch(jnp.stack(fields), N=(12, 8))
    for field, got in zip(fields, batched, strict=True):
        assert jnp.array_equal(got, VT.backward(field, N=(12, 8)))


if __name__ == "__main__":
    # test_forward_backward_2d(Fourier.Fourier)
    test_forward_backward_composite(Legendre.Legendre)


@pytest.mark.parametrize(
    "space", (Chebyshev.Chebyshev, ChebyshevU.ChebyshevU, Legendre.Legendre)
)
def test_forward_backward_complex(space) -> None:
    """A complex coefficient array must round-trip as a real one does.

    That is the wall-normal array in any Fourier x polynomial tensor product, and
    `ChebyshevU` transforms through `dst`, which used to take -Im of one complex
    FFT and so mixed the real and imaginary halves of a complex input.
    """
    D = space(12)
    rng = np.random.default_rng(0)
    ue = jnp.asarray(rng.standard_normal(12) + 1j * rng.standard_normal(12))
    assert jnp.linalg.norm(D.forward(D.backward(ue)) - ue) < ulp(1000)
