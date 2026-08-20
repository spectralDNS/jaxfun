"""`RFourier` against `Fourier`: the same field, half the spectrum.

Every test here is a comparison rather than a tolerance on `RFourier` alone.
The half spectrum is meant to hold exactly the coefficients `Fourier` holds at
k >= 0 and to reconstruct exactly the same (real) field from them, so the sharp
statement is always "these two agree", and it is what would break first if the
normalisation, the wavenumbers or the padding convention drifted apart.
"""

from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jax import Array

from jaxfun.galerkin import (
    FunctionSpace,
    Legendre,
    TensorProduct,
    TensorProductSpace,
)
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier, RFourier
from jaxfun.galerkin.inner import inner, project
from jaxfun.la import BaseMatrix
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import Domain, ulp

M = 16
LX = 3.0


def pair(**kw) -> tuple[Fourier, RFourier]:
    """Return a `Fourier` and an `RFourier` resolving the same physical field."""
    return Fourier(M, Domain(0, LX), **kw), RFourier(M, Domain(0, LX), **kw)


def real_field(seed: int = 0) -> Array:
    """Return random real samples on the shared quadrature mesh."""
    return jnp.asarray(np.random.default_rng(seed).standard_normal(M))


def hermitian_pair() -> tuple[Array, Array]:
    """Return the two spectra of one real field, with the Nyquist zeroed.

    Zeroing it is what makes the two representations agree about derivatives and
    about padding; see `RFourier`'s docstring. Every comparison below that goes
    through either starts here.
    """
    F, R = pair()
    cf = F.forward(real_field()).at[M // 2].set(0)
    return cf, cf[: M // 2 + 1]


def test_sizes() -> None:
    F, R = pair()
    assert (F.N, F.num_quad_points) == (M, M)
    assert (R.N, R.num_quad_points) == (M // 2 + 1, M)
    assert R.dim == R.num_dofs == M // 2 + 1
    assert R.shape == F.shape == (M,)
    assert jnp.allclose(R.mesh(), F.mesh())
    assert jnp.array_equal(R.wavenumbers(), jnp.arange(M // 2 + 1))


def test_forward_is_the_nonnegative_half() -> None:
    F, R = pair()
    u = real_field()
    assert jnp.abs(F.forward(u)[: M // 2 + 1] - R.forward(u)).max() < ulp(100)
    assert jnp.abs(F.scalar_product(u)[: M // 2 + 1] - R.scalar_product(u)).max() < ulp(
        100
    )


def test_roundtrip() -> None:
    _, R = pair()
    u = real_field()
    c = R.forward(u)
    assert c.shape == (M // 2 + 1,)
    assert jnp.abs(R.backward(c) - u).max() < ulp(100)


def test_backward_matches_fourier() -> None:
    F, R = pair()
    cf, cr = hermitian_pair()
    uf = F.backward(cf)
    assert jnp.abs(uf.imag).max() < ulp(100), "the test field must be real"
    assert jnp.abs(R.backward(cr) - uf.real).max() < ulp(100)
    assert not jnp.iscomplexobj(R.backward(cr))


@pytest.mark.parametrize("Np", (M, 24, 32))
def test_padding_matches_fourier(Np: int) -> None:
    F, R = pair()
    cf, cr = hermitian_pair()
    assert jnp.abs(R.backward(cr, Np) - F.backward(cf, Np).real).max() < ulp(100)
    # ... and truncates back to exactly what it came from.
    assert jnp.abs(R.forward(R.backward(cr, Np)) - cr).max() < ulp(1000)


@pytest.mark.parametrize("k", (1, 2, 3))
def test_derivatives_match_fourier(k: int) -> None:
    F, R = pair()
    cf, cr = hermitian_pair()
    expected = F.backward_primitive(cf, k=k, N=24).real
    got = R.backward_primitive(cr, k=k, N=24)
    assert jnp.abs(got - expected).max() < ulp(100) * max(
        float(jnp.abs(expected).max()), 1.0
    )


def test_evaluate_at_scattered_points() -> None:
    """`evaluate` must agree with `backward`, which is not automatic here.

    The reconstruction folds the dropped conjugate half back in as a weight and
    takes the real part, so it is *not* `vandermonde @ c` -- the one place the
    half spectrum breaks an identity the other spaces satisfy.
    """
    F, R = pair()
    cf, cr = hermitian_pair()
    x = jnp.asarray(np.random.default_rng(1).uniform(0, LX, 9))
    assert jnp.abs(R.evaluate(x, cr) - F.evaluate(x, cf).real).max() < ulp(100)
    assert jnp.abs(R.evaluate(R.mesh(), cr) - R.backward(cr)).max() < ulp(1000)


def test_truncated_coefficients_evaluate_consistently() -> None:
    """A coefficient vector short of the Nyquist keeps the doubling weight."""
    F, R = pair()
    cf, _ = hermitian_pair()
    cut = M // 2 - 2
    x = jnp.asarray(np.random.default_rng(2).uniform(0, LX, 5))
    # The same truncation on the full spectrum drops |k| >= cut from both halves.
    expected = F.evaluate(x, cf.at[cut : M - cut + 1].set(0)).real
    assert jnp.abs(R.evaluate(x, cf[:cut]) - expected).max() < ulp(100)


def test_complex_samples_are_rejected() -> None:
    """A complex array is refused rather than silently losing its imaginary part."""
    _, R = pair()
    for fn in (R.forward, R.scalar_product):
        with pytest.raises(AssertionError, match="real fields"):
            fn(real_field() + 0j)


def test_mass_matrix_is_the_restricted_fourier_one() -> None:
    F, R = pair()
    Mf = F.mass_matrix().todense()
    Mr = R.mass_matrix().todense()
    n = M // 2 + 1
    assert jnp.abs(jnp.diag(Mr) - jnp.diag(Mf)[:n]).max() < ulp(100)


def tensor_pair() -> tuple[TensorProductSpace, TensorProductSpace]:
    """Return Fourier x Legendre and RFourier x Legendre, Dirichlet in y."""
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    out = []
    for tag, cls in (("F", Fourier), ("R", RFourier)):
        Fx = FunctionSpace(M, cls, domain=Domain(0, LX), name="X" + tag)
        D = FunctionSpace(12, Legendre.Legendre, bcs=hom, name="D" + tag)
        out.append(TensorProduct(Fx, D, name="V" + tag))
    return out[0], out[1]


def tensor_projection() -> tuple[TensorProductSpace, Array, TensorProductSpace, Array]:
    """Project one smooth function into both spaces and return both expansions."""
    VF, VR = tensor_pair()
    out = []
    for V in (VF, VR):
        x, y = V.system.base_scalars()
        out += [V, project(sp.sin(2 * sp.pi * x / LX) * (1 - y**2), V)]
    return VF, out[1], VR, out[3]


def test_tensor_product_transforms() -> None:
    VF, cf, VR, cr = tensor_projection()
    n = M // 2 + 1
    assert VR.num_dofs == (n, VF.num_dofs[1])
    assert jnp.abs(cf[:n] - cr).max() < ulp(100)

    uf, ur = VF.backward(cf), VR.backward(cr)
    assert not jnp.iscomplexobj(ur)
    assert jnp.abs(ur - uf.real).max() < ulp(100)
    assert jnp.abs(VR.forward(ur) - cr).max() < ulp(1000)

    pad = (24, 18)
    assert jnp.abs(VR.backward(cr, pad) - VF.backward(cf, pad).real).max() < ulp(100)
    assert jnp.abs(
        VR.backward_primitive(cr, (1, 1)) - VF.backward_primitive(cf, (1, 1)).real
    ).max() < ulp(1000)


def test_tensor_product_evaluate() -> None:
    VF, cf, VR, cr = tensor_projection()
    rng = np.random.default_rng(3)
    pts = jnp.stack(
        [jnp.asarray(rng.uniform(0, LX, 6)), jnp.asarray(rng.uniform(-1, 1, 6))],
        axis=1,
    )
    assert VR.is_hermitian_half and not VF.is_hermitian_half
    assert jnp.abs(VR.evaluate(pts, cr) - VF.evaluate(pts, cf).real).max() < ulp(100)
    assert jnp.abs(VR.evaluate_mesh(cr) - VR.backward(cr)).max() < ulp(100)


def test_tensor_product_batch() -> None:
    _, _, VR, cr = tensor_projection()
    ur = VR.backward(cr)
    batch = VR.backward_batch(jnp.stack([cr, 2 * cr]))
    assert jnp.abs(batch[0] - ur).max() < ulp(100)
    assert jnp.abs(batch[1] - 2 * ur).max() < ulp(1000)
    back = VR.forward_batch(jnp.stack([ur, 2 * ur]))
    assert jnp.abs(back[1] - 2 * cr).max() < ulp(1000)


def test_direct_sum_with_inhomogeneous_boundary() -> None:
    """A Dirichlet lifting in y is orthogonal to the half spectrum in x.

    `DirectSum` is not an `OrthogonalSpace`, so it carries the
    `is_hermitian_half` flag itself; this is what checks that a `DirectSumTPS`
    still reports and reconstructs correctly with an `RFourier` axis. It is the
    shape a temperature field with fixed wall values takes.
    """
    bc = {"left": {"D": 1}, "right": {"D": 0}}
    out = []
    for tag, cls in (("F", Fourier), ("R", RFourier)):
        Fx = FunctionSpace(M, cls, domain=Domain(0, LX), name="B" + tag)
        D = FunctionSpace(12, Legendre.Legendre, bcs=bc, name="Bd" + tag)
        V = TensorProduct(Fx, D, name="Vb" + tag)
        x, y = V.system.base_scalars()
        out += [V, project(sp.sin(2 * sp.pi * x / LX) * (1 - y) / 2, V)]
    VF, cf, VR, cr = out[0], out[1], out[2], out[3]

    assert VR.is_hermitian_half and not VF.is_hermitian_half
    ur = VR.backward(cr)
    assert not jnp.iscomplexobj(ur)
    assert jnp.abs(ur - VF.backward(cf).real).max() < ulp(100)
    assert jnp.abs(
        VR.to_orthogonal(cr) - VF.to_orthogonal(cf)[: M // 2 + 1]
    ).max() < ulp(100)

    rng = np.random.default_rng(4)
    pts = jnp.stack(
        [jnp.asarray(rng.uniform(0, LX, 5)), jnp.asarray(rng.uniform(-1, 1, 5))],
        axis=1,
    )
    assert jnp.abs(VR.evaluate(pts, cr) - VF.evaluate(pts, cf).real).max() < ulp(100)
    assert jnp.abs(VR.evaluate_mesh(cr) - ur).max() < ulp(100)


def test_poisson_solve_matches_full_spectrum() -> None:
    """The dropped equations were redundant, so the kept ones solve to the same."""
    VF, VR = tensor_pair()
    got = []
    for V in (VF, VR):
        x, y = V.system.base_scalars()
        ue = sp.sin(2 * sp.pi * x / LX) * (1 - y**2)
        u, v = TrialFunction(V, name="u"), TestFunction(V, name="v")
        A, b = cast(
            tuple[BaseMatrix, Array],
            inner(
                Div(Grad(u)) * v - (sp.diff(ue, x, 2) + sp.diff(ue, y, 2)) * v,
                sparse=True,
            ),
        )
        uh = A.solve(b)
        assert jnp.abs(uh - project(ue, V)).max() < ulp(1000)
        got.append(uh)
    assert jnp.abs(got[0][: M // 2 + 1] - got[1]).max() < ulp(1000)


@pytest.mark.spmd
def test_sharded_transform_round_trip() -> None:
    """A half spectrum shards like any other -- when its length divides.

    What has to divide by the device count is the coefficient count, N/2 + 1,
    not N. `RFourier(14)` gives 8 and shards over two devices; `RFourier(16)`
    gives 9 and does not, which is what the second half of this checks.
    """
    import jax

    if jax.device_count() < 2:
        pytest.skip("needs at least 2 devices")

    hom = {"left": {"D": 0}, "right": {"D": 0}}

    def build(n: int) -> TensorProductSpace:
        Fx = FunctionSpace(n, RFourier, domain=Domain(0, LX), name="Fs")
        D = FunctionSpace(12, Legendre.Legendre, bcs=hom, name="Ds")
        return TensorProduct(Fx, D, name="Vs")

    divides = 2 * (jax.device_count() * 4 - 1)  # N/2 + 1 == 4 * device_count
    V = build(divides)
    x, y = V.system.base_scalars()
    c = project(sp.sin(2 * sp.pi * x / LX) * (1 - y**2), V)
    u = V.backward(c)
    assert not jnp.iscomplexobj(u)
    assert jnp.abs(V.forward(u) - c).max() < ulp(1000)

    # Two more quadrature points is one more coefficient, which no device count
    # above one divides any more.
    V_odd = build(divides + 2)
    x, y = V_odd.system.base_scalars()
    with pytest.raises(ValueError, match="divisible"):
        V_odd.backward(project(sp.sin(2 * sp.pi * x / LX) * (1 - y**2), V_odd))
