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
@pytest.mark.parametrize("n", [14, 16, 128], ids=["divides", "odd", "power-of-two"])
def test_sharded_transform_round_trip(n: int) -> None:
    """A half spectrum shards whatever its length, because it pads itself.

    What has to divide by the device count is the coefficient count, not `N`.
    `RFourier(16)` stores 9 wavenumbers and no device count above one divides
    that -- so it stores padding after them instead, and it is the padded count
    that is split. `N = 128` is the case that motivated it: a power of two for
    the FFT, and 65 wavenumbers, which nothing divides.
    """
    import jax

    if jax.device_count() < 2:
        pytest.skip("needs at least 2 devices")

    Fx = FunctionSpace(n, RFourier, domain=Domain(0, LX), name="Fs")
    D = FunctionSpace(12, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    V = TensorProduct(Fx, D, name="Vs")

    assert Fx.n_real == n // 2 + 1
    assert Fx.N % jax.device_count() == 0, "the stored axis has to split"
    assert Fx.N - Fx.n_real < jax.device_count(), "no more padding than needed"

    x, y = V.system.base_scalars()
    c = project(sp.sin(2 * sp.pi * x / LX) * (1 - y**2), V)
    u = V.backward(c)
    assert not jnp.iscomplexobj(u)
    assert jnp.abs(V.forward(u) - c).max() < ulp(1000)
    # The padding carries no field: it starts at zero and no transform fills it.
    # Summed rather than maxed so the unpadded case, which is a legitimate one,
    # reduces an empty slice instead of raising.
    assert jnp.abs(c[Fx.n_real :]).sum() == 0.0
    assert jnp.abs(V.forward(u)[Fx.n_real :]).sum() == 0.0


@pytest.mark.spmd
def test_padding_stays_empty_through_a_nonlinear_solve() -> None:
    """A whole time loop must leave the padding at zero, not just a projection.

    The invariant is structural -- a scalar product zero-fills the padding, a
    Fourier operator is diagonal and cannot move anything into it, and a solve
    whose right-hand side is zero there returns zero -- but every one of those
    three is an implementation detail of a different file, so the composition is
    worth pinning down. The nonlinear term is what makes it a real test: it goes
    out to the padded quadrature mesh and back at every stage.
    """
    import jax

    from jaxfun.integrators import ARS443, IMEXRungeKutta
    from jaxfun.operators import Constant

    if jax.device_count() < 2:
        pytest.skip("needs at least 2 devices")

    n = jax.device_count()
    Fx = FunctionSpace(16 * n, RFourier, domain=Domain(0, LX), name="Fx")
    D = FunctionSpace(
        8 * n, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}}
    )
    V = TensorProduct(Fx, D, name="Vn")
    # 16*n quadrature points store 8*n + 1 coefficients, which n never divides,
    # so the padding is always exercised. The polynomial axis is sized off the
    # device count too, so the transform takes its distributed path rather than
    # falling back for want of a divisible quadrature count.
    assert Fx.n_extra, f"{Fx.n_real} wavenumbers should have needed padding"

    x, y = V.system.base_scalars()
    t = V.system.base_time()
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    stepper = IMEXRungeKutta(
        v * (u.diff(t) - Constant("nu", 0.5) * Div(Grad(u))) + v * u**2,
        tableau=ARS443,
        initial=sp.cos(2 * sp.pi * x / LX) * (1 - y**2),
        sparse=True,
    )
    u0 = jnp.asarray(stepper.initial_coefficients())
    assert jnp.abs(u0[Fx.n_real :]).sum() == 0.0

    uh = stepper.solve(dt=1e-3, steps=20, state0=u0, n_batches=2, progress=False)
    assert jnp.isfinite(uh).all()
    assert jnp.abs(jnp.asarray(uh)[Fx.n_real :]).max() < ulp(100)


@pytest.mark.spmd
def test_unpadded_indivisible_axis_is_refused() -> None:
    """Padding off and a count that does not divide is an error, not a fallback.

    Running the transform on one device instead would be correct and silently
    half the speed, which is the one outcome worth refusing.
    """
    import jax

    if jax.device_count() < 2:
        pytest.skip("needs at least 2 devices")

    n = jax.device_count()
    # A half spectrum this device count cannot divide, whatever it happens to
    # be: 8n quadrature points store 4n + 1 coefficients, and 4n + 1 leaves a
    # remainder of 1 for every n > 1.
    Fx = RFourier(8 * n, domain=Domain(0, LX), name="Fs", n_extra=0)
    D = FunctionSpace(12, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    V = TensorProduct(Fx, D, name="Vs")
    assert Fx.N % n, f"{Fx.N} coefficients happen to divide by {n}"
    with pytest.raises(ValueError, match="cannot divide"):
        V.backward(jnp.zeros(V.num_dofs, dtype=complex))


# ---------------------------------------------------------------------------
# The half axis constrains the tensor product: at most one, and it must be
# first, and the separable transforms have to keep it in the right place.
# ---------------------------------------------------------------------------
def test_two_half_axes_are_rejected() -> None:
    """Two `RFourier` axes would drop coefficients nothing determines.

    A real field's spectrum is Hermitian under a joint reflection of every axis,
    which pairs half the coefficients with the other half exactly once. Halving
    one axis spends that symmetry; halving a second keeps one quadrant of four
    and throws away two the first has no relation to.
    """
    R = [
        FunctionSpace(M, RFourier, domain=Domain(0, LX), name=f"R{i}") for i in range(2)
    ]
    with pytest.raises(ValueError, match="at most one axis"):
        TensorProduct(*R, name="Vrr")


def test_half_axis_must_come_first() -> None:
    """`Fourier x RFourier` is valid mathematically but not supported here.

    It is what `rfft2` does. But the forward transform runs in axis order, so
    the leading complex axis would make the array complex before the `rfft`
    sees it, and the sharded paths make the same assumption. Refused at
    construction rather than left to fail inside a transform.
    """
    F = FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Ff")
    R = FunctionSpace(M, RFourier, domain=Domain(0, LX), name="Rf")
    with pytest.raises(ValueError, match="must be the first axis"):
        TensorProduct(F, R, name="Vfr")


@pytest.mark.parametrize("pad", [None, (M, 3 * M // 2), (M, 2 * M), (M, 3 * M)])
def test_backward_keeps_the_half_axis_last(pad: tuple[int, int] | None) -> None:
    """Padding the other axis must not reorder the transforms.

    `backward` orders axes cheapest-first, and a half axis grows by about 2x
    (N/2 + 1 -> N), so it lands last on cost alone for as long as the other
    axis is padded by less than that. It must land last for *correctness*:
    `irfft` returns a real array, so running it first discards the imaginary
    parts still carrying the other axis's information -- silently, with no
    error. Padding the second axis by 2x or more is what used to flip the order.
    """
    Vr = TensorProduct(
        FunctionSpace(M, RFourier, domain=Domain(0, LX), name="Rb"),
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Fb"),
        name="Vrb",
    )
    Vf = TensorProduct(
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Ff2"),
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Ff3"),
        name="Vff",
    )
    # Periodic on [0, LX) and well below Nyquist in both directions: the
    # comparison is between two representations of the *same* interpolant, so
    # any content at Nyquist would compare their differing conventions for it
    # rather than the axis ordering this test is about.
    g = jnp.linspace(0, LX, M, endpoint=False)
    xx, yy = jnp.meshgrid(g, g, indexing="ij")
    kx, ky = 2 * jnp.pi * xx / LX, 2 * jnp.pi * yy / LX
    u = jnp.sin(2 * kx) * jnp.cos(3 * ky) + 0.5 * jnp.cos(kx + 2 * ky) + 1.3

    got = Vr.backward(Vr.forward(u), N=pad).real
    ref = Vf.backward(Vf.forward(u), N=pad).real
    assert jnp.abs(got - ref).max() < ulp(1000) * max(1.0, float(jnp.abs(ref).max()))


def test_hermitian_axis_index() -> None:
    """`hermitian_axis` names the axis; `is_hermitian_half` stays the predicate."""
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    D = FunctionSpace(12, Legendre.Legendre, bcs=hom, name="Dh")
    Vr = TensorProduct(
        FunctionSpace(M, RFourier, domain=Domain(0, LX), name="Rh"), D, name="Vrh"
    )
    Vf = TensorProduct(
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Fh"), D, name="Vfh"
    )
    assert Vr.hermitian_axis == 0
    assert Vr.is_hermitian_half
    assert Vf.hermitian_axis is None
    assert not Vf.is_hermitian_half


# ---------------------------------------------------------------------------
# `real=True`: the user declares the field real, the factory does the rest.
# ---------------------------------------------------------------------------
def test_real_flag_halves_the_leading_fourier_axis() -> None:
    """`real=True` swaps `Fourier` for `RFourier` in place, and nothing else."""
    hom = {"left": {"D": 0}, "right": {"D": 0}}

    def build(real: bool) -> TensorProductSpace:
        return TensorProduct(
            FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Fr"),
            FunctionSpace(12, Legendre.Legendre, bcs=hom, name="Dr"),
            name="Vr" + str(real),
            real=real,
        )

    Vr, Vf = build(True), build(False)
    assert Vr.hermitian_axis == 0
    assert Vf.hermitian_axis is None
    assert Vr.num_dofs == (M // 2 + 1, Vf.num_dofs[1])
    # Same field, same physical mesh: only the storage differs.
    assert Vr.shape == Vf.shape
    x, y = Vr.system.base_scalars()
    f = sp.sin(2 * sp.pi * x / LX) * (1 - y**2)
    u = Vr.backward(project(f, Vr))
    assert not jnp.iscomplexobj(u)
    xf, yf = Vf.system.base_scalars()
    ref = Vf.backward(project(sp.sin(2 * sp.pi * xf / LX) * (1 - yf**2), Vf)).real
    assert jnp.abs(u - ref).max() < ulp(1000)


def test_real_flag_halves_only_one_of_two_fourier_axes() -> None:
    """Fourier x Fourier keeps the second axis whole: one joint symmetry, one axis."""
    V = TensorProduct(
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="F1"),
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="F2"),
        name="Vff2",
        real=True,
    )
    assert V.hermitian_axis == 0
    assert V.num_dofs == (M // 2 + 1, M)


def test_real_flag_is_idempotent() -> None:
    """Asking for a half spectrum that is already there is not an error."""
    V = TensorProduct(
        FunctionSpace(M, RFourier, domain=Domain(0, LX), name="Ri"),
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Fi"),
        name="Vri",
        real=True,
    )
    assert V.hermitian_axis == 0
    assert V.num_dofs == (M // 2 + 1, M)


@pytest.mark.parametrize("second_is_fourier", [True, False])
def test_real_flag_needs_fourier_first(second_is_fourier: bool) -> None:
    """The half axis has to be first, so `real=True` refuses to reorder for you."""
    hom = {"left": {"D": 0}, "right": {"D": 0}}
    second = (
        FunctionSpace(M, Fourier, domain=Domain(0, LX), name="Fn")
        if second_is_fourier
        else FunctionSpace(12, Legendre.Legendre, bcs=hom, name="Dn2")
    )
    with pytest.raises(ValueError, match="first axis has to be Fourier"):
        TensorProduct(
            FunctionSpace(12, Legendre.Legendre, bcs=hom, name="Dn1"),
            second,
            name="Vbad",
            real=True,
        )
