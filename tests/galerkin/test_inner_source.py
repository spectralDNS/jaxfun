"""Source terms kept apart from the time they are evaluated at.

`inner` folds a linear form's coefficient into the quadrature with `float()`, so
a source that depends on time cannot reach it at all -- it raises. That is the
right answer for a single assembly and the wrong one for a time integrator,
which needs the load vector again at every stage.

`inner_source` returns it as a function of time instead. The time dependence has
to factor out of the quadrature for that to be cheap: `separatevars` puts
everything free of the coordinates into `coeff`, so a source written as
`sum_k g_k(t) * h_k(x)` arrives with the time in one scalar per term and the
coordinate factors untouched. Each `h_k` then assembles once, through the same
`_assemble_linear_form` `inner` uses, and a stage costs one scalar evaluation
and one axpy per term.

That shared assembly is what these tests lean on: with a *constant* `g` the
result must be `inner`'s own vector, bit for bit, not merely close to it.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import R
from jaxfun.galerkin import Fourier, FunctionSpace, Legendre, TensorProduct
from jaxfun.galerkin.arguments import TestFunction
from jaxfun.galerkin.forms import split
from jaxfun.galerkin.inner import inner, inner_source
from jaxfun.typing import Array

pytestmark = pytest.mark.integration

W = 3.0  # a number, not a bare symbol -- see the stray-symbol test below


def _vec(expr: sp.Expr) -> Array:
    """`inner`'s load vector, narrowed: it returns a union over four shapes."""
    return cast(Array, inner(expr))


def _spaces_2d():
    R2 = R(2)
    x, y = R2.base_scalars()
    F = Fourier.Fourier(8, name="Fsrc")
    D = FunctionSpace(
        10, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}}, name="Dsrc"
    )
    T = TensorProduct(F, D, name="Tsrc")
    return T, TestFunction(T, name="vsrc"), x, y, R2.base_time()


SEPARABLE = {
    "product": lambda x, y: sp.sin(sp.pi * y) * sp.cos(2 * x),
    "multivar": lambda x, y: sp.sqrt(3 + x + y),
    "polynomial": lambda x, y: y**3 * sp.cos(x),
}


@pytest.mark.parametrize("name", sorted(SEPARABLE))
def test_a_constant_time_factor_reproduces_inner_exactly(name: str) -> None:
    """The oracle. `exp(-t)` at `t=0` is 1, so the two must agree bit for bit.

    Not `allclose`: the source path substitutes `coeff -> 1` and multiplies the
    assembled vector afterwards, so any difference would mean the assembly
    itself diverged rather than that the arithmetic reassociated.
    """
    T, v, x, y, t = _spaces_2d()
    h = SEPARABLE[name](x, y)
    source = inner_source(v * sp.exp(-t) * h)
    assert source is not None
    assert jnp.array_equal(source(0.0), _vec(v * h))


def test_the_vector_scales_with_its_time_factor() -> None:
    """A separable source is its steady vector times a scalar, at every time."""
    T, v, x, y, t = _spaces_2d()
    h = sp.sin(sp.pi * y) * sp.cos(2 * x)
    source = inner_source(v * sp.exp(-t) * h)
    assert source is not None
    base = _vec(v * h)
    for ti in (0.0, 0.5, 1.0):
        assert jnp.allclose(source(ti), float(jnp.exp(jnp.array(-ti))) * base)


def test_a_steady_source_is_not_a_source_forcing() -> None:
    """Nothing to defer, so nothing is built -- this is what keeps the old path."""
    T, v, x, y, _t = _spaces_2d()
    assert inner_source(v * sp.sin(sp.pi * y) * sp.cos(2 * x)) is None


def test_a_sum_of_separable_terms_is_carried_term_by_term() -> None:
    """Each term keeps its own time factor; they are not merged into one."""
    T, v, x, y, t = _spaces_2d()
    a = sp.exp(-t) * sp.sin(sp.pi * y)
    b = t**2 * sp.cos(sp.pi * y)
    source = inner_source(v * (a + b))
    assert source is not None
    want = _vec(v * sp.sin(sp.pi * y)) * float(jnp.exp(jnp.array(-0.4)))
    want = want + _vec(v * sp.cos(sp.pi * y)) * 0.16
    assert jnp.allclose(source(0.4), want)


def test_a_travelling_wave_qualifies_after_expand_trig() -> None:
    """`sin(kx - wt)` is separable once expanded, which is why it is expanded.

    Written as it stands the time rides inside the `x` factor and would be
    refused; `expand_trig` turns it into two terms that each factor cleanly.
    `split_transient_terms` applies that on the caller's behalf.
    """
    T, v, x, y, t = _spaces_2d()
    raw = sp.sin(x - W * t) * sp.sin(sp.pi * y)
    with pytest.raises(NotImplementedError, match="inside a coordinate factor"):
        inner_source(v * raw)
    source = inner_source(v * sp.expand(sp.expand_trig(raw)))
    assert source is not None
    assert not jnp.allclose(source(0.0), source(1.0))


@pytest.mark.parametrize(
    ("tag", "build"),
    [
        ("coordinate_factor", lambda x, y, t: sp.sqrt(3 + x + t) * sp.sin(sp.pi * y)),
        ("both_factors", lambda x, y, t: sp.exp(-t * (x**2 + y**2))),
        ("inside_a_product", lambda x, y, t: sp.sin(x * t) * sp.sin(sp.pi * y)),
    ],
)
def test_time_mixed_into_a_coordinate_factor_is_refused(tag: str, build) -> None:
    """Refused rather than mishandled: the quadrature itself would move.

    These need `uj` re-evaluated on the mesh at every stage, not a scalar
    rescaling of a fixed vector. The message names the offending factor, because
    the fix is to rewrite the source and the user has to see which part blocks
    it.
    """
    T, v, x, y, t = _spaces_2d()
    with pytest.raises(NotImplementedError, match="inside a coordinate factor"):
        inner_source(v * build(x, y, t))


def test_a_bilinear_form_is_refused() -> None:
    """Operators are assembled once; `inner_source` must not be handed one."""
    from jaxfun.galerkin.arguments import TrialFunction

    T, v, x, y, t = _spaces_2d()
    u = TrialFunction(T, name="usrc")
    with pytest.raises(ValueError, match="no bilinear forms"):
        inner_source(v * u * sp.exp(-t))


def test_the_time_factor_is_not_merged_into_a_multivar_factor() -> None:
    """`add_result` used to bury the time factor, making a valid source invalid.

    Two terms sharing every base-scalar key are merged by folding `coeff` into
    `multivar` and resetting it to 1. For a numeric coefficient that is
    harmless; for a time factor it moves the time out of the one place the
    source path looks for it and into the coordinate-dependent part, where it is
    then refused.
    """
    T, v, x, y, t = _spaces_2d()
    h = sp.sqrt(3 + x + y)
    forms = split(v * (sp.exp(-t) * h + h))["linear"]
    assert len(forms) == 2, "the two terms must not be merged"
    for form in forms:
        buried = [
            value
            for key, value in form.items()
            if key != "coeff" and sp.sympify(value).has(t)
        ]
        assert not buried, buried
    source = inner_source(v * (sp.exp(-t) * h + h))
    assert source is not None
    # Only the moving half is deferred; the steady one stays for
    # `assemble_linear_term` to fold into `linear_forcing` as it always has.
    assert len(source._vectors) == 1
    assert jnp.allclose(source(0.0), _vec(v * h))


def test_a_source_forcing_traces_under_jit() -> None:
    """It is rebuilt inside the stepped loop, so it has to be traceable in `t`."""
    T, v, x, y, t = _spaces_2d()
    source = inner_source(v * sp.exp(-t) * sp.sin(sp.pi * y))
    assert source is not None
    jitted = jax.jit(lambda ti: source(ti))
    assert jnp.allclose(jitted(0.3), source(0.3))
    assert "sin" not in str(jax.make_jaxpr(jitted)(0.3)), (
        "the spatial factors belong to the assembled vector, not the trace"
    )


def test_a_time_factor_with_a_free_parameter_is_refused() -> None:
    """A bare `sp.Symbol` frequency has no value to evaluate at.

    The factor is lambdified against time alone, so a stray symbol becomes an
    undefined name inside the generated function and fails from within a trace,
    far from the expression that caused it.
    """
    T, v, x, y, t = _spaces_2d()
    omega = sp.Symbol("omega")
    with pytest.raises(ValueError, match="free symbol"):
        inner_source(v * sp.cos(omega * t) * sp.sin(sp.pi * y))
