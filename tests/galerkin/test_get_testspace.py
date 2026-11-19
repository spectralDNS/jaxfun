"""`get_testspace` on orthogonal, tensor-product and Cartesian product spaces."""

from functools import partial
from typing import cast

import jax.numpy as jnp
import pytest
from jax import Array

from jaxfun.galerkin import (
    CartesianProduct,
    Chebyshev,
    Fourier,
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.cartesianproductspace import (
    CartesianProductSpace,
    CartesianTensorProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.ChebyshevU import ChebyshevU
from jaxfun.galerkin.composite import GalerkinRecombined, TauRecombined
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import DirectSumTPS, TensorProductSpace
from jaxfun.galerkin.Ultraspherical import Ultraspherical
from jaxfun.la import BaseMatrix
from jaxfun.typing import RankTag
from jaxfun.utils.common import n

DD = {"left": {"D": 0}, "right": {"D": 0}}
# The tau polynomials U_{N-2} and C2_{N-1}.
TAU_BASIS = [(ChebyshevU, -2), (partial(Ultraspherical, lambda_=2), -1)]


@pytest.mark.parametrize("kind", ["G", "PG", "GR", "tau"])
def test_orthogonal_space_is_its_own_testspace(kind):
    F = Fourier.Fourier(8)
    assert F.get_testspace(kind) is F


def test_orthogonal_testspace_with_a_name_is_a_renamed_copy():
    F = Fourier.Fourier(8, name="F")
    W = F.get_testspace("PG", name="W", fun_str="w")
    assert W is not F and type(W) is type(F)
    assert (W.name, W.fun_str) == ("W", "w")
    assert (F.name, F.fun_str) == ("F", "E")
    assert W.orthogonal is W
    assert W.system is F.system and W.N == F.N


def test_orthogonal_testspace_rejects_scaling():
    with pytest.raises(NotImplementedError, match="no scaled test space"):
        Fourier.Fourier(8).get_testspace(scaling=n + 1)


def test_renamed_galerkin_testspace_assembles_like_the_space():
    C = Chebyshev.Chebyshev(8)
    W = C.get_testspace("G", name="W", fun_str="w")
    assert type(W) is Chebyshev.Chebyshev and W is not C
    (x,) = C.system.base_scalars()
    u = TrialFunction(C)
    A = cast(BaseMatrix, inner(u.diff(x) * TestFunction(C)))
    B = cast(BaseMatrix, inner(u.diff(x) * TestFunction(W)))
    assert jnp.allclose(A.todense(), B.todense())


def test_unknown_testspace_kind_is_rejected():
    with pytest.raises(ValueError, match="not a valid TestSpaceKind"):
        Fourier.Fourier(8).get_testspace("no-such-kind")


def test_tensorproduct_testspace_takes_each_axis_testspace():
    T = TensorProduct(
        Fourier.Fourier(8), FunctionSpace(10, Chebyshev.Chebyshev, bcs=DD)
    )
    P = T.get_testspace("GR")
    assert type(P) is TensorProductSpace
    assert P.system is T.system
    assert isinstance(P[0], Fourier.Fourier)
    assert isinstance(P[1], GalerkinRecombined)
    assert P.num_dofs == T.num_dofs


def test_tensorproduct_tau_testspace_sizes_the_tau_basis_per_axis():
    D = [FunctionSpace(N, Chebyshev.Chebyshev, bcs=DD) for N in (10, 11)]
    T = TensorProduct(Fourier.Fourier(8), *D)
    P = T.get_testspace("tau", tau_basis=TAU_BASIS)
    assert isinstance(P[0], Fourier.Fourier)
    for Pi, Di in zip(list(P)[1:], D, strict=True):
        ref = Di.get_testspace("tau", tau_basis=TAU_BASIS)
        assert isinstance(Pi, TauRecombined)
        assert jnp.array_equal(Pi.S.todense(), ref.S.todense())


def test_tensorproduct_galerkin_testspace_is_the_space_itself():
    D = FunctionSpace(10, Chebyshev.Chebyshev, bcs=DD)
    T = TensorProduct(D, D)
    assert T.get_testspace("G") is T
    assert T.get_testspace("G", name="V") is not T


def _inhomogeneous() -> tuple[DirectSumTPS, list]:
    Dx = FunctionSpace(8, Chebyshev.Chebyshev, bcs=DD)
    Dy = FunctionSpace(8, Chebyshev.Chebyshev, bcs={"left": {"D": 1}})
    T = TensorProduct(Dx, Dy)
    assert isinstance(T, DirectSumTPS)
    return T, [Dx, Dy]


@pytest.mark.integration
def test_directsum_testspace_is_homogeneous():
    T, axes = _inhomogeneous()
    P = T.get_testspace("GR")
    assert type(P) is TensorProductSpace
    assert P.system is T.system
    assert all(isinstance(s, GalerkinRecombined) for s in P)
    assert P.num_dofs == tuple(s.num_dofs for s in axes)


@pytest.mark.integration
def test_directsum_testspace_assembles_like_the_per_axis_one():
    T, (Dx, Dy) = _inhomogeneous()
    P = T.get_testspace("GR")
    Q = TensorProduct(Dx.get_testspace("GR"), Dy.get_testspace("GR"))
    u = TrialFunction(T)
    forms: list[tuple[BaseMatrix, Array]] = []
    for V in (P, Q):
        x, y = V.system.base_scalars()
        v = TestFunction(V)
        form = inner((u.diff(y) - u.diff(x)) * v, sparse=True, kind="system")
        forms.append(cast(tuple[BaseMatrix, Array], form))
    (A, b), (B, c) = forms
    assert jnp.allclose(A.todense(), B.todense())
    assert jnp.allclose(b, c)


def test_cartesian_product_testspace_takes_each_component_testspace():
    D = FunctionSpace(10, Chebyshev.Chebyshev, bcs=DD)
    C = CartesianProduct(D, Chebyshev.Chebyshev(10), name="C")
    P = C.get_testspace("GR")
    assert type(P) is CartesianProductSpace
    assert P.name == "C" and P.system is C.system
    assert all(isinstance(s, GalerkinRecombined) for s in P)
    assert C.get_testspace("G") is C


def test_cartesian_product_tau_testspace_takes_the_tau_basis():
    D = FunctionSpace(10, Chebyshev.Chebyshev, bcs=DD)
    V = TensorProduct(Fourier.Fourier(8), D)
    P = CartesianProduct(V, V, rank=1).get_testspace("tau", tau_basis=TAU_BASIS)
    ref = D.get_testspace("tau", tau_basis=TAU_BASIS)
    for T in P:
        assert isinstance(T[1], TauRecombined)
        assert jnp.array_equal(T[1].S.todense(), ref.S.todense())


def test_nested_cartesian_product_testspace_keeps_structure_and_rank():
    D = FunctionSpace(10, Chebyshev.Chebyshev, bcs=DD)
    V = TensorProduct(Fourier.Fourier(8), D)
    Q = TensorProduct(Fourier.Fourier(8), Chebyshev.Chebyshev(10))
    W = CartesianProduct(CartesianProduct(V, V, rank=1), Q, name="W")
    P = W.get_testspace("GR", name="P")
    assert type(P) is CartesianTensorProductSpace and P.name == "P"
    assert type(P[0]) is VectorTensorProductSpace and P[0].rank == RankTag.VECTOR
    assert P.system is W.system
    for i, T in enumerate(P.flatten()):
        assert isinstance(T[0], Fourier.Fourier)
        assert isinstance(T[1], GalerkinRecombined)
        assert T.global_index == i and T.leaf is P
    assert W[0].get_testspace("G") is W[0]
