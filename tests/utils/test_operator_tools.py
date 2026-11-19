from typing import cast

import jax.numpy as jnp
from jax import Array

from jaxfun.galerkin import (
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.la import TPMatrix
from jaxfun.utils.operator_tools import assemble_linear_term


def test_a_pure_source_in_two_dimensions_is_forcing():
    D = FunctionSpace(4, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    v = TestFunction(T)
    operator, forcing = assemble_linear_term(v * x * y, sparse=True, sparse_tol=1000)
    assert operator is None
    assert forcing is not None
    assert jnp.allclose(forcing, cast(Array, inner(v * x * y)))


def test_an_operator_and_source_in_two_dimensions_split():
    D = FunctionSpace(4, Legendre.Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})
    T = TensorProduct(D, D)
    x, y = T.system.base_scalars()
    v, u = TestFunction(T), TrialFunction(T)
    operator, forcing = assemble_linear_term(
        v * (u + x * y), sparse=True, sparse_tol=1000
    )
    assert isinstance(operator, TPMatrix)
    assert forcing is not None and forcing.shape == (2, 2)
    assert jnp.allclose(forcing, cast(Array, inner(v * x * y)))
