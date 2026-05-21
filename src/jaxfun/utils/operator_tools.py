"""Helpers for normalizing assembled Galerkin operators."""

from typing import cast

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.inner import inner
from jaxfun.la import Matrix
from jaxfun.typing import Array, GalerkinAssembledForm, GalerkinOperator

type AssembledTerm = tuple[GalerkinOperator | None, Array | None]


def _normalize_assembled_operator(
    operator: GalerkinOperator | Array | None,
) -> GalerkinOperator | None:
    """Return one concrete matrix-like operator for time integration."""
    if operator is None:
        return None

    if isinstance(operator, Array):
        if operator.ndim <= 1:
            return None
        return Matrix(operator)

    return operator


def split_operator_and_forcing(
    form: GalerkinAssembledForm,
) -> AssembledTerm:
    """Split an assembled Galerkin form into operator and forcing pieces."""
    if isinstance(form, list):
        raise ValueError(
            "`assemble_linear_term` expects collapsed assembly output. "
            "Raw operator lists are only produced by inner(..., return_all_items=True)."
        )

    if form is None:
        return None, None
    if isinstance(form, tuple):
        operator, forcing = cast(tuple[GalerkinOperator | Array, Array | None], form)
        rhs = jnp.asarray(forcing) if forcing is not None else None
        return _normalize_assembled_operator(operator), rhs

    if isinstance(form, Array):
        if form.ndim <= 1:
            return None, form
        return Matrix(form), None

    return form, None


def assemble_linear_term(
    expr: sp.Expr, *, sparse: bool, sparse_tol: int
) -> AssembledTerm:
    """Assemble a linear weak-form expression into reusable operator data."""
    if sp.sympify(expr) == 0:
        return None, None

    linear_form = cast(
        GalerkinAssembledForm, inner(expr, sparse=sparse, sparse_tol=sparse_tol)
    )
    return split_operator_and_forcing(linear_form)
