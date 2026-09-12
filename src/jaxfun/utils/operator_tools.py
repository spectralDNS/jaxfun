"""Helpers for normalizing assembled Galerkin operators."""

from typing import cast

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.inner import (
    BoundaryForcing,
    SourceForcing,
    inner,
    inner_boundary,
    inner_source,
)
from jaxfun.la import BaseMatrix, Matrix
from jaxfun.typing import Array, GalerkinAssembledForm

type AssembledTerm = tuple[BaseMatrix | None, Array | None]


def _normalize_assembled_operator(
    operator: BaseMatrix | Array | None,
) -> BaseMatrix | None:
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
    """Split an assembled Galerkin form into operator and forcing pieces.

    The result is normalized so that the assembled expression is always
    ``operator @ u + forcing``. `inner` does not use that convention on its own:
    when a form has a bilinear part it moves the linear part to the other side
    of the equation and returns it negated (`_linear_sign`), while a form
    consisting only of a linear part is returned as written. Undo that flip here
    so callers get one convention regardless of whether the expression happens
    to contain an operator.
    """
    if isinstance(form, list) or (
        isinstance(form, tuple) and any(isinstance(part, list) for part in form)
    ):
        raise ValueError(
            "`assemble_linear_term` expects collapsed assembly output. "
            "Raw operator lists are only produced by inner_items(...)."
        )

    if form is None:
        return None, None
    if isinstance(form, tuple):
        operator, forcing = cast(tuple[BaseMatrix | Array, Array | None], form)
        rhs = -jnp.asarray(forcing) if forcing is not None else None
        return _normalize_assembled_operator(operator), rhs

    if isinstance(form, Array):
        if form.ndim <= 1:
            return None, form
        return Matrix(form), None

    return cast(BaseMatrix, form), None


def assemble_linear_term(
    expr: sp.Expr, *, sparse: bool, sparse_tol: int
) -> AssembledTerm:
    """Assemble a linear weak-form expression into reusable operator data."""
    if sp.sympify(expr) == 0:
        return None, None

    linear_form = inner(expr, sparse=sparse, sparse_tol=sparse_tol)
    return split_operator_and_forcing(linear_form)


def assemble_boundary_term(expr: sp.Expr) -> BoundaryForcing | None:
    """Assemble the boundary blocks of `expr` without contracting them.

    The counterpart of `assemble_linear_term` for boundary data that moves.
    `assemble_linear_term` returns a forcing vector with the boundary
    contribution already folded in at the space's current lifting; this returns
    that contribution as a function of time instead, in the same
    `operator @ u + forcing` convention, so a caller can subtract the frozen
    copy and add back a live one.
    """
    # The convention is `split_operator_and_forcing`'s, not `inner`'s: the
    # result is negated, which is what makes the subtraction remove exactly the
    # copy `inner` folded in. That flip is the one applied to the tuple branch
    # there, and an evolution equation's mass and stiffness terms always
    # assemble an operator alongside the boundary block, so they always take it.
    #
    # No `sparse` flag because there is nothing to sparsify: the boundary
    # operators are contracted straight into a vector rather than kept as a
    # system to solve, and `inner` does not sparsify them either.
    if sp.sympify(expr) == 0:
        return None
    return inner_boundary(expr, outer_sign=-1)


def assemble_source_term(expr: sp.Expr) -> SourceForcing | None:
    """Assemble a time-dependent source as a function of time.

    The counterpart of `assemble_linear_term` for a source that moves. `expr`
    holds only the weak form's time-dependent additive terms, which
    `split_transient_terms` has already separated out.

    Unlike `assemble_boundary_term` there is **no** `outer_sign` here, and the
    asymmetry is real rather than an oversight. A boundary block is assembled
    alongside an operator, so `inner` negates the linear part and
    `split_operator_and_forcing` undoes it; the boundary term has to ask for the
    same flip to land in the caller's convention. A source expression carries no
    bilinear form at all, so neither flip fires and what comes back is already
    `operator @ u + forcing`.
    """
    if sp.sympify(expr) == 0:
        return None
    return inner_source(expr)
