"""Helpers for assembling, inspecting, and applying Galerkin operators."""

from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.tensorproductspace import TensorMatrix, TPMatrices, TPMatrix
from jaxfun.la import DiaMatrix, Matrix
from jaxfun.typing import (
    Array,
    GalerkinAssembledForm,
    GalerkinOperator,
    GalerkinOperatorLike,
)


@dataclass(frozen=True)
class AssembledTerm:
    """Container for an assembled operator, forcing term, and diagonal shortcut."""

    operator: GalerkinOperatorLike | None = None
    forcing: Array | None = None
    diagonal: Array | None = None


def sparse_diagonal(mat: DiaMatrix) -> Array | None:
    """Return the diagonal of a sparse DiaMatrix matrix when it is purely diagonal."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None
    if mat.offsets != (0,):
        return None
    return mat.diagonal()


def _sum_diagonals(operators: list[GalerkinOperator]) -> Array | None:
    """Return the summed diagonal of several operators, if all are diagonal."""
    diag_sum: Array | None = None
    for operator in operators:
        diag = operator_diagonal(operator)
        if diag is None:
            return None
        diag_sum = diag if diag_sum is None else diag_sum + diag
    return diag_sum


def _tpmatrix_diagonal(op: TPMatrix) -> Array | None:
    """Return the tensor-product diagonal implied by a TPMatrix."""
    if len(op.mats) == 0:
        return None

    diagonals: list[Array] = []
    for diag in (operator_diagonal(item) for item in op.mats):
        if diag is None or diag.ndim != 1:
            return None
        diagonals.append(diag)

    diagonal = diagonals[0]
    for axis, diag in enumerate(diagonals[1:], start=1):
        shape = (1,) * axis + (diag.shape[0],)
        diagonal = diagonal[..., None] * diag.reshape(shape)
    return diagonal


def operator_diagonal(obj: GalerkinOperatorLike | None) -> Array | None:
    """Return a diagonal representation when an operator acts diagonally."""
    if obj is None:
        return None
    if isinstance(obj, list):
        return _sum_diagonals(cast(list[GalerkinOperator], obj))
    if isinstance(obj, DiaMatrix):
        return sparse_diagonal(obj)
    if isinstance(obj, Matrix):
        return obj.data.diagonal()
    if isinstance(obj, TPMatrices):
        return _sum_diagonals(cast(list[GalerkinOperator], list(obj.tpmats)))
    if isinstance(obj, TPMatrix):
        return _tpmatrix_diagonal(obj)
    if isinstance(obj, TensorMatrix):
        return operator_diagonal(obj.mat)
    arr = jnp.asarray(obj)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        diag = jnp.diag(arr)
        if bool(jnp.allclose(arr, jnp.diag(diag), atol=1e-12)):
            return diag
    return None


def apply_operator(op: GalerkinOperatorLike | None, u: Array) -> Array:
    """Apply a Galerkin operator-like object to a coefficient array."""
    if op is None:
        return jnp.zeros_like(u)
    if isinstance(op, list):
        items = cast(list[GalerkinOperator], op)
        applied = [apply_operator(item, u) for item in items]
        return jnp.sum(jnp.stack(applied), axis=0)
    if isinstance(op, TPMatrices | TPMatrix):
        return op(u)
    if isinstance(op, TensorMatrix):
        return apply_operator(op.mat, u)
    if isinstance(op, DiaMatrix | Matrix):
        return op @ u
    arr = jnp.asarray(op)
    if arr.ndim == u.ndim:
        return arr * u
    if arr.ndim != 2:
        raise ValueError("Can only apply rank-1 or rank-2 operators")
    return arr @ u


def _sum_dense_operators(operators: list[GalerkinOperator]) -> Array:
    """Convert and sum a list of operators in dense form."""
    mats = [operator_to_dense(op) for op in operators]
    return sum(mats[1:], mats[0]) if len(mats) > 0 else jnp.array([])


def _tpmatrix_to_dense(op: TPMatrix) -> Array:
    """Convert a TPMatrix into an explicit dense Kronecker product."""
    mats = [operator_to_dense(item) for item in op.mats]
    if len(mats) == 0:
        return jnp.array([])

    dense: Array = mats[0]
    for mat in mats[1:]:
        dense = jnp.kron(dense, mat)
    return dense * jnp.asarray(op.scale)


def operator_to_dense(op: GalerkinOperatorLike) -> Array:
    """Convert a supported Galerkin operator representation to a dense array."""
    if isinstance(op, DiaMatrix):
        return op.todense()
    if isinstance(op, Matrix):
        return op.data
    if isinstance(op, list):
        return _sum_dense_operators(cast(list[GalerkinOperator], op))
    if isinstance(op, TPMatrices):
        return _sum_dense_operators(cast(list[GalerkinOperator], list(op.tpmats)))
    if isinstance(op, TPMatrix):
        return _tpmatrix_to_dense(op)
    if isinstance(op, TensorMatrix):
        return operator_to_dense(op.mat)
    return jnp.asarray(op)


def solve_operator(op: GalerkinOperatorLike, rhs: Array) -> Array:
    """Solve a dense linear system represented by ``op`` against ``rhs``."""
    mat = operator_to_dense(op)
    if mat.ndim != 2:
        raise ValueError("Can only solve systems with rank-2 operators")
    b = rhs.reshape((-1,))
    x = jnp.linalg.solve(mat, b)
    return x.reshape(rhs.shape)


def split_operator_and_forcing(
    form: GalerkinAssembledForm,
) -> tuple[GalerkinOperatorLike | None, Array | None]:
    """Split an assembled Galerkin form into operator and forcing pieces."""
    if form is None:
        return None, None
    if isinstance(form, tuple):
        operator, forcing = cast(tuple[GalerkinOperatorLike, Array | None], form)
        rhs = jnp.asarray(forcing) if forcing is not None else None
        return operator, rhs
    if isinstance(
        form, list | DiaMatrix | Matrix | TPMatrix | TensorMatrix | TPMatrices
    ):
        return form, None
    arr = jnp.asarray(form)
    if arr.ndim <= 1:
        return None, arr
    return arr, None


def assemble_linear_term(
    expr: sp.Expr, *, sparse: bool, sparse_tol: int
) -> AssembledTerm:
    """Assemble a linear weak-form expression into reusable operator data."""
    if sp.sympify(expr) == 0:
        return AssembledTerm()

    linear_form = cast(
        GalerkinAssembledForm, inner(expr, sparse=sparse, sparse_tol=sparse_tol)
    )
    operator, forcing = split_operator_and_forcing(linear_form)
    return AssembledTerm(
        operator=operator,
        forcing=forcing,
        diagonal=operator_diagonal(operator),
    )
