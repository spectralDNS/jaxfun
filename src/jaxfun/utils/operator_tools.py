"""Helpers for assembling, inspecting, and applying Galerkin operators."""

from dataclasses import dataclass
from typing import cast

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.inner import inner
from jaxfun.la import DiaMatrix, Matrix, TensorMatrix, TPMatrices, TPMatrix
from jaxfun.la.matrixprotocol import SolverNotApplicable
from jaxfun.typing import (
    Array,
    GalerkinAssembledForm,
    GalerkinOperator,
    GalerkinOperatorLike,
)


@dataclass(frozen=True, eq=False)
class LinearTerm:
    """Normalized linear term assembled from a Galerkin weak form."""

    operator: GalerkinOperatorLike | None = None
    forcing: Array | None = None
    diagonal: Array | None = None

    @property
    def is_zero(self) -> bool:
        """Return True when the term has no operator or forcing contribution."""
        return self.operator is None and self.forcing is None and self.diagonal is None

    def diagonal_or_none(self) -> Array | None:
        """Return an efficient diagonal representation when available."""
        if self.diagonal is not None:
            return self.diagonal
        return operator_diagonal(self.operator)

    def apply(self, u: Array) -> Array:
        """Apply the operator part of this term to a coefficient state."""
        diagonal = self.diagonal_or_none()
        if diagonal is not None:
            return _apply_diagonal(diagonal, u)

        if self.operator is None:
            return jnp.zeros_like(u)

        return apply_operator(self.operator, u)

    def solve(self, rhs: Array) -> Array:
        """Solve the operator part of this term against a right-hand side."""
        diagonal = self.diagonal_or_none()
        if diagonal is not None:
            return _solve_diagonal(diagonal, rhs)

        if self.operator is None:
            return rhs

        return solve_operator(self.operator, rhs)

    def todense(self, state_size: int, *, identity_if_zero: bool = False) -> Array:
        """Return a dense global matrix for this term."""
        diagonal = self.diagonal_or_none()
        if diagonal is not None:
            return jnp.diag(diagonal.reshape((-1,)))
        if self.operator is None:
            if identity_if_zero:
                return jnp.eye(state_size)
            return jnp.zeros((state_size, state_size))
        return operator_to_dense(self.operator)


AssembledTerm = LinearTerm


def _apply_diagonal(diagonal: Array, u: Array) -> Array:
    """Apply a diagonal stored either in coefficient shape or flattened shape."""
    if diagonal.shape == u.shape:
        return diagonal * u
    flat = diagonal.reshape((-1,)) * u.reshape((-1,))
    return flat.reshape(u.shape)


def _solve_diagonal(diagonal: Array, rhs: Array) -> Array:
    """Solve against a diagonal stored either in coefficient or flattened shape."""
    if diagonal.shape == rhs.shape:
        return rhs / diagonal
    flat = rhs.reshape((-1,)) / diagonal.reshape((-1,))
    return flat.reshape(rhs.shape)


def _array_diagonal(arr: Array) -> Array | None:
    """Return the main diagonal only when a dense array is actually diagonal."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    diag = jnp.diag(arr)
    if bool(jnp.allclose(arr, jnp.diag(diag), atol=1e-12)):
        return diag
    return None


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
        return _array_diagonal(obj.todense())
    if isinstance(obj, TPMatrices):
        return _sum_diagonals(cast(list[GalerkinOperator], list(obj.tpmats)))
    if isinstance(obj, TPMatrix):
        return _tpmatrix_diagonal(obj)
    if isinstance(obj, TensorMatrix):
        return operator_diagonal(obj.data)
    return _array_diagonal(jnp.asarray(obj))


def apply_operator(op: GalerkinOperatorLike | None, u: Array) -> Array:
    """Apply a Galerkin operator-like object to a coefficient array."""
    if op is None:
        return jnp.zeros_like(u)
    if isinstance(op, list):
        items = cast(list[GalerkinOperator], op)
        applied = [apply_operator(item, u) for item in items]
        return jnp.sum(jnp.stack(applied), axis=0)
    if isinstance(op, TPMatrices | TPMatrix):
        return op @ u
    if isinstance(op, TensorMatrix):
        return op @ u
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
        return operator_to_dense(op.data)
    return jnp.asarray(op)


def solve_operator(op: GalerkinOperatorLike, rhs: Array) -> Array:
    """Solve a linear system represented by ``op`` against ``rhs``."""
    if isinstance(op, list):
        items = cast(list[GalerkinOperator], op)
        if len(items) == 1:
            return solve_operator(items[0], rhs)

    if isinstance(op, TPMatrices | TPMatrix | TensorMatrix):
        return op.solve(rhs)

    if isinstance(op, DiaMatrix | Matrix):
        b = rhs.reshape((-1,))
        return op.solve(b).reshape(rhs.shape)

    mat = operator_to_dense(op)
    if mat.ndim == rhs.ndim:
        return _solve_diagonal(mat, rhs)
    if mat.ndim != 2:
        raise ValueError("Can only solve systems with rank-2 operators")
    b = rhs.reshape((-1,))
    x = jnp.linalg.solve(mat, b)
    return x.reshape(rhs.shape)


def warm_operator_solve_cache(op: GalerkinOperatorLike | None) -> None:
    """Warm native solver caches for operators that support factorization."""
    if op is None:
        return
    if isinstance(op, list):
        for item in cast(list[GalerkinOperator], op):
            warm_operator_solve_cache(item)
        return
    lu_factor = getattr(op, "lu_factor", None)
    if lu_factor is None:
        return
    try:
        lu_factor()
    except (SolverNotApplicable, ValueError, TypeError, RuntimeError):
        # Some structured sums intentionally fall back from factored solves to
        # dense/Kronecker solves. Leave those paths to the operator's solve().
        return


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


def assemble_linear_term(expr: sp.Expr, *, sparse: bool, sparse_tol: int) -> LinearTerm:
    """Assemble a linear weak-form expression into reusable operator data."""
    if sp.sympify(expr) == 0:
        return LinearTerm()

    linear_form = cast(
        GalerkinAssembledForm, inner(expr, sparse=sparse, sparse_tol=sparse_tol)
    )
    operator, forcing = split_operator_and_forcing(linear_form)
    return LinearTerm(
        operator=operator,
        forcing=forcing,
        diagonal=operator_diagonal(operator),
    )
