"""Pinned (constrained) linear system.

Produced by :meth:`DiaMatrix.pin` and :meth:`Matrix.pin`.  Encapsulates
both the row-substituted matrix and the constraint values so that the RHS
can be consistently modified and the LU factorisation reused across calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    import jax

    from jaxfun.la import DiaMatrix, Matrix

    Array = jax.Array


@nnx.dataclass
class PinnedSystem(nnx.Pytree):
    """A linear system with pinned (constrained) degrees of freedom.

    Created by calling :meth:`~jaxfun.la.DiaMatrix.pin` or
    :meth:`~jaxfun.la.Matrix.pin` on a matrix.  The underlying matrix has
    its pinned rows replaced by identity rows (``A[i, :] = e_i``), and the
    LU factorisation of that modified matrix is cached on the first
    :meth:`solve` call and reused thereafter.

    Attributes:
        matrix: The row-substituted :class:`Matrix` or :class:`DiaMatrix`.
            Treated as a JAX pytree child so the enclosed arrays are visible
            to :func:`jax.jit`, :func:`jax.grad`, etc.
        constraints: Immutable tuple of ``(index, value)`` pairs recording
            the pinned DOFs.  Stored as static pytree metadata (not traced),
            because constraint indices and values are compile-time constants.

    Usage::

        # One-time setup
        A_sys = A.pin({0: 0.0})  # row 0 → identity, pin value 0
        A_sys.lu_factor()  # optional explicit warm-up

        # Repeated solves
        for b in rhs_sequence:
            x = A_sys.solve(b)  # modifies b[0] then solves

    Args:
        matrix: The row-substituted :class:`Matrix` or :class:`DiaMatrix`.
        constraints: Mapping from DOF index to pinned value
            (e.g. ``{0: 0.0, -1: 1.0}``).  Converted to a tuple internally.
    """

    matrix: Matrix | DiaMatrix
    constraints: tuple[tuple[int, float], ...]

    def __init__(
        self,
        matrix: Matrix | DiaMatrix,
        constraints: dict[int, float],
    ) -> None:
        self.matrix = matrix
        # Store as a sorted tuple so the pytree treedef is deterministic and
        # hashable (dicts are neither).
        self.constraints = tuple(sorted(constraints.items()))

    @property
    def shape(self) -> tuple[int, int]:
        """``(n, n)`` shape of the underlying matrix."""
        return self.matrix.shape

    def lu_factor(self):
        """Pre-compute and cache the LU factorisation.

        Calling this explicitly before the loop avoids paying the
        factorisation cost on the first :meth:`solve` call.
        """
        return self.matrix.lu_factor()

    def fix_rhs(self, b: Array, axis: int = 0) -> Array:
        """Return ``b`` with constraint values inserted at pinned positions.

        For each ``(index, value)`` pair in the constraints, the slice of
        ``b`` at position ``index`` along ``axis`` is set to ``value``.

        Args:
            b:    Right-hand side array.
            axis: Axis of ``b`` corresponding to the DOF dimension.

        Returns:
            Modified array with the same shape as ``b``.

        Examples::

            # 1-D:  b[0] = 0.0
            b_mod = sys.fix_rhs(b)

            # 2-D row-major batch (k, n), axis=1:  b[:, 0] = 0.0
            b_mod = sys.fix_rhs(b, axis=1)
        """
        axis = axis % b.ndim
        b_moved = jnp.moveaxis(b, axis, 0)  # pinned axis → front
        for idx, val in self.constraints:
            b_moved = b_moved.at[idx].set(val)
        return jnp.moveaxis(b_moved, 0, axis)

    def solve(self, b: Array, axis: int = 0) -> Array:
        """Modify the RHS and solve the pinned system.

        Applies :meth:`fix_rhs` to ``b`` then solves using the cached LU
        factorisation of the modified matrix.

        Args:
            b:    Right-hand side array.  ``b.shape[axis]`` must equal ``n``.
            axis: Axis of ``b`` along which the system is solved.  All other
                  axes are treated as independent batch dimensions.  The
                  output has the same shape as ``b``.

        Returns:
            Solution array with the same shape as ``b``.
        """
        b_mod = self.fix_rhs(b, axis=axis)
        return self.matrix.solve(b_mod, axis=axis)

    def __repr__(self) -> str:
        pins = ", ".join(f"{k}={v}" for k, v in self.constraints)
        return f"PinnedSystem(shape={self.shape}, constraints={{{pins}}})"
