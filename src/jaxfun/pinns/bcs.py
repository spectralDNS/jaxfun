"""Boundary condition utilities for PINNs.

Currently provides:
    - DirichletBC: Assemble Dirichlet boundary target values for one or
      multiple (component) expressions over a boundary mesh.
"""

import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun.utils import lambdify

from .module import FlaxFunction


def DirichletBC(
    u: FlaxFunction, bnd_mesh: jax.Array, *bcs: sp.Expr | float
) -> jax.Array:
    """Assemble Dirichlet boundary values for a field/function.

    Each entry in bcs is either:
      * A scalar Number (broadcast over all boundary points), or
      * A SymPy expression in the coordinate symbols (and optionally time)
        returned by u.get_args(Cartesian=False).

    The function evaluates every boundary expression at the coordinates
    in bnd_mesh and stacks the results column-wise.

    Args:
        u: FlaxFunction describing the unknown (used to get symbolic args).
        bnd_mesh: Boundary coordinates with shape (N, d). Columns must match
            the order of u.get_args(Cartesian=False) (spatial [+ time]).
        *bcs: One or more boundary value specifications (Number or SymPy
            Expr). For vector fields pass one expression/number per
            component (in desired order).

    Returns:
        jax.Array of shape (N, len(bcs)) containing boundary target values.

    Examples:
        Scalar field:
            gvals = DirichletBC(u, xbnd, 0.0)
        Vector field with two components:
            gvals = DirichletBC(v, xbnd, 0.0, sp.sin(x))
    """
    g = []
    for b in bcs:
        s = u.get_args(Cartesian=False)
        if isinstance(b, sp.Number | float | int):  # overkill check for ty
            g.append(float(b) * jnp.ones(bnd_mesh.shape[0]))
        else:
            g.append(lambdify(s, b)(*bnd_mesh.T))
    return jnp.array(g).T
