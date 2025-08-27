from numbers import Number

import jax
import jax.numpy as jnp
import sympy as sp

from jaxfun.utils import lambdify

from .module import FlaxFunction


def DirichletBC(
    u: FlaxFunction, bnd_mesh: jax.Array, *bcs: sp.Expr | Number
) -> jax.Array:
    g = []
    for b in bcs:
        s = u.get_args(Cartesian=False)
        if isinstance(b, Number):
            g.append(float(b) * jnp.ones(bnd_mesh.shape[0]))
        else:
            g.append(lambdify(s, b)(*bnd_mesh.T))
    return jnp.array(g).T
