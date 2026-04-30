# ruff: noqa: I001, E402
import jax
import jax.numpy as jnp
from flax import nnx

jax.config.update("jax_enable_x64", True)

import sympy as sp

from jaxfun import Cross, Curl, Div, Dot, Grad, Outer, get_CoordSys
from jaxfun.arguments import Constant, Identity
from jaxfun.pinns.module import (
    FlaxFunction,
    MLPSpace,
)

r, theta, z = sp.symbols("r,theta,z", real=True, positive=True)
C = get_CoordSys(
    "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
)
V = MLPSpace([8], dims=3, rank=1, system=C)  # Vector space for velocity
Q = MLPSpace([8], dims=3, rank=0, system=C)  # Scalar space for pressure

u = FlaxFunction(V, "u", rngs=nnx.Rngs(2002))
p = FlaxFunction(Q, "p", rngs=nnx.Rngs(2002))

Re = 10.0  # Define Reynolds number
nu = Constant(
    "nu", 2.0 / Re
)  # Define kinematic viscosity. A number works as well, but the Constant prints better.
# R1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
I = Identity(V.system)
R1 = Div(Outer(u, u)) - Div(
    nu * (Grad(u) + Grad(u).T - sp.Rational(2, 3) * Div(u) * I) + p * I
)  # Alternative form
# R1 = Cross(Curl(u), u) + sp.S.Half*Grad(Dot(u, u)) - nu * Div(Grad(u)) + Grad(p)
R2 = Div(u)
