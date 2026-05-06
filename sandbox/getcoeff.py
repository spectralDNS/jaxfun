# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin import (
    FunctionSpace,
    JAXFunction,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    VectorTensorProductSpace,
)
from jaxfun.operators import *  # noqa: F403
from jaxfun.pinns import FlaxFunction

N = 8
V = FunctionSpace(
    N, Legendre.Legendre, bcs={"left": {"D": 1}, "right": {"D": 1}}, name="V"
)
# V = FunctionSpace(N, Chebyshev.Chebyshev, name="V", domain=(-2, 2))
# V = FunctionSpace(N, Legendre.Legendre, name="V")
# V = FunctionSpace(N, Fourier.Fourier, name="V")

case = 1

if case == 1:
    T = TensorProduct(V, V, name="T")
    x, y = T.system.base_scalars()
    ue = sp.sin(x) * sp.sin(y)
    w = JAXFunction(ue, T, name="w")
    ff = FlaxFunction(T, name="ff")
    ff.module.kernel = w.array
    v = TestFunction(T, name="v")
    u = TrialFunction(T, name="u")
    xj = T.mesh()
    TO = T.get_orthogonal()

elif case == 2:
    x = V.system.x
    ue = -sp.cos(x)
    # ff = FlaxFunction(V, name="ff")
    # ff.module.kernel = w.array[None, :]
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u")
    xj = V.mesh()
    w = JAXFunction(ue, V, name="w")


elif case == 3:
    T = TensorProduct(V, V, name="T")
    V = VectorTensorProductSpace(T, name="V")
    w = JAXFunction(jnp.ones(V.num_dofs), V, name="w")
    ff = FlaxFunction(V, name="ff")
    ff.module.kernel = w.array
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u")
    x, y = T.system.base_scalars()
    xj = T.mesh()

elif case == 4:
    T = TensorProduct(V, V, name="T")
    w = JAXFunction(jnp.ones(T.num_dofs), T, name="w")
    # ff = FlaxFunction(V, name="ff")
    # ff.module.kernel = w.array[None, :]
    v = TestFunction(T, name="v")
    u = TrialFunction(T, name="u")
    x, y = T.system.base_scalars()
    xj = T.mesh()
