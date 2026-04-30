import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import project
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.utils.common import lambdify

M = 20
D = FunctionSpace(M, Chebyshev)
#F = FunctionSpace(M, Chebyshev)
F = FunctionSpace(M, Fourier)
T = TensorProduct(F, D, name="T")

x, y = T.system.base_scalars()
ue = sp.cos(2 * x) * (1 - y**2)
uh = project(ue, T)

uj = T.backward(uh, kind="quadrature", N=(100, 100))
xj = T.mesh(N=(100, 100), kind="quadrature", broadcast=True)
uej = lambdify((x, y), ue)(*xj)

assert jnp.linalg.norm(uj - uej) < 1e-13

uj = T.backward(uh, kind="uniform", N=(100, 100))
xj = T.mesh(kind="uniform", N=(100, 100), broadcast=True)
uej = lambdify((x, y), ue)(*xj)

assert jnp.linalg.norm(uj - uej) < 1e-6, f"Error was {jnp.linalg.norm(uj - uej)}"
