# Solve Poisson's equation in 2D using 2 local devices and sharding.
# ruff: noqa: E402
import os

import pytest

pytestmark = pytest.mark.spmd

# Note that jax.config.update("jax_num_cpu_devices", 2) is set in conftest.py
# for this example to work with pytest and github actions. To run this example
# locally, make sure to set the number of CPU devices to 2 or more here or via
# environment variable:
# import jax
# jax.config.update("jax_num_cpu_devices", 2)
# The example should also run with distributed devices and not just local, but
# that requires additional setup using jax.distributed.initialize.

import jax.numpy as jnp
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction, x, y
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.tensorproductspace import TensorProduct
from jaxfun.operators import Div, Grad
from jaxfun.utils.common import lambdify, ulp

ue = (1 - y**2) * (sp.cos(2 * x)) * sp.exp(sp.cos(sp.pi * y))

M, N = 60, 20
bcs = {"left": {"D": ue.subs(y, -1)}, "right": {"D": ue.subs(y, 1)}}
D = FunctionSpace(M, Legendre, bcs, name="D", fun_str="psi")
F = FunctionSpace(N, Fourier, name="F", fun_str="E")
T = TensorProduct(F, D, name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")

# Method of manufactured solution
x, y = T.system.base_scalars()
ue = T.system.expr_psi_to_base_scalar(ue)

# Returned b is sharded.
A, b = inner(v * Div(Grad(u)) - v * Div(Grad(ue)), sparse=True)

# Parallel solve across all devices. uh will be sharded as b.
uh = A.solve(b)

N = 100
uj = T.backward(uh, N=(2 * N, 2 * M))
xj = T.mesh(N=(2 * N, 2 * M), broadcast=True)
uej = lambdify((x, y), ue)(*xj)
error = jnp.linalg.norm(uj - uej) / N

uhn = T.forward(uj)

error_fwd = jnp.linalg.norm(uhn - uh) / N

if "PYTEST" in os.environ:
    assert error < ulp(100), error
    assert error_fwd < ulp(1000), error_fwd

print("Error roundtrip transforms =", error_fwd)
print("Error =", error)
