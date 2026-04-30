# ruff: noqa: E402

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from scipy import sparse as scipy_sparse

from jaxfun.coordinates import CartCoordSys
from jaxfun.galerkin import (
    Fourier,
    FunctionSpace,
    JAXFunction,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.tensorproductspace import TPMatrices, tpmats_to_scipy_kron
from jaxfun.operators import Derivative

# from jaxfun.utils.common import Domain

t, x = sp.symbols("t x", real=True)

A = 25
B = 16
u0 = (
    3 * A**2 / sp.cosh(sp.S.Half * A * (x - sp.pi + 2)) ** 2
    + 3 * B**2 / sp.cosh(sp.S.Half * B * (x - sp.pi + 1)) ** 2
)
bcs = {"left": {"D": u0}}
dt = 0.0025

C = CartCoordSys("R", (t, x))
F = Fourier.Fourier(256, name="F")
L = FunctionSpace(4, Legendre.Legendre, name="L", bcs=bcs, domain=(0, dt))
T = TensorProduct(L, F, name="T", system=C)
t, x = T.system.base_scalars()

u = TrialFunction(T, name="u")
v = TestFunction(T, name="v")
ua = JAXFunction(jnp.zeros(T.shape()), T, name="ua")
raise SystemExit
eq = u.diff(t) + u.diff(x, 3) - 0.5 * Derivative(ua*ua, x)
A, b = inner(v * eq)

M = tpmats_to_scipy_kron(A)
un = jnp.array(scipy_sparse.linalg.spsolve(M, b.flatten()).reshape(b.shape))

# Assemble a boundary condition matrix for updating boundary values at each timestep.
Tf = TensorProduct(
    T.basespaces[0].basespaces[1].to_composite_like(), F, name="Tf", system=C
)
f = TrialFunction(Tf, name="f")
eq2 = f.diff(t) - f.diff(x, 2)
b11 = inner(v * eq2)
MB = tpmats_to_scipy_kron(b11)
key = list(T.bndvals.keys())[0]
xj = F.mesh()
result = [T.evaluate_mesh((jnp.array([0.0]), xj), un, use_einsum=True)[0].real]

for _ in range(1, 50):
    # New boundary condition
    u_2 = T.evaluate_mesh((jnp.array([dt]), xj), un, use_einsum=True).real
    result.append(u_2[0])
    uf = F.forward(u_2[0])[None, :]
    # New right-hand side for boundary condition update
    b = -(MB @ uf.flatten()).reshape(b.shape)
    un = jnp.array(scipy_sparse.linalg.spsolve(M, b.flatten()).reshape(b.shape))
    # Update boundary values required for evaluate_mesh
    T.bndvals[key] = uf

# Note: No need to update the domain of L since we evaluate_mesh at dt all the time,
# which corresponds to step*dt if we were to update the domain. The internal boundary
# condition is time-dependent but we are always evaluating at the correct time step,
# so the domain can remain fixed.
u_2 = T.evaluate_mesh((jnp.array([dt]), xj), un, use_einsum=True).real
result.append(u_2[0])

plt.plot(xj, jnp.array(result).T)
plt.show()
