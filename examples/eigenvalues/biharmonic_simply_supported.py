# ruff: noqa: E402
import os
import sys

import jax

if "PYTEST" not in os.environ:
    jax.config.update("jax_enable_x64", True)

from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

from jaxfun.galerkin import (
    Chebyshev,  # noqa: F401
    FunctionSpace,
    Legendre,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.composite import BoundaryConditions
from jaxfun.la import TPMatrices
from jaxfun.operators import Div, Grad
from jaxfun.utils import ulp

bcs = BoundaryConditions({"left": {"D": 0, "N2": 0}, "right": {"D": 0, "N2": 0}})
family = Chebyshev.Chebyshev  # Legendre.Legendre
kind = "GR"  # "G", "PG"

# Only the unweighted (Legendre) Galerkin matrices are symmetric; the Chebyshev
# weight makes A non-symmetric.
symmetric = family is Legendre.Legendre and kind == "G"

if "PYTEST" in os.environ:
    N = 24
    k = 6
else:
    N = 64
    k = 150

B = FunctionSpace(N, family, bcs=bcs, domain=(0, 1))
T = TensorProduct(B, B, name="T")
P = T.get_testspace(kind, name="P")
x, y = T.system.base_scalars()
u = TrialFunction(T, name="u")
v = TestFunction(P, name="v")
A = inner(v, Div(Grad(Div(Grad(u)))), sparse=True)  # P⊗M + 2S⊗S + M⊗P
M = inner(v, u, sparse=True)

# A has three distinct matrices per axis and cannot be diagonalized, but its
# separable part P⊗M + M⊗P can. For Legendre, int u_xx u_yy = int u_xy^2 >= 0
# gives Asep <= A <= 2 Asep. The Chebyshev weight breaks that proof, but the
# spectrum of Asep^{-1} A still lies in [1, 2] up to small imaginary parts, so
# either way the preconditioned solve converges in ~20 iterations.
Asep = inner(v, u.diff(x, 4) + u.diff(y, 4), sparse=True)
Asep_solve = cast(TPMatrices, Asep).lu_factor().solve

shape = T.num_dofs
ndofs = int(np.prod(shape))
G = P.basespaces[0]
# d = (G.S @ jnp.linalg.pinv(B.S.todense())).diagonal()
d = G.S.diagonal()
D = jnp.outer(1 / d, 1 / d)

matvec_A = jax.jit(lambda z: (D * cast(TPMatrices, A)(z.reshape(shape))).ravel())
matvec_M = jax.jit(lambda z: (D * cast(TPMatrices, M)(z.reshape(shape))).ravel())
precond = jax.jit(lambda z: Asep_solve(z.reshape(shape) / D).ravel())


@jax.jit
def solve_A(b):
    if symmetric:
        return jax.scipy.sparse.linalg.cg(
            matvec_A, b, M=precond, tol=ulp(10), maxiter=100
        )[0]
    return jax.scipy.sparse.linalg.bicgstab(
        matvec_A, b, M=precond, tol=ulp(10), maxiter=100
    )[0]


def linop(f):
    return LinearOperator(
        shape=(ndofs, ndofs),
        dtype=float,
        matvec=lambda z: np.asarray(f(jnp.asarray(z, dtype=float))),  # ty: ignore[unknown-argument]
    )


# Iterate with the compact operator A^{-1} M, whose largest eigenvalues
# mu = 1/lambda come out to relative precision. A dense eig of M^{-1} A has
# errors ~ eps * lambda_max ~ eps * N^8 instead (~1e-8 at N = 64).
# Ask for a few extra: Lanczos/Arnoldi can miss copies of a multiple eigenvalue
# at the cutoff (here m^2 + n^2 = 205 is 4-fold and straddles k = 150).
if symmetric:
    # Shift-invert Lanczos about 0 for A x = lambda M x.
    lam = eigsh(
        linop(matvec_A),
        k=k + 10,
        M=linop(matvec_M),
        sigma=0,
        OPinv=linop(solve_A),
        which="LM",
        tol=ulp(10),
        return_eigenvectors=False,
    )
else:
    # Arnoldi on A^{-1} M directly; eigs' shift-invert mode needs M symmetric.
    mu = eigs(
        linop(lambda z: solve_A(matvec_M(z))),
        k=k + 10,
        which="LM",
        tol=ulp(10),
        return_eigenvectors=False,
    )
    lam = (1 / mu).real
lam = np.sort(lam)[:k]

print("lambda_1 =", repr(lam[0]))

m = np.arange(1, 100)[:, None]
n = np.arange(1, 100)[None, :]
exact = np.sort((((m**2 + n**2) * np.pi**2) ** 2).ravel())[:k]
err = np.abs(lam - exact) / exact

if "PYTEST" in os.environ:
    assert err.max() < ulp(1000), err.max()
    sys.exit(0)

print("max relative error:", err.max())
plt.semilogy(err, "o")
plt.ylabel("relative error")
plt.show()
