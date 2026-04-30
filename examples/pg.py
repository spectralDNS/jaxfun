import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp

from jaxfun.galerkin import Composite
from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.inner import inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.la import Matrix
from jaxfun.utils.common import ulp

n = sp.Symbol("n", positive=True, integer=True)
N = 50
biharmonic = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
dirichlet = {"left": {"D": 0}, "right": {"D": 0}}
C = Composite(N, Chebyshev, dirichlet, name="C")
v = Matrix(jax.random.normal(jax.random.PRNGKey(1), shape=(N, N)))
g = C.S @ v @ C.S.T
g1 = C.apply_stencil_galerkin(v)
D = C.S
vn = v.todense()
gn = D @ vn @ D.T
assert jnp.linalg.norm(gn - g.data) < 1e-7
assert jnp.linalg.norm(gn - g1.data) < 1e-7
# Galerkin (dense)
u = TrialFunction(C, name="u")
v = TestFunction(C, name="v")
x = C.system.x
D = inner(v * u.diff(x, 2), sparse=True, sparse_tol=1000)
# Petrov-Galerkin method (https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
G = Composite(N, Chebyshev, dirichlet, scaling=n + 1, name="G")
PG = Composite(
    N + 2,
    Chebyshev,
    biharmonic,
    stencil={
        0: 1 / (2 * sp.pi * (n + 1) * (n + 2)),
        2: -1 / (sp.pi * (n**2 + 4 * n + 3)),
        4: 1 / (2 * sp.pi * (n + 2) * (n + 3)),
    },
    name="PG",
)
L = Composite(N, Legendre, dirichlet, scaling=n + 1, name="L")
LG = Composite(
    N + 2,
    Legendre,
    biharmonic,
    stencil={
        0: 1 / (2 * (2 * n + 3)),
        2: -(2 * n + 5) / (2 * n + 7) / (2 * n + 3),
        4: 1 / (2 * (2 * n + 7)),
    },
    name="LG",
)
A0 = inner(
    TestFunction(PG) * TrialFunction(G).diff(x, 2),
    sparse=True,
    sparse_tol=1000,
)  # bidiagonal
A1 = inner(
    TestFunction(LG) * TrialFunction(L).diff(x, 2),
    sparse=True,
    sparse_tol=1000,
)  # bidiagonal

if "PYTEST" in os.environ:
    assert A1.nnz == 2 * N - 6
    assert A0.nnz == 2 * N - 6
    sys.exit(1)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
ax0.spy(D.todense())
ax0.set_title("Galerkin Cheb")
ax1.spy(A0.todense())
ax1.set_title("PG Chebyshev")
ax2.spy(A1.todense())
ax2.set_title("PG Legendre")
plt.show()
