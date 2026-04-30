import jax

# ruff: noqa: E731
import jax.numpy as jnp
import sympy as sp
from la import (
    TDMA_LU,
    TDMA_O_LU,
    TDMA_BSolve,
    TDMA_BSolve2,
    TDMA_Fwd,
    TDMA_Fwd2,
    TDMA_O_Solve,
    TDMA_O_solve,
    TDMA_Sol,
    TDMA_Sol2,
    TDMA_solve,
    tridiagonal_solve_jax,
)

from jaxfun import *

x = sp.Symbol("x")

# N = 8
# C = Composite(Chebyshev.Chebyshev, N, {'left': {'D': 0}, 'right': {'D': 0}}, domain=(0, 4))
# u = TrialFunction(x, C)
# v = TestFunction(x, C)
# A = inner(u*v)
# ld, d, ud = A.diagonal(-2), A.diagonal(), A.diagonal(2)
# nld, nd = TDMA_LU(ld, d, ud, N-1)
# u0 = jnp.ones(N-1)
# uj = TDMA_solve(u0, nld, nd, ud, N-1)
# A = common.eliminate_near_zeros(A, tol=1000)
# lu_piv = jax.scipy.linalg.lu_factor(A)
# u1 = jax.scipy.linalg.lu_solve(lu_piv, u0)
# assert jnp.linalg.norm(u1-uj) < 1e-10
#
# As = common.tosparse(A)
# sparse.linalg.linalg.splu(As)

N = 8
As = la.diags(
    [jnp.array([1.0]), jnp.array([-2.0]), jnp.array([1.0])],
    offsets=(-1, 0, 1),
    shape=(N, N),
)
Af = As.todense()
ld, d, ud = Af.diagonal(-1), Af.diagonal(), Af.diagonal(1)

dl = jnp.hstack((jnp.array([0]), ld))
du = jnp.hstack((ud, jnp.array([0])))

u0 = jnp.ones(N)
u2 = jax.lax.linalg.tridiagonal_solve(dl, d, du, u0[:, None])
nld, nd = TDMA_O_LU(ld, d, ud, N)
uj = TDMA_O_solve(u0, nld, nd, ud, N)
uj2 = TDMA_O_Solve(u0, ld, d, ud)

a, b, c = dl, d, du
c_ = TDMA_Fwd(a, b, c)
x_ = TDMA_BSolve(u0, a, b, c_)

a0, b, c0 = ld, d, ud  # short ld, ud
_a, _b = TDMA_Fwd2(a0, b, c0)
x2_ = TDMA_BSolve2(u0, _a, _b, c0)  # long du

u3 = tridiagonal_solve_jax(dl, d, du, u0[:, None])
