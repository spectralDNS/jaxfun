# Solve Poisson's equation in 2D
import operator
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sympy as sp
from jax import Array, device_put, lax
from jax._src import dtypes
from jax._src.lax import lax as lax_internal
from jax.tree_util import Partial, tree_leaves, tree_map, tree_reduce, tree_structure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import sparse as scipy_sparse

from jaxfun.arguments import TestFunction, TrialFunction
from jaxfun.basespace import n
from jaxfun.Chebyshev import Chebyshev
from jaxfun.functionspace import FunctionSpace
from jaxfun.inner import inner
from jaxfun.Legendre import Legendre
from jaxfun.operators import Div, Grad
from jaxfun.tensorproductspace import (
    TensorProduct,
    TPMatrices,
    TPMatrix,
    tpmats_to_scipy_kron,
)
from jaxfun.utils.common import lambdify, ulp

M = 20
bcs = {"left": {"D": 0}, "right": {"D": 0}}
D = FunctionSpace(M, Chebyshev, bcs, name="D", fun_str="psi")
T = TensorProduct((D, D), name="T")
v = TestFunction(T, name="v")
u = TrialFunction(T, name="u")
alpha = 1.0

# Method of manufactured solution
x, y = T.system.base_scalars()
ue = (1 - x**2) * (1 - y**2) * sp.exp(sp.cos(sp.pi * x)) * sp.exp(sp.sin(sp.pi * y))

# A, b = inner(-Dot(Grad(u), Grad(v)) - v * Div(Grad(ue)), sparse=False)
A, b = inner(
    alpha * u * v - v * Div(Grad(u)) - (alpha * ue * v - v * Div(Grad(ue))),
    sparse=False,
)


class Helmholtz2D:
    def __init__(self, A):
        self.tpmats = A


class ADD:
    def __init__(self, a):
        self.N = a.shape[0]
        self.k = jnp.arange(self.N)
        self._transpose = False
        self.mat = a
        self.scale = -1 if a[0, 0] < 0 else 1

    def diagonal(self):
        return self.mat.diagonal()

    @property
    def T(self):
        self._transpose = not self._transpose
        return self

    @partial(jax.jit, static_argnums=0)
    def matvec(self, u: Array):
        N: int = len(u)
        a0 = (2 * self.k) * u

        # cumsum is 3-4 times slower than the scan below and does not work for odd N
        # aoe = (jnp.cumsum(u.reshape(-1, 2)[::-1], axis=0)[::-1]).reshape(u.shape)
        # return jnp.pi * (self.k + 1) * (a0 - 4 * aoe)
        def inner_loop(
            carry: tuple[float, float], i: int
        ) -> tuple[tuple[float, float], Array]:
            t0: float = carry[0] + u[i]
            t1: float = carry[1] + u[i - 1]
            return (t0, t1), jnp.array([t0, t1])

        _, xs = lax.scan(inner_loop, (u[-1], u[-2]), jnp.arange(N - 3, 0, -2))
        xs = jnp.hstack((xs.ravel(), xs.ravel()[-2] + u[0])) if N % 2 else xs.ravel()
        z = jnp.hstack((jnp.array([u[-1], u[-2]]), xs))
        return (self.scale * jnp.pi) * (self.k + 1) * (a0 + 4 * z[::-1])

    @partial(jax.jit, static_argnums=0)
    def __matmul__(self, u):
        return jax.vmap(self.matvec, in_axes=1, out_axes=1)(u)

    @partial(jax.jit, static_argnums=0)
    def __rmatmul__(self, u):
        return jax.vmap(self.matvec, in_axes=0, out_axes=0)(u)


class BDD:
    def __init__(self, a):
        self.N = a.shape[0]
        self.ck = jnp.pi * jnp.hstack((1.5 * jnp.ones(1), jnp.ones(self.N - 1)))
        self._transpose = False
        self.mat = a
        self.scale = -1 if a[0, 0] < 0 else 1

    def diagonal(self):
        return self.mat.diagonal()

    @property
    def T(self):
        self._transpose = not self._transpose
        return self

    @partial(jax.jit, static_argnums=0)
    def matvec(self, u: Array):
        a0 = jnp.hstack((-jnp.pi / 2 * u[2:], jnp.zeros(2))) + jnp.hstack(
            (jnp.zeros(2), -jnp.pi / 2 * u[:-2])
        )
        return self.scale * (a0 + self.ck * u)

        # N: int = len(u)

        # def inner_loop(carry: float, i: int) -> tuple[float, float]:
        #    t: float = (-u[i - 2] + 2 * u[i] - u[i + 2]) / 2
        #    return carry, t

        # _, xs = jax.lax.scan(inner_loop, 0.0, jnp.arange(2, N - 2))
        #
        # u0: float = 1.5 * u[0] - u[2] / 2
        # u1: float = u[1] - u[3] / 2
        # uNm2: float = -u[-4] / 2 + u[-2]
        # uNm1: float = -u[-3] / 2 + u[-1]
        # return jnp.hstack((jnp.array([u0, u1]), xs, jnp.array([uNm2, uNm1]))) * jnp.pi

    @partial(jax.jit, static_argnums=0)
    def __matmul__(self, u):
        return jax.vmap(self.matvec, in_axes=1, out_axes=1)(u)

    @partial(jax.jit, static_argnums=0)
    def __rmatmul__(self, u):
        return jax.vmap(self.matvec, in_axes=0, out_axes=0)(u)


def _vdot_real_tree(x, y):
    return sum(tree_leaves(tree_map(_vdot_real_part, x, y)))


def _vdot_tree(x, y):
    return sum(
        tree_leaves(tree_map(partial(jnp.vdot, precision=lax.Precision.HIGHEST), x, y))
    )


def _mul(scalar, tree):
    return tree_map(partial(operator.mul, scalar), tree)


_add = partial(tree_map, operator.add)
_sub = partial(tree_map, operator.sub)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)


@Partial
def _identity(x):
    return x


# aliases for working with pytrees
def _vdot_real_part(x, y):
    """Vector dot-product guaranteed to have a real valued result despite
    possibly complex input. Thus neglects the real-imaginary cross-terms.
    The result is a real float.
    """
    # all our uses of vdot() in CG are for computing an operator of the form
    #  z^H M z
    #  where M is positive definite and Hermitian, so the result is
    # real valued:
    # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
    result = _vdot(x.real, y.real)
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        result += _vdot(x.imag, y.imag)
    return result


def _bicgstab_solve(A, b, maxiter=100, tol=1e-5, atol=0.0, M=_identity):
    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.bicgstab
    # bs = _vdot_real_tree(b, b)
    bs = jnp.tensordot(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))
    x0 = jnp.zeros_like(b)

    b, x0 = device_put((b, x0))

    # https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB

    def cond_fun(value):
        x, r, *_, k = value
        # rs = _vdot_real_tree(r, r)
        rs = jnp.tensordot(r, r)
        # the last condition checks breakdown
        return (rs > atol2) & (k < maxiter) & (k >= 0)

    @jax.jit
    def body_fun(value):
        x, r, rhat, alpha, omega, rho, p, q, k = value
        # rho_ = _vdot_tree(rhat, r)
        rho_ = jnp.tensordot(rhat, r)
        beta = rho_ / rho * alpha / omega
        # p_ = _add(r, _mul(beta, _sub(p, _mul(omega, q))))
        p_ = r + beta * (p - omega * q)
        phat = M(p_)
        q_ = A(phat)
        # alpha_ = rho_ / _vdot_tree(rhat, q_)
        alpha_ = rho_ / jnp.tensordot(rhat, q_)
        # s = _sub(r, _mul(alpha_, q_))
        s = r - alpha_ * q_
        # exit_early = _vdot_real_tree(s, s) < atol2
        exit_early = jnp.tensordot(s, s) < atol2
        shat = M(s)
        t = A(shat)
        # omega_ = _vdot_tree(t, s) / _vdot_tree(t, t)  # make cases?
        omega_ = jnp.tensordot(t, s) / jnp.tensordot(t, t)
        x_ = tree_map(
            partial(jnp.where, exit_early),
            _add(x, _mul(alpha_, phat)),
            _add(x, _add(_mul(alpha_, phat), _mul(omega_, shat))),
        )
        r_ = tree_map(partial(jnp.where, exit_early), s, _sub(s, _mul(omega_, t)))
        # x_ = partial(jnp.where, exit_early)(
        #    x + alpha_ * phat, x + alpha_ * phat + omega_ * shat
        # )
        # r_ = partial(jnp.where, exit_early)(s, s - omega_ * t)
        k_ = jnp.where((omega_ == 0) | (alpha_ == 0), -11, k + 1)
        k_ = jnp.where((rho_ == 0), -10, k_)
        return x_, r_, rhat, alpha_, omega_, rho_, p_, q_, k_

    # r0 = _sub(b, A(x0))
    r0 = b - A(x0)
    # rho0 = alpha0 = omega0 = lax_internal._convert_element_type(
    #    1, *dtypes._lattice_result_type(*tree_leaves(b))
    # )
    rho0 = alpha0 = omega0 = jnp.array(1.0)
    initial_value = (x0, r0, r0, alpha0, omega0, rho0, r0, r0, 0)

    x_final, *_ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final


# jax can only do kron for dense matrices
# C = jnp.kron(*A[0].mats) + jnp.kron(*A[1].mats)
# uh = jnp.linalg.solve(C, b.flatten()).reshape(b.shape)

# Alternative scipy sparse implementation
# A0 = tpmats_to_scipy_kron(A)
# uh = jnp.array(scipy_sparse.linalg.spsolve(A0, b.flatten()).reshape(b.shape))

# Alternative iterative solver
A0 = TPMatrices(A)
uh = jax.scipy.sparse.linalg.bicgstab(A0, b, maxiter=500, M=A0.precond, tol=ulp(10))[0]
bicgstab = partial(_bicgstab_solve, maxiter=500, tol=ulp(10))
uh2 = lax.custom_linear_solve(A0, b, solve=bicgstab, symmetric=True)

H = [
    TPMatrix((ADD(A[0].mats[0]), BDD(A[0].mats[1])), 1, D, D),
    TPMatrix((BDD(A[1].mats[0]), ADD(A[1].mats[1])), 1, D, D),
    TPMatrix((BDD(A[2].mats[0]), BDD(A[2].mats[1])), 1, D, D), 
]
H0 = TPMatrices(H)
# uh3 = jax.scipy.sparse.linalg.bicgstab(H0, b, maxiter=500, tol=ulp(10))[0]
uh3 = lax.custom_linear_solve(H0, b, solve=bicgstab)


print(jnp.linalg.norm(uh - uh2), jnp.linalg.norm(uh - uh3))

N = 100
uj = T.backward(uh, kind="uniform", N=(N, N))
xj = T.mesh(kind="uniform", N=(N, N), broadcast=True)
uej = lambdify((x, y), ue)(*xj)

error = jnp.linalg.norm(uj - uej) / N
if "pytest" in os.environ:
    assert error < ulp(1000), error
    sys.exit(1)

print("Error =", error)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
xj = T.mesh(kind="uniform", N=(N, N), broadcast=False)
ax1.contourf(xj[0], xj[1], uj)
ax2.contourf(xj[0], xj[1], uej)
ax2.set_autoscalex_on(False)
c3 = ax3.contourf(xj[0], xj[1], uej - uj)
axins = inset_axes(
    ax3,
    width="5%",  # width = 10% of parent_bbox width
    height="100%",  # height : 50%
    loc=6,
    bbox_to_anchor=(1.05, 0.0, 1, 1),
    bbox_transform=ax3.transAxes,
    borderpad=0,
)
cbar = plt.colorbar(c3, cax=axins)
ax1.set_title("Jaxfun")
ax2.set_title("Exact")
ax3.set_title("Error")
plt.show()
