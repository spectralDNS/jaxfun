from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
from sympy import Number

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import jacn, lambdify

n = sp.Symbol("n", integer=True, positive=True)  # index


class Domain(NamedTuple):
    lower: Number
    upper: Number


class BaseSpace:
    def __init__(
        self,
        N: int,
        domain: Domain = None,
        system: CoordSys = None,
        name: str = None,
        fun_str: str = "psi",
    ) -> None:
        from jaxfun.arguments import CartCoordSys, x

        self.N = N
        self._domain = Domain(*domain)
        self.name = name
        self.fun_str = fun_str
        self.system = CartCoordSys("N", (x,)) if system is None else system
        self.bcs = None
        self.orthogonal = self
        self.stencil = {0: 1}
        self.S = BCOO.from_scipy_sparse(scipy_sparse.diags((1,), (0,), (N, N)))

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        raise RuntimeError

    def quad_points_and_weights(self, N: int = 0) -> Array:
        raise RuntimeError

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_basis_derivative(self, X: Array, k: int = 0) -> Array:
        return jacn(self.eval_basis_functions, k)(X)

    @partial(jax.jit, static_argnums=0)
    def vandermonde(self, X: Array) -> Array:
        r"""Return pseudo-Vandermonde matrix
        
        Evaluates basis function :math:`\psi_k(x)` for all wavenumbers, and all
        ``x``. Returned Vandermonde matrix is an N x M matrix with N the length
        of ``x`` and M the number of bases.

        .. math::

            \begin{bmatrix}
                \psi_0(x_0) & \psi_1(x_0) & \ldots & \psi_{M-1}(x_0)\\
                \psi_0(x_1) & \psi_1(x_1) & \ldots & \psi_{M-1}(x_1)\\
                \vdots & \ldots \\
                \psi_{0}(x_{N-1}) & \psi_1(x_{N-1}) & \ldots & \psi_{M-1}(x_{N-1})
            \end{bmatrix}

        Parameters
        ----------
        x: Array

        """
        return self.evaluate_basis_derivative(X, 0)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, X: float, i: int) -> float:
        raise RuntimeError

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, X: float) -> Array:
        raise RuntimeError

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        xj = self.mesh(kind=kind, N=N)
        return self.evaluate(self.map_reference_domain(xj), c)

    def mass_matrix(self) -> BCOO:
        return BCOO.from_scipy_sparse(
            scipy_sparse.diags((self.norm_squared(),), (0,), shape=(self.dim, self.dim))
        )

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_galerkin(self, b: Array) -> Array:
        return b

    @partial(jax.jit, static_argnums=0)
    def apply_stencils_petrovgalerkin(self, b: Array, P: BCOO) -> Array:
        return b @ P.T

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_left(self, b: Array) -> Array:
        return b

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_right(self, a: Array) -> Array:
        return a

    @property
    def dim(self):
        return self.N

    @property
    def dims(self):
        return 1

    @property
    def rank(self):
        return 0

    @property
    def domain(self) -> Domain:
        return self._domain

    @property
    def reference_domain(self) -> Domain:
        raise RuntimeError

    @property
    def domain_factor(self) -> Number:
        a, b = self.domain
        c, d = self.reference_domain
        L = b - a
        R = d - c
        return R / L if abs(L - R) > 1e-12 else 1

    def map_expr_reference_domain(self, u: sp.Expr) -> sp.Expr:
        """Return`u(x)` mapped to reference domain"""
        x = u.free_symbols
        if len(x) == 0:
            return u
        assert len(x) == 1
        a = self.domain.lower
        c = self.reference_domain.lower
        d = self.domain_factor
        x = x.pop()
        return u.xreplace({x: c + (x - a) * d})

    def map_expr_true_domain(self, u: sp.Expr) -> sp.Expr:
        """Return reference point `x` mapped to true domain"""
        x = u.free_symbols
        if len(x) == 0:
            return u
        assert len(x) == 1
        a = self.domain.lower
        c = self.reference_domain.lower
        d = self.domain_factor
        x = x.pop()
        return u.xreplace({x: a + (x - c) / d})

    def map_reference_domain(self, x: sp.Symbol | Array) -> sp.Expr | Array:
        """Return true point `x` mapped to reference domain"""

        if self.domain != self.reference_domain:
            a = self.domain.lower
            c = self.reference_domain.lower
            if isinstance(x, Array | float):
                x = float(c) + (x - float(a)) * float(self.domain_factor)
            else:
                x = c + (x - a) * self.domain_factor
        return x

    def map_true_domain(self, X: sp.Symbol | Array) -> sp.Expr | Array:
        """Return reference point `x` mapped to true domain"""
        if self.domain != self.reference_domain:
            a = self.domain.lower
            c = self.reference_domain.lower
            if isinstance(X, Array | float):
                X = float(a) + (X - float(c)) / float(self.domain_factor)
            else:
                X = a + (X - c) / self.domain_factor
        return X

    def mesh(self, kind: str = "quadrature", N: int = 0) -> Array:
        """Return mesh in the domain of self"""
        if kind == "quadrature":
            return self.map_true_domain(self.quad_points_and_weights(N)[0])
        elif kind == "uniform":
            a, b = self.domain
            M = N if N != 0 else self.N
            return jnp.linspace(float(a), float(b), M)

    def cartesian_mesh(self, kind: str = "quadrature", N: int = 0) -> tuple[Array, ...]:
        rv = self.system._position_vector
        t = self.system.base_scalars()[0]
        xj = self.mesh(kind, N)
        mesh = []
        for r in rv:
            mesh.append(lambdify(t, r, modules="jax")(xj))
        return tuple(mesh)

    def __len__(self) -> int:
        return 1

    def __add__(self, b: BaseSpace):
        from jaxfun.composite import DirectSum

        return DirectSum(self, b)

    def get_padded(self, N: int):
        return self.__class__(
            N,
            domain=self.domain,
            system=self.system,
            name=self.name + "p",
            fun_str=self.fun_str + "p",
        )
