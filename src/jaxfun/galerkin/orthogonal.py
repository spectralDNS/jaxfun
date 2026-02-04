"""Base class for 1D orthogonal (or orthogonal-like) polynomial spaces.

Provides a unified API for:
- Basis / series evaluation (and derivatives via automatic jacobians)
- Quadrature based scalar products and forward/backward transforms
- Optional stencil interface (overridden in Composite subclasses)
- Mapping between true domain [a, b] and reference domain (usually [-1, 1])

Subclasses must implement:
    quad_points_and_weights
    eval_basis_function
    eval_basis_functions
    norm_squared
    reference_domain
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Self, overload

import jax
import jax.numpy as jnp
import sympy as sp
from jax.experimental import sparse
from jax.experimental.sparse import BCOO

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import CoordSys
from jaxfun.typing import Array
from jaxfun.utils.common import Domain, jacn, jit_vmap, lambdify

if TYPE_CHECKING:
    from jaxfun.galerkin.composite import BCGeneric, BoundaryConditions, DirectSum


class OrthogonalSpace(BaseSpace):
    """Abstract 1D orthogonal function space (polynomials / Fourier etc.).

    Args:
        N: Number of (raw) basis functions.
        domain: Physical domain (tuple (a, b)); if None reference used.
        system: Coordinate system (metric / symbols). Required for forms.
        name: Space name identifier.
        fun_str: Symbol stem for basis functions (for SymPy printing).

    Attributes:
        N: Number of modes.
        _domain: Physical Domain (None -> reference).
        _num_quad_points: Default quadrature resolution (== N).
        S: Stencil matrix (identity here; overridden in Composite).
        stencil: Dict describing diagonal shifts (0:1 for identity).
        orthogonal: Self alias (Composite replaces with underlying).
    """

    def __init__(
        self,
        N: int,
        *,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "OrthogonalSpace",
        fun_str: str = "psi",
        **kw,
    ) -> None:
        self.N: int = N
        self._num_quad_points: int = N
        if domain is None:
            domain = self.reference_domain
        self._domain = Domain(*domain)
        self.bcs: BoundaryConditions | None = None
        self.orthogonal: Self = self
        self.stencil = {0: 1}
        self.S = sparse.BCOO(
            (jnp.ones(N), jnp.vstack((jnp.arange(N),) * 2).T), shape=(N, N)
        )
        super().__init__(system, name, fun_str)

    @abstractmethod
    def norm_squared(self) -> Array:
        """Return norms squared"""
        pass

    @abstractmethod
    def quad_points_and_weights(self, N: int = 0) -> tuple[Array, Array]:
        """Return (points, weights) for orthogonality measure (abstract)."""
        pass

    @property
    def num_quad_points(self) -> int:
        """Return default number of quadrature points."""
        return self._num_quad_points

    @jit_vmap(in_axes=(0, None))
    def evaluate(self, X: float | Array, c: Array) -> Array:
        """Evaluate truncated series sum_k c_k psi_k(X).

        Args:
            X: Evaluation point(s) in reference coordinates.
            c: Coefficient vector (length <= N).

        Returns:
            Array of shape like X containing series evaluation.
        """
        return self.eval_basis_functions(X)[: c.shape[0]] @ c

    @jax.jit(static_argnums=0)
    def vandermonde(self, X: Array) -> Array:
        r"""Return pseudo-Vandermonde matrix V_{m,k}=psi_k(X_m).

        Args:
            X: 1D array of sample points (reference domain).

        Returns:
            Array shape (len(X), N) with basis values.
        """
        return self.evaluate_basis_derivative(X, 0)

    @abstractmethod
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        """Evaluate single basis function psi_i at point X (abstract)."""
        pass

    @abstractmethod
    def eval_basis_functions(self, X: float | Array) -> Array:
        """Evaluate all basis functions psi_0..psi_{N-1} at X (abstract)."""
        pass

    @jax.jit(static_argnums=(0, 2))
    def evaluate_basis_derivative(self, X: Array, k: int = 0) -> Array:
        """Return k-th derivative Vandermonde (automatic Jacobian stack)."""
        return jacn(self.eval_basis_functions, k)(X)

    # backward is wrapped because padding may require non-jitable code
    @jax.jit(static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        """Backward transform (coefficients -> samples) via evaluate."""
        return self._backward(c, kind, N)

    @jax.jit(static_argnums=(0, 2, 3))
    def _backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        """Implementation of backward (allows subclass override)."""
        xj = self.mesh(kind=kind, N=N)
        return self.evaluate(self.map_reference_domain(xj), c)

    @jax.jit(static_argnums=0)
    def mass_matrix(self) -> BCOO:
        """Return diagonal mass matrix (orthogonality) in sparse format."""
        return BCOO(
            (
                self.norm_squared() / self.domain_factor,
                jnp.vstack((jnp.arange(self.N),) * 2).T,
            ),
            shape=(self.N, self.N),
        )

    @jax.jit(static_argnums=0)
    def forward(self, u: Array) -> Array:
        """Forward projection (samples -> coefficients) using orthogonality."""
        A = self.norm_squared() / self.domain_factor
        L = self.scalar_product(u)
        return L / A

    @jax.jit(static_argnums=0)
    def scalar_product(self, u: Array) -> Array:
        """Return vector of inner products <u, psi_i> (weighted)."""
        xj, wj = self.quad_points_and_weights()
        Pi = self.vandermonde(xj)
        sg = self.system.sg / self.domain_factor
        if sp.sympify(sg).is_number:
            wj = wj * float(sg)
        else:
            sg = lambdify(self.system.base_scalars()[0], self.map_expr_true_domain(sg))(
                xj
            )
            wj = wj * sg
        return (u * wj) @ jnp.conj(Pi)

    @jax.jit(static_argnums=0)
    def apply_stencil_galerkin(self, b: Array) -> Array:
        """Apply (left,right) stencil in Galerkin case (identity here)."""
        return b

    @jax.jit(static_argnums=0)
    def apply_stencils_petrovgalerkin(self, b: Array, P: BCOO) -> Array:
        """Apply trial stencil only (identity left) for Petrov–Galerkin."""
        return b @ P.T

    @jax.jit(static_argnums=0)
    def apply_stencil_left(self, b: Array) -> Array:
        """Apply test-side stencil (identity in pure orthogonal space)."""
        return b

    @jax.jit(static_argnums=0)
    def apply_stencil_right(self, a: Array) -> Array:
        """Apply trial-side stencil (identity in pure orthogonal space)."""
        return a

    @property
    def dim(self) -> int:
        """Return total number of raw modes N."""
        return self.N

    @property
    def dims(self) -> int:
        """Return spatial dimensionality (always 1)."""
        return 1

    @property
    def num_dofs(self) -> int:
        """Return number of active degrees of freedom."""
        return self.dim

    @property
    def rank(self):
        """Return tensor rank (0 for scalar spaces)."""
        return 0

    @property
    def domain(self) -> Domain:
        """Return physical domain."""
        return self._domain

    @property
    @abstractmethod
    def reference_domain(self) -> Domain:
        """Return canonical reference domain (implemented in subclass)."""
        pass

    @property
    def domain_factor(self) -> int | float:
        """Return scaling factor mapping true -> reference length.

        Value = (reference_length / true_length). If lengths are equal
        (within tolerance) returns 1 for numerical stability.
        """
        a, b = self.domain
        c, d = self.reference_domain
        L = b - a
        R = d - c
        return R / L if abs(L - R) > 1e-12 else 1

    @overload
    def map_expr_reference_domain(self, u: sp.Expr) -> sp.Expr: ...
    @overload
    def map_expr_reference_domain(self, u: sp.Basic) -> sp.Basic: ...
    def map_expr_reference_domain(self, u: sp.Expr | sp.Basic) -> sp.Expr | sp.Basic:
        """Return expression u(x) rewritten with reference coord X.

        Maps physical x into reference X so u can be evaluated in
        reference space routines.
        """
        x = u.free_symbols
        if len(x) == 0:
            return u
        a = self.domain.lower
        c = self.reference_domain.lower
        d = self.domain_factor
        x = self.system.base_scalars()[0]
        return u.xreplace({x: c + (x - a) * d})

    @overload
    def map_expr_true_domain(self, u: sp.Expr) -> sp.Expr: ...
    @overload
    def map_expr_true_domain(self, u: sp.Basic) -> sp.Basic: ...
    def map_expr_true_domain(self, u: sp.Expr | sp.Basic) -> sp.Expr | sp.Basic:
        """Return expression u(X) rewritten with true coordinate x."""
        x = u.free_symbols
        if len(x) == 0:
            return u
        a = self.domain.lower
        c = self.reference_domain.lower
        d = self.domain_factor
        x = self.system.base_scalars()[0]
        return u.xreplace({x: a + (x - c) / d})

    @overload
    def map_reference_domain(self, x: Array) -> Array: ...
    @overload
    def map_reference_domain(self, x: sp.Symbol) -> sp.Expr: ...
    def map_reference_domain(self, x: sp.Symbol | Array) -> sp.Expr | Array:
        """Map true domain point x to reference coordinate X."""
        X = x
        if self.domain != self.reference_domain:
            a = self.domain.lower
            c = self.reference_domain.lower
            if isinstance(x, Array | float):
                X = float(c) + (x - float(a)) * float(self.domain_factor)
            else:
                X = c + (x - a) * self.domain_factor
        return X

    @overload
    def map_true_domain(self, X: Array) -> Array: ...
    @overload
    def map_true_domain(self, X: sp.Symbol) -> sp.Expr: ...
    def map_true_domain(self, X: sp.Symbol | Array) -> sp.Expr | Array:
        """Map reference coordinate X to true domain point x."""
        x = X
        if self.domain != self.reference_domain:
            a = self.domain.lower
            c = self.reference_domain.lower
            if isinstance(X, Array | float):
                x = float(a) + (X - float(c)) / float(self.domain_factor)
            else:
                x = a + (X - c) / self.domain_factor
        return x

    @jax.jit(static_argnums=(0, 1, 2))
    def mesh(self, kind: str = "quadrature", N: int = 0) -> Array:
        """Return sampling mesh in true domain.

        Args:
            kind: 'quadrature' (default) or 'uniform'.
            N: Number of uniform points (0 -> num_quad_points).
        """
        if kind == "quadrature":
            return self.map_true_domain(self.quad_points_and_weights(N)[0])
        assert kind == "uniform"
        a, b = self.domain
        M = N if N != 0 else self.num_quad_points
        return jnp.linspace(float(a), float(b), M)

    def cartesian_mesh(self, kind: str = "quadrature", N: int = 0) -> tuple[Array, ...]:
        """Return physical Cartesian mesh (tuple) for current coordinate system."""
        rv = self.system._position_vector
        t = self.system.base_scalars()[0]
        xj = self.mesh(kind, N)
        mesh = []
        for r in rv:
            mesh.append(lambdify(t, r, modules="jax")(xj))
        return tuple(mesh)

    def __len__(self) -> int:
        """Return number of spatial dimensions (always 1)."""
        return 1

    def __add__(self, b: BCGeneric) -> DirectSum:
        """Direct sum self ⊕ b (delegated to composite.DirectSum)."""
        from jaxfun.galerkin.composite import DirectSum

        return DirectSum(self, b)

    def get_padded(self, N: int):
        """Return new instance with padded/truncated number of modes N."""
        return self.__class__(
            N,
            domain=self.domain,
            system=self.system,
            name=self.name + "p",
            fun_str=self.fun_str + "p",
        )
