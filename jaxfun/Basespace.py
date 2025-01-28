from __future__ import annotations
from typing import NamedTuple, Union
from functools import partial
from scipy import sparse as scipy_sparse
from sympy import Number
import copy
import sympy as sp
import jax
from jax import Array
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxfun.utils.common import jacn
from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import lambdify

n = sp.Symbol("n", integer=True, positive=True)  # index


class Domain(NamedTuple):
    lower: Number
    upper: Number


class BoundaryConditions(dict):
    """Boundary conditions as a dictionary"""

    def __init__(self, bc: dict, domain: Domain = None) -> None:
        bcs = {"left": {}, "right": {}}
        bcs.update(copy.deepcopy(bc))
        dict.__init__(self, bcs)

    def orderednames(self) -> list[str]:
        return ["L" + bci for bci in sorted(self["left"].keys())] + [
            "R" + bci for bci in sorted(self["right"].keys())
        ]

    def orderedvals(self) -> list[Number]:
        ls = []
        for lr in ("left", "right"):
            for key in sorted(self[lr].keys()):
                val = self[lr][key]
                ls.append(val[1] if isinstance(val, (tuple, list)) else val)
        return ls

    def num_bcs(self) -> int:
        return len(self.orderedvals())

    def num_derivatives(self):
        n = {"D": 0, "R": 0, "N": 1, "N2": 2, "N3": 3, "N4": 4}
        num_diff = 0
        for val in self.values():
            for k in val:
                num_diff += n[k]
        return num_diff

    def is_homogeneous(self):
        for val in self.values():
            for v in val.values():
                if v != 0:
                    return False
        return True

    def get_homogeneous(self):
        bc = {}
        for k, v in self.items():
            bc[k] = {}
            for s in v:
                bc[k][s] = 0
        return BoundaryConditions(bc)


class BaseSpace:
    def __init__(
        self,
        N: int,
        domain: Domain = Domain(-1, 1),
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
        self.S = BCOO.from_scipy_sparse(scipy_sparse.diags((1,), (0,), (N + 1, N + 1)))

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        raise RuntimeError

    def quad_points_and_weights(self, N: int = 0) -> Array:
        raise RuntimeError

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_basis_derivative(self, x: Array, k: int = 0) -> Array:
        return jacn(self.eval_basis_functions, k)(x)

    @partial(jax.jit, static_argnums=0)
    def vandermonde(self, x: Array) -> Array:
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
        return self.evaluate_basis_derivative(x, 0)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        raise RuntimeError

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        raise RuntimeError

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
        return self.N + 1

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

    def map_reference_domain(self, x: Union[sp.Symbol, Array]) -> Union[sp.Expr, Array]:
        """Return true point `x` mapped to reference domain"""

        if not self.domain == self.reference_domain:
            a = self.domain.lower
            c = self.reference_domain.lower
            if isinstance(x, (Array, float)):
                x = float(c) + (x - float(a)) * float(self.domain_factor)
            else:
                x = c + (x - a) * self.domain_factor
        return x

    def map_true_domain(self, x: Union[sp.Symbol, Array]) -> Union[sp.Expr, Array]:
        """Return reference point `x` mapped to true domain"""
        if not self.domain == self.reference_domain:
            a = self.domain.lower
            c = self.reference_domain.lower
            if isinstance(x, (Array, float)):
                x = float(a) + (x - float(c)) / float(self.domain_factor)
            else:
                x = a + (x - c) / self.domain_factor
        return x

    def mesh(self, kind: str = "quadrature", N: int = 0) -> Array:
        """Return mesh in the domain of self"""
        if kind == "quadrature":
            return self.map_true_domain(self.quad_points_and_weights()[0])
        elif kind == "uniform":
            a, b = self.domain
            M = N if N != 0 else self.N
            return jnp.linspace(float(a), float(b), M)

    def cartesian_mesh(self, kind: str = "quadrature", N: int = 0):
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
