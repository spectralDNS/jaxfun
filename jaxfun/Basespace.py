from typing import NamedTuple, Union
from functools import partial
from numbers import Number
import copy
import sympy as sp
import jax
from jax import Array
from jaxfun.utils.common import jacn
from jaxfun.coordinates import CoordSys

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


class BaseSpace:
    def __init__(
        self,
        N: int,
        domain: Domain = Domain(-1, 1),
        coordinates: CoordSys = None,
        name: str = None,
        fun_str: str = 'psi'
    ) -> None:
        from jaxfun.arguments import CartCoordSys1D
        self.N = N
        self._domain = Domain(*domain)
        self.name = name
        self.fun_str = fun_str
        self.system = CartCoordSys1D if coordinates is None else coordinates
        self.bcs = None
        self.stencil = None
        
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
        return self.evaluate_basis_derivative(x, 0)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        raise RuntimeError

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        raise RuntimeError

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
        return float(R / L) if abs(L - R) > 1e-12 else 1

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

    def __len__(self) -> int:
        return 1
