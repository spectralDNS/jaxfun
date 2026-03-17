import jax.numpy as jnp
import sympy as sp
from jax.experimental import sparse
from sympy import Expr, Number, Symbol

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain

from .Jacobi import Jacobi


class Ultraspherical(Jacobi):
    r"""Ultraspherical (Gegenbauer) polynomial basis space.

    Implements an ultraspherical basis via the Jacobi formulation with
    alpha = beta = lambda - 1/2. Provides several evaluation kernels:
      * eval_basis_function: Single C^{(\lambda)}_i evaluation.
      * eval_basis_functions: Vectorized generation of all modes < N.

    The series expansion (degree N-1):
        p(X) = sum_{k=0}^{N-1} c_k C^{(\lambda)}_k(X)

    The ultraspherical polynomials are defined as

    .. math::
        C^{(\lambda)}_n(X) = g_n^{(\lambda-1/2)} P_n^{(\lambda-1/2, \lambda-1/2)}(X)

    where g_n^{(\lambda-1/2)} is a scaling factor that ensures C^{(\lambda)}_n(\pm 1)
    = (\pm 1)^n. Hence, g_n^{(\lambda-1/2)} = 1 / P_n^{(\lambda-1/2, \lambda-1/2)}(1).

    Args:
        N: Number of basis functions (polynomial order = N-1).
        domain: Physical interval (maps to reference [-1, 1]).
        system: Coordinate system (optional).
        name: Basis family name.
        fun_str: Symbol stem for basis functions (default "C").
        **kw: Extra keyword args passed to parent Jacobi constructor.
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Ultraspherical",
        fun_str: str = "C",
        lambda_: Number | float = 0,
        **kw,
    ) -> None:
        Jacobi.__init__(
            self,
            N,
            domain=domain,
            system=system,
            name=name,
            fun_str=fun_str,
            alpha=lambda_ - sp.S.Half,
            beta=lambda_ - sp.S.Half,
        )

    @property
    def lambda_(self):
        return self.alpha + sp.S.Half

    def h(self, n: Symbol | int, k: int) -> Expr:
        if self.lambda_ == 0:  # Chebyshev
            if k > 0:
                return sp.simplify(
                    sp.pi * n * sp.gamma(n + k) / (2 * sp.factorial(n - k))
                )
            return sp.Piecewise((sp.pi, sp.Eq(n, 0)), (sp.pi / 2, True))
        return super().h(n, k)

    def gn(self, n: Symbol | int) -> Expr:
        """Return scaling g_n used in Jacobi-based normalization.

        Args:
            n: Polynomial index symbol.

        Returns:
            SymPy expression 1 / P_n^{(alpha,beta)}(1).
        """
        return sp.S.One / sp.jacobi(n, self.alpha, self.beta, 1)

    def a(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        if self.lambda_ == 0:  # Chebyshev requires tweaking
            if (i - j) == 1:
                return sp.Piecewise((1, sp.Eq(j, 0)), (sp.S.Half, True))
            if (j - i) == 1:
                return sp.S.Half
            return 0
        return super().a(i, j)

    def b(self, i: Symbol | int, j: Symbol | int) -> Expr | float:
        if self.lambda_ == 0:  # Chebyshev requires tweaking
            if (i - j) == 1:
                return sp.Piecewise((1, sp.Eq(j, 0)), (1 / (2 * i), True))
            if (j - i) == 1:
                return -1 / (2 * i)
            return 0
        return super().b(i, j)


def matrices(
    test: tuple[Ultraspherical, int], trial: tuple[Ultraspherical, int]
) -> sparse.BCOO | None:
    """Return sparse mass matrix for (i,j)=(0,0) else None.

    Args:
        test: (space, derivative order) for test function.
        trial: (space, derivative order) for trial function.

    Returns:
        BCOO diagonal mass matrix or None if derivative combo unsupported.
    """
    v, i = test
    u, j = trial
    if i == 0 and j == 0:
        return sparse.BCOO(
            (v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
            shape=(v.N, u.N),
        )
    return None
