from functools import partial
from typing import  NamedTuple
import jax
import jax.numpy as jnp
from jax import Array
from jaxfun.utils.common import jacn
import sympy as sp
from scipy.special import roots_jacobi
from scipy import sparse as scipy_sparse
from jax.experimental.sparse import BCOO
from shenfun import Domain

n = sp.Symbol("n", integer=True, positive=True)


class Jacobi:
    """Space of all polynomials of order less than or equal to N"""

    def __init__(self, N: int, domain: NamedTuple = Domain(-1, 1), alpha: float = 0, beta: float = 0):
        self.N = N
        self.alpha = alpha 
        self.beta = beta
        self.bcs = None
        self.stencil = {0: 1}
        self.S = BCOO.from_scipy_sparse(scipy_sparse.diags((1,), 0, (N, N)))
        self.orthogonal = self

    # Scaling function (see Eq. (2.28) of https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    @staticmethod
    def gn(alpha, beta, n):
        return 1

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        """
        Evaluate a Jacobi series at points x.

        .. math:: p(x) = c_0 * L_0(x) + c_1 * L_1(x) + ... + c_n * L_n(x)

        Parameters
        ----------
        x : float
        c : Array

        Returns
        -------
        values : Array

        Notes
        -----
        The evaluation uses Clenshaw recursion, aka synthetic division.

        """
        raise NotImplementedError


    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N if N == 0 else N
        return jnp.array(roots_jacobi(N, self.alpha, self.beta))


    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_basis_derivative(self, x: Array, k: int = 0) -> Array:
        return jacn(self.eval_basis_functions, k)(x)


    @partial(jax.jit, static_argnums=0)
    def vandermonde(self, x: Array) -> Array:
        return self.evaluate_basis_derivative(x, 0)


    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> float:
        return self.evaluate(x, (0,) * i + (1,))


    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        raise NotImplementedError
