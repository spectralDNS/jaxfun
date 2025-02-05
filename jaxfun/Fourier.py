from functools import partial

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse

from jaxfun.Basespace import BaseSpace, Domain
from jaxfun.coordinates import CoordSys

# ruff: noqa: F706


class Fourier(BaseSpace):
    """Space of all Fourier exponentials of order less than or equal to N"""

    def __init__(
        self,
        N: int,
        domain: Domain = None,
        system: CoordSys = None,
        name: str = "Fourier",
        fun_str: str = "E",
    ) -> None:
        assert N % 2 == 0, "Fourier must use an even number of modes"
        domain = Domain(0, 2 * sp.pi) if domain is None else domain
        BaseSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self._k = {k: self.wavenumbers()[k].item() for k in range(N)}

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        """
        Evaluate a Fourier series at points x.

        .. math:: p(x) = c_0 + c_1 * exp(ix) + ... + c_n * exp(iNx)

        Parameters
        ----------
        X : float
        c : Array

        Returns
        -------
        values : Array

        """
        k: Array = self.wavenumbers()

        def body_fun(i: int, c1: Array) -> Array:
            return c1 + c[i] * jax.lax.exp(1j * k[i] * X)

        c0 = jnp.ones_like(X, dtype=complex) * c[0]
        return jax.lax.fori_loop(1, len(c), body_fun, c0)

    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N if N == 0 else N
        points = jnp.arange(N, dtype=float) * 2 * jnp.pi / N
        return points, jnp.array([2 * jnp.pi / N]*N)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, x: float, i: int) -> complex:
        return jax.lax.exp(1j * self._k[i] * x)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        n: int = self.N if N == 0 else N
        if n < len(c):  # truncation
            k = self.wavenumbers(n)
            return jnp.fft.ifft(c[k], norm="forward")
        return jnp.fft.ifft(c, norm="forward")

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, x: float) -> Array:
        return jax.lax.exp(1j * self.wavenumbers() * x)

    @property
    def reference_domain(self) -> Domain:
        return Domain(0, 2 * sp.pi)

    def wavenumbers(self, N: int = 0) -> Array:
        N = self.N if N == 0 else N
        return jnp.fft.fftfreq(N, 1 / N).astype(int)

    def norm_squared(self) -> Array:
        return jnp.ones(self.N) * 2 * jnp.pi

    def mass_matrix(self) -> BCOO:
        return BCOO.from_scipy_sparse(
            scipy_sparse.diags((self.norm_squared(),), (0,), shape=(self.N, self.N))
        )


def matrices(test: tuple[Fourier, int], trial: tuple[Fourier, int]) -> Array:
    from jax.experimental import sparse
    from scipy import sparse as scipy_sparse

    v, i = test
    u, j = trial
    k = (1j * v.wavenumbers()) ** j * (-1j * u.wavenumbers()) ** i
    if i + j % 2 == 0:
        return sparse.BCOO.from_scipy_sparse(
            scipy_sparse.diags((k.real * v.norm_squared(),), (0,), (v.N, u.N), "csr")
        )
    return sparse.BCOO.from_scipy_sparse(
        scipy_sparse.diags((k * v.norm_squared(),), (0,), (v.N, u.N), "csr")
    )
