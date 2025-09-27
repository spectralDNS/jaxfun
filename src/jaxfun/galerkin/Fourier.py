from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap

from .orthogonal import OrthogonalSpace


class Fourier(OrthogonalSpace):
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
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        w = self.wavenumbers()
        self._k = {k: w[k].item() for k in range(N)}  # map index to wavenumber

    @jit_vmap(in_axes=(0, None))
    def evaluate2(self, X: float | Array, c: Array) -> Array:
        r"""
        Alternative evaluate a Fourier series at points X that are not
        necessarily on a uniform grid.

        .. math:: p(X) = \sum_{k=-N/2+1}^{N/2} c_j exp(ikX)

        Args:
            X (float) : Evaluation point in reference space
            c (Array) : Expansion coefficients

        Returns:
            values (Array)

        """
        k: Array = self.wavenumbers()

        def body_fun(i: int, c1: Array) -> Array:
            return c1 + c[i] * jax.lax.exp(1j * k[i] * X)

        c0 = jnp.ones_like(X, dtype=complex) * c[0]
        return jax.lax.fori_loop(1, len(c), body_fun, c0)

    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.N if N == 0 else N
        points = jnp.arange(N, dtype=float) * 2 * jnp.pi / N
        return points, jnp.full(N, 2 * jnp.pi / N)

    @jit_vmap(in_axes=(0, None))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        return jax.lax.exp(1j * self._k[i] * X)

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        return jax.lax.exp(1j * self.wavenumbers() * X)

    # Cannot jax.jit in case of padding
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        n: int = self.N if N == 0 else N
        if n > self.N:
            c0 = np.zeros(n, dtype=c.dtype)
            k = self.wavenumbers()
            c0[k] = c
            c = jnp.array(c0)
        return self._backward(c, kind, n)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        n: int = self.N if N == 0 else N
        if n < len(c):  # truncation
            k = self.wavenumbers(n)
            return jnp.fft.ifft(c[k], norm="forward")
        return jnp.fft.ifft(c, norm="forward")

    @partial(jax.jit, static_argnums=0)
    def scalar_product(self, c: Array) -> Array:
        return jnp.fft.fft(c, norm="forward") * 2 * jnp.pi / self.domain_factor

    @partial(jax.jit, static_argnums=0)
    def forward(self, c: Array) -> Array:
        return jnp.fft.fft(c, norm="forward")

    @property
    def reference_domain(self) -> Domain:
        return Domain(0, 2 * sp.pi)

    def wavenumbers(self, N: int = 0) -> Array:
        N = self.N if N == 0 else N
        return jnp.fft.fftfreq(N, 1 / N).astype(int)

    def norm_squared(self) -> Array:
        return jnp.ones(self.N) * 2 * jnp.pi


def matrices(test: tuple[Fourier, int], trial: tuple[Fourier, int]) -> Array:
    from jax.experimental import sparse

    v, i = test
    u, j = trial
    k = (1j * v.wavenumbers()) ** j * (-1j * u.wavenumbers()) ** i
    if i + j % 2 == 0:
        return sparse.BCOO(
            (k.real * v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
            shape=(v.N, u.N),
        )
    return sparse.BCOO(
        (k * v.norm_squared(), jnp.vstack((jnp.arange(v.N),) * 2).T),
        shape=(v.N, u.N),
    )
