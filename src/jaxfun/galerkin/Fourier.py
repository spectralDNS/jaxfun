import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.experimental import sparse

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap

from .orthogonal import OrthogonalSpace


class Fourier(OrthogonalSpace):
    """Complex exponential Fourier basis on a periodic 1D interval.

    Basis functions:
        E_k(x) = exp(i k x),  k = -N/2+1, ..., N/2   (N even)

    Coefficient ordering follows numpy / JAX fftfreq. The physical domain
    defaults to [0, 2π] unless provided.

    Args:
        N: Even number of modes (must satisfy N % 2 == 0).
        domain: Physical domain (defaults to [0, 2π]).
        system: Optional coordinate system.
        name: Space name (default "Fourier").
        fun_str: Symbol stem for basis functions (default "E").

    Attributes:
        _k: Mapping from coefficient index to integer wavenumber.
    """

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
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
        r"""Evaluate Fourier series at arbitrary (not necessarily uniform) X.

        Uses an explicit loop (fori_loop) accumulating:
            p(X) = sum_{j} c_j exp(i k_j X)

        Args:
            X: Evaluation point(s) in physical domain.
            c: Expansion coefficients (length N).

        Returns:
            Array of p(X) with shape matching X.
        """
        k: Array = self.wavenumbers()

        def body_fun(i: int, c1: Array) -> Array:
            return c1 + c[i] * jax.lax.exp(1j * k[i] * X)

        c0 = jnp.ones_like(X, dtype=complex) * c[0]
        return jax.lax.fori_loop(1, len(c), body_fun, c0)

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> tuple[Array, Array]:
        """Return equispaced quadrature points and uniform weights.

        Args:
            N: Number of points (defaults to self.num_quad_points if 0).

        Returns:
            (points, weights) where points.shape == (N,) and weights == 2π/N.
        """
        N = self.num_quad_points if N == 0 else N
        points = jnp.arange(N, dtype=float) * 2 * jnp.pi / N
        return points, jnp.full(N, 2 * jnp.pi / N)

    @jit_vmap(in_axes=(0, None))
    def eval_basis_function(self, X: float, i: int) -> Array:
        """Evaluate single basis function exp(i k_i X).

        Args:
            X: Points in domain.
            i: Basis index (0 <= i < N).

        Returns:
            exp( i * k_i * X ).
        """
        return jax.lax.exp(1j * self._k[i] * X)

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float) -> Array:
        """Evaluate all basis functions at points X.

        Args:
            X: Points in domain.

        Returns:
            Array shape (N,) for each X containing exp(i k_j X).
        """
        return jax.lax.exp(1j * self.wavenumbers() * X)

    @jax.jit(static_argnums=(0, 2))
    def evaluate_basis_derivative(self, X: Array, k: int = 0) -> Array:
        """Return k-th derivative Vandermonde."""
        v = self.wavenumbers(eliminate_highest_freq=k % 2 and self.N % 2 == 0)
        y = self.eval_basis_functions(X)
        z = (1j * v) ** k * y
        # z = jacn(self.eval_basis_functions, k)(X)
        # if k % 2 == 1 and self.N % 2 == 0:
        #   z = z.at[:, X.shape[0] // 2].set(0)
        return z

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        """Inverse FFT (possible padding) to physical space.

        Args:
            c: Coefficient array.
            kind: Integration strategy (unused placeholder).
            N: Transform length. If N > len(c), pads coefficients with zeros in the
                middle (high wavenumbers).

        Returns:
            Inverse FFT samples (complex), norm="forward".
        """
        n: int = self.N if N == 0 else N
        assert n >= len(c), "Backward transform only supports padding, not truncation"
        if n > len(c):
            c = jnp.hstack(
                (
                    c[: len(c) // 2],
                    jnp.zeros(n - len(c), dtype=c.dtype),
                    c[len(c) // 2 :],
                )
            )
        return jnp.fft.ifft(c, norm="forward")

    @jax.jit(static_argnums=0)
    def scalar_product(self, c: Array) -> Array:
        """Return inner products <c, E_k> via forward FFT.

        Args:
            c: Physical samples (length N).

        Returns:
            Coefficients scaled by 2π / domain_factor.
        """
        return jnp.fft.fft(c, norm="forward") * 2 * jnp.pi / self.domain_factor

    @jax.jit(static_argnums=(0, 2))
    def forward(self, c: Array, N: int = 0) -> Array:
        """Forward FFT (physical -> spectral coefficients).

        Args:
            c: Physical array.
            N: Target number of modes for transform length If N < len(c) then
                the output is truncated.
        """
        n: int = self.N if N == 0 else N
        assert n <= len(c), "Forward transform only supports truncation, not padding"
        out = jnp.fft.fft(c, norm="forward")
        if len(c) > n:
            return out[self.wavenumbers(n)]
        return out

    @property
    def reference_domain(self) -> Domain:
        """Return canonical reference domain [0, 2π]."""
        return Domain(0, 2 * sp.pi)

    @jax.jit(static_argnums=(0, 1, 2))
    def wavenumbers(self, N: int = 0, eliminate_highest_freq: bool = False) -> Array:
        """Return ordered integer wavenumbers matching FFT layout.

        Args:
            N: Number of modes (0 -> self.N).

        Returns:
            Integer array of length N with ordering from fftfreq.
        """
        N = self.N if N == 0 else N
        k = jnp.fft.fftfreq(N, 1 / N).astype(int)
        if eliminate_highest_freq and N % 2 == 0:
            k = k.at[N // 2].set(0)
        return k

    def norm_squared(self) -> Array:
        """Return L2 norm squared of each basis function over [0, 2π]."""
        return jnp.ones(self.N) * 2 * jnp.pi


def matrices(test: tuple[Fourier, int], trial: tuple[Fourier, int]) -> sparse.BCOO:
    """Return sparse operator matrix for Fourier test/trial derivatives.

    Builds diagonal matrix with entries:
        (i k)^{j} * (-i k)^{i} * norm_squared
    where i, j are derivative orders for test/trial functions.

    Args:
        test: Tuple (v, i) with space v and test derivative order i.
        trial: Tuple (u, j) with space u and trial derivative order j.

    Returns:
        sparse.BCOO diagonal matrix shape (v.N, u.N).
    """
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
