import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.la import DiaMatrix, diags
from jaxfun.typing import MeshKind
from jaxfun.utils.common import Domain, cache_static, jit_vmap

from .orthogonal import OrthogonalSpace


@jax.jit(static_argnums=(0, 1))
def _fourier_wavenumbers(N: int, eliminate_highest_freq: bool = False) -> Array:
    indices = jnp.arange(N)
    k = jnp.where(indices < (N + 1) // 2, indices, indices - N)
    if eliminate_highest_freq and N % 2 == 0:
        k = k.at[N // 2].set(0)
    return k


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

    """

    # `backward` is an FFT/DCT, so derivatives stay in coefficient space.
    has_fast_transform = True

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

    @jit_vmap(in_axes=(0, None))
    def _evaluate2(self, X: float | Array, c: Array) -> Array:
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

    @cache_static
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        """Return equispaced quadrature points and uniform weights.

        Args:
            N: Number of points (defaults to self.num_quad_points if None).

        Returns:
            (points, weights) where points.shape == (N,) and weights == 2π/N.
        """
        N = self.num_quad_points if N is None else N
        points = jnp.arange(N, dtype=float) * 2 * jnp.pi / N
        return points, jnp.full(N, 2 * jnp.pi / N)

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        """Evaluate single basis function exp(i k_i X).

        Args:
            X: Points in domain.
            i: Basis index (0 <= i < N).

        Returns:
            exp( i * k_i * X ).
        """
        X = jnp.asarray(X)
        return jax.lax.exp(1j * self.wavenumbers()[i] * X)

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        """Evaluate all basis functions at points X.

        Args:
            X: Points in domain.

        Returns:
            Array shape (N,) for each X containing exp(i k_j X).
        """
        X = jnp.asarray(X)
        return jax.lax.exp(1j * self.wavenumbers() * X)

    @jax.jit(static_argnums=(0, 2))
    def evaluate_basis_derivative(self, X: Array, k: int = 0) -> Array:
        """Return k-th derivative Vandermonde."""
        # `wavenumbers` decides for itself whether there is a Nyquist mode to
        # eliminate, which is not the same test in every Fourier layout.
        v = self.wavenumbers(eliminate_highest_freq=bool(k % 2))
        y = self.eval_basis_functions(X)
        z = (1j * v) ** k * y
        return z

    @jax.jit(static_argnums=(0, 2))
    def backward(self, c: Array, N: int | None = None) -> Array:
        """Inverse FFT (possible padding) to physical space.

        Args:
            c: Coefficient array.
            N: Transform length. If N > len(c), pads coefficients with zeros in the
                middle (high wavenumbers).

        Returns:
            Inverse FFT samples (complex), norm="forward".
        """
        n: int = self.N if N is None else N
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
        out = jnp.fft.fft(c, norm="forward") * 2 * jnp.pi / float(self.domain_factor)
        if len(c) > self.N:
            return out[self.wavenumbers()]
        return out

    @jax.jit(static_argnums=0)
    def forward(self, c: Array) -> Array:
        """Forward FFT (physical -> spectral coefficients).

        Args:
            c: Physical array.
            N: Target number of modes for transform length. If N < len(c) then
                the output is truncated.
        """
        assert len(c) >= self.N, (
            "Forward transform only supports truncation, not padding"
        )
        out = jnp.fft.fft(c, norm="forward")
        if len(c) > self.N:
            return out[self.wavenumbers()]
        return out

    @property
    def reference_domain(self) -> Domain:
        """Return canonical reference domain [0, 2π]."""
        return Domain(0, 2 * sp.pi)

    @jax.jit(static_argnums=(0, 1, 2))
    def wavenumbers(
        self, N: int | None = None, eliminate_highest_freq: bool = False
    ) -> Array:
        """Return ordered integer wavenumbers matching FFT layout.

        Args:
            N: Number of modes (None -> self.N).

        Returns:
            Integer array of length N with ordering from fftfreq.
        """
        N = self.N if N is None else N
        return _fourier_wavenumbers(N, eliminate_highest_freq)

    def norm_squared(self) -> Array:
        """Return L2 norm squared of each basis function over [0, 2π]."""
        return jnp.ones(self.N) * 2 * jnp.pi

    def derivative_coeffs(self, c: Array, k: int = 0) -> Array:
        """
        Args:
            c: Coefficients of orthogonal series.
            k: Order of derivative to compute.

        Returns:
            Array (N,) of coefficients for the k'th derivative of the series.
        """
        if k == 0:
            return c

        m = self.wavenumbers(eliminate_highest_freq=k % 2 == 1)
        return (1j * m) ** k * c

    @jax.jit(static_argnums=(0, 1, 2))
    def mesh(
        self, kind: MeshKind | str = MeshKind.QUADRATURE, N: int | None = None
    ) -> Array:
        """Return (periodic) sampling mesh in true domain.

        Args:
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
                MeshKind.UNIFORM).
            N: Number of points (defaults to self.num_quad_points).
        """
        # Both quadrature and uniform meshes are equispaced for Fourier, so ignore kind.
        a, b = self.domain
        N = self.num_quad_points if N is None else N
        return jnp.linspace(float(a), float(b), N, endpoint=False)

    def _matrices(
        self, i: int, trial: tuple[OrthogonalSpace, int], q: int = 0
    ) -> DiaMatrix | None:
        """Return sparse operator matrix for Fourier test/trial derivatives.

        Builds diagonal matrix with entries:
            (i k)^{j} * (-i k)^{i} * norm_squared
        where i, j are derivative orders for test/trial functions.

        Args:
            i: Derivative order for test function.
            trial: Tuple (u, j) with space u and trial derivative order j.
            q: polynomial degree of coefficient.

        Returns:
            DiaMatrix diagonal matrix or None if combination unsupported.
        """
        if q != 0:
            return None
        u, j = trial
        assert (
            isinstance(u, Fourier)
            and u.is_hermitian_half is self.is_hermitian_half
            and u.N == self.N
        ), (
            "Trial spaces must be Fourier spaces of the same kind and size as the "
            "test space (both full spectrum or both half) for Fourier matrices"
        )
        k = (1j * self.wavenumbers()) ** j * (-1j * u.wavenumbers()) ** i
        if (i + j) % 2 == 0:
            k = k.real
        diagonal = k * 2 * jnp.pi
        return diags([diagonal], offsets=(0,), shape=(self.N, u.N))


class RFourier(Fourier):
    r"""Real-to-complex Fourier basis: the non-negative half of the spectrum.

    A real periodic field has a Hermitian spectrum, ``c_{-k} = conj(c_k)``, so
    the negative wavenumbers carry no information. This space stores only
    ``k = 0, 1, ..., N/2`` -- ``N/2 + 1`` coefficients for ``N`` quadrature
    points -- and transforms with `rfft`/`irfft`. The coefficients are exactly
    the ones `Fourier` holds at those wavenumbers, so every operator matrix is
    `Fourier`'s restricted to them: the equations for ``-k`` are the conjugates
    of the ones for ``+k`` and are dropped as redundant, not approximated away.

    Halving the spectrum halves the linear algebra and the wall-normal
    transforms of a Fourier x polynomial solver, which is the point.

    Args:
        N: Even number of quadrature points. The number of coefficients is
            ``N // 2 + 1``, so `RFourier(N)` and `Fourier(N)` resolve the same
            physical field -- pass the same `N` when swapping one for the other.
        domain: Physical domain (defaults to [0, 2π]).
        system: Optional coordinate system.
        name: Space name (default "RFourier").
        fun_str: Symbol stem for basis functions (default "E").

    Two consequences of storing half a spectrum are worth knowing:

    The reconstruction is real-linear, not complex-linear: it adds the conjugate
    half back, so it is ``Re(sum_k w_k c_k E_k(x))`` with ``w_0 = w_{N/2} = 1``
    and ``w_k = 2`` in between, and *no* matrix ``V`` satisfies
    ``backward(c) == V @ c``. `vandermonde` therefore holds the basis functions
    used to assemble matrices and to take scalar products, and is not what
    `backward` inverts; `eval_reconstruction` carries the weights instead.

    The Nyquist mode is only well defined at zero. A real field cannot carry a
    phase there, ``d/dx`` of it is not representable (`derivative_coeffs` zeroes
    it, as `Fourier` does), and padding it to a finer mesh is ambiguous -- this
    space splits it evenly, `Fourier` does not split it at all, and the two
    disagree unless it vanishes. Hold it at zero and everything is exact.

    One practical consequence of the odd-looking ``N/2 + 1``: the multi-device
    transform shards the spectral axis, so that count -- not `N` -- is what has
    to divide by the number of devices. ``RFourier(14)`` shards over two devices,
    ``RFourier(16)`` does not.
    """

    is_hermitian_half = True

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "RFourier",
        fun_str: str = "E",
    ) -> None:
        assert N % 2 == 0, "RFourier must use an even number of quadrature points"
        domain = Domain(0, 2 * sp.pi) if domain is None else domain
        OrthogonalSpace.__init__(
            self, N // 2 + 1, domain=domain, system=system, name=name, fun_str=fun_str
        )
        # Set after the base constructor, which sizes the quadrature by the
        # number of coefficients -- the one place where the two differ here.
        self._num_quad_points = N

    @jax.jit(static_argnums=(0, 1, 2))
    def wavenumbers(
        self, N: int | None = None, eliminate_highest_freq: bool = False
    ) -> Array:
        """Return the non-negative wavenumbers 0, 1, ..., N/2.

        Args:
            N: Number of modes (None -> self.N).
            eliminate_highest_freq: Zero the Nyquist wavenumber, which is the
                last one -- but only when the full spectrum is asked for, since
                a truncated one does not reach it.

        Returns:
            Integer array of length N.
        """
        N = self.N if N is None else N
        k = jnp.arange(N)
        return k.at[-1].set(0) if eliminate_highest_freq and N == self.N else k

    @cache_static
    def hermitian_weights(self) -> Array:
        """Return how many times each stored mode appears in the real field.

        Twice for every wavenumber whose conjugate partner was dropped, once for
        the two that are their own partners: k = 0 and the Nyquist.
        """
        return jnp.full(self.N, 2.0).at[0].set(1.0).at[-1].set(1.0)

    def eval_reconstruction(self, X: float | Array) -> Array:
        """Return the weighted basis values whose real part rebuilds the field."""
        return self.eval_basis_functions(X) * self.hermitian_weights()

    @jax.jit(static_argnums=(0, 2))
    def backward(self, c: Array, N: int | None = None) -> Array:
        """Inverse real FFT (possibly padded) to physical space.

        Args:
            c: Coefficient array of non-negative wavenumbers.
            N: Number of physical points. If it exceeds the transform length of
                `c`, the high wavenumbers are zero-padded.

        Returns:
            Real samples on the quadrature mesh, norm="forward".
        """
        n: int = self.num_quad_points if N is None else N
        assert n // 2 + 1 >= len(c), (
            "Backward transform only supports padding, not truncation"
        )
        return jnp.fft.irfft(c, n=n, norm="forward")

    @jax.jit(static_argnums=0)
    def scalar_product(self, c: Array) -> Array:
        """Return inner products <c, E_k> for k >= 0 via forward real FFT.

        Args:
            c: Real physical samples, at least self.num_quad_points of them.

        Returns:
            Coefficients scaled by 2π / domain_factor, truncated to self.N.
        """
        assert not jnp.iscomplexobj(c), (
            "RFourier represents real fields; pass real samples (a complex array "
            "whose imaginary part is zero still has to be taken .real explicitly)"
        )
        out = jnp.fft.rfft(c, norm="forward") * 2 * jnp.pi / float(self.domain_factor)
        return out[: self.N]

    @jax.jit(static_argnums=0)
    def forward(self, c: Array) -> Array:
        """Forward real FFT (physical -> spectral), truncated to self.N modes.

        Args:
            c: Real physical samples, at least self.num_quad_points of them.
        """
        assert len(c) >= self.num_quad_points, (
            "Forward transform only supports truncation, not padding"
        )
        assert not jnp.iscomplexobj(c), (
            "RFourier represents real fields; pass real samples (a complex array "
            "whose imaginary part is zero still has to be taken .real explicitly)"
        )
        return jnp.fft.rfft(c, norm="forward")[: self.N]
