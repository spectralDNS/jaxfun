from collections.abc import Callable

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jit_vmap

from .orthogonal import OrthogonalSpace


class SinCos(OrthogonalSpace):
    """Real Fourier series functionspace"""

    alpha = 0
    beta = 0

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "FourierReal",
        fun_str: str = "sincos",
        **kw,
    ) -> None:
        assert N % 2 == 1, "N must be odd for SinCos"
        domain = Domain(0, 2 * sp.pi) if domain is None else Domain(*domain)
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self._num_quad_points = (N + 1) // 2

    @property
    def reference_domain(self) -> Domain:
        """Reference domain of the Fourier series

        Returns:
            Domain: Reference domain
        """
        return Domain(0, 2 * sp.pi)

    @jit_vmap(in_axes=(0, None))
    def _evaluate(self, X: float | Array, c: Array) -> Array:
        r"""
        Evaluate a Fourier series at points X.

        .. math:: p(x) = c_0 + \sum_{k=1}^{N} c_{2k-1} sin(kX) + c_{2k} cos(kX)

        Args:
            X : Evaluation point in reference space
            c : Expansion coefficients

        Returns:
            values (Array)

        """
        import jax

        def body_fun(i: int, c1: Array) -> Array:
            return (
                c1
                + c[2 * i] * jax.lax.cos(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)
                + c[2 * i - 1] * jax.lax.sin(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)
            )

        c0 = c[0] * jnp.ones_like(X)
        return jax.lax.fori_loop(1, (self.dim + 1) // 2, body_fun, c0)

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        N = self.num_quad_points if N is None else N
        points = jnp.arange(N, dtype=float) * jnp.pi / N
        return points, jnp.full(N, jnp.pi)

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        return jax.lax.cond(
            i % 2,
            lambda *_: jax.lax.sin(
                (i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST
            ),
            lambda *_: jax.lax.cos(
                (i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST
            ),
        )

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        i = jnp.arange(self.dim)
        return jax.lax.select(
            i % 2 == 1,
            jax.lax.sin((i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST),
            jax.lax.cos((i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST),
        )

    def norm_squared(self) -> Array:
        return jnp.ones(self.dim) * 2 * jnp.pi


class SinCosHalf(OrthogonalSpace):
    """Real Fourier series on half-range functionspace"""

    alpha = 0
    beta = 0

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "FourierRealHalf",
        fun_str: str = "sincoshalf",
        **kw,
    ) -> None:
        assert N % 2 == 1, "N must be odd for SinCos"
        domain = Domain(0, sp.pi) if domain is None else Domain(*domain)
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )

    @property
    def reference_domain(self) -> Domain:
        """Reference domain of the Fourier series

        Note that we use a half period, such that the series can represent
        non-periodic functions on the true domain. If chosen as [-pi, pi],
        then the series can only represent periodic functions.

        Returns:
            Domain: Reference domain
        """
        return Domain(0, sp.pi)

    def bnd_values(
        self, k: int = 0
    ) -> tuple[
        Callable[[int | sp.Symbol], sp.Expr],
        Callable[[int | sp.Symbol], sp.Expr],
    ]:
        """Return lambda function for computing boundary values"""
        x = sp.symbols("x", real=True)
        return (
            lambda i: sp.diff(sp.cos((i + 1) // 2 * x), x, k).subs(x, 0)
            if i % 2 == 0
            else sp.diff(sp.sin((i + 1) // 2 * x), x, k).subs(x, 0),
            lambda i: sp.diff(sp.cos((i + 1) // 2 * x), x, k).subs(x, sp.pi)
            if i % 2 == 0
            else sp.diff(sp.sin((i + 1) // 2 * x), x, k).subs(x, sp.pi),
        )

    @jit_vmap(in_axes=(0, None))
    def _evaluate(self, X: float | Array, c: Array) -> Array:
        r"""
        Evaluate a half-range Fourier series at points X.

        .. math:: p(x) = c_0 + \sum_{k=1}^{N} c_{2k-1} cos(kX) + c_{2k} sin(kX)

        Args:
            X : Evaluation point in reference space
            c : Expansion coefficients

        Returns:
            values (Array)

        """
        import jax

        def body_fun(i: int, c1: Array) -> Array:
            return (
                c1
                + c[2 * i] * jax.lax.cos(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)
                + c[2 * i - 1] * jax.lax.sin(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)
            )

        c0 = c[0] * jnp.ones_like(X)
        return jax.lax.fori_loop(1, (self.dim + 1) // 2, body_fun, c0)

    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        N = self.dim if N is None else N
        points = jnp.arange(N, dtype=float) * jnp.pi / (N - 1)
        return points, jnp.full(N, jnp.pi)

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        return jax.lax.cond(
            i % 2,
            lambda *_: jax.lax.sin(
                (i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST
            ),
            lambda *_: jax.lax.cos(
                (i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST
            ),
        )

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        i = jnp.arange(self.dim)
        return jax.lax.select(
            i % 2 == 1,
            jax.lax.sin((i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST),
            jax.lax.cos((i + 1) // 2 * X, accuracy=jax.lax.AccuracyMode.HIGHEST),
        )

    def norm_squared(self) -> Array:
        return jnp.ones(self.dim) * jnp.pi


class Cosine(OrthogonalSpace):
    """Real Fourier cosine series functionspace"""

    alpha = 0
    beta = 0

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "FourierRealCos",
        fun_str: str = "cos",
        **kw,
    ) -> None:
        domain = Domain(0, sp.pi) if domain is None else Domain(*domain)
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )

    @property
    def reference_domain(self) -> Domain:
        """Reference domain of the Fourier cosine series

        Note that we use a half period, such that the series can represent
        even functions on the true domain.

        Returns:
            Domain: Reference domain
        """
        return Domain(0, sp.pi)

    @jit_vmap(in_axes=(0, None))
    def _evaluate(self, X: float | Array, c: Array) -> Array:
        r"""
        Evaluate a half-range Fourier cosine series at points X.

        .. math:: p(x) = \sum_{k=0}^{N-1} c_k cos(kX)

        Args:
            X : Evaluation point in reference space
            c : Expansion coefficients

        Returns:
            values (Array)

        """
        import jax

        def body_fun(i: int, c1: Array) -> Array:
            return c1 + c[i] * jax.lax.cos(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)

        c0 = c[0] * jnp.ones_like(X)
        return jax.lax.fori_loop(1, self.dim, body_fun, c0)

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        N = self.dim if N is None else N
        points = jnp.arange(N, dtype=float) * jnp.pi / (N - 1)
        return points, jnp.full(N, jnp.pi)

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        return jax.lax.cos(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        i = jnp.arange(self.dim)
        return jax.lax.cos(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)

    def norm_squared(self) -> Array:
        return jnp.ones(self.dim) * jnp.pi


class Sine(OrthogonalSpace):
    """Real Fourier sine series functionspace"""

    alpha = 0
    beta = 0

    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "FourierRealSine",
        fun_str: str = "sin",
        **kw,
    ) -> None:
        domain = Domain(0, sp.pi) if domain is None else Domain(*domain)
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )

    @property
    def reference_domain(self) -> Domain:
        """Reference domain of the Fourier sine series

        Note that we use a half period, such that the series can represent
        odd functions on the true domain.

        Returns:
            Domain: Reference domain
        """
        return Domain(0, sp.pi)

    @jit_vmap(in_axes=(0, None))
    def _evaluate(self, X: float | Array, c: Array) -> Array:
        r"""
        Evaluate a half-range Fourier series at points X.

        .. math:: p(x) = \sum_{k=0}^{N-1} c_k sin((k+1)X)

        Args:
            X : Evaluation point in reference space
            c : Expansion coefficients

        Returns:
            values (Array)

        """
        import jax

        def body_fun(i: int, c1: Array) -> Array:
            return c1 + c[i] * jax.lax.sin(
                (i + 1) * X, accuracy=jax.lax.AccuracyMode.HIGHEST
            )

        c0 = jnp.zeros_like(X)
        return jax.lax.fori_loop(0, self.dim, body_fun, c0)

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        N = self.dim if N is None else N
        points = jnp.arange(N, dtype=float) * jnp.pi / (N - 1)
        return points, jnp.full(N, jnp.pi)

    @jit_vmap(in_axes=(0, None), static_argnums=(0, 2))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        return jax.lax.sin((i + 1) * X, accuracy=jax.lax.AccuracyMode.HIGHEST)

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float | Array) -> Array:
        i = jnp.arange(1, self.dim + 1)
        return jax.lax.sin(i * X, accuracy=jax.lax.AccuracyMode.HIGHEST)

    def norm_squared(self) -> Array:
        return jnp.ones(self.dim) * jnp.pi


def matrices(test: tuple[SinCos, int], trial: tuple[SinCos, int]) -> Array | None:
    return None
