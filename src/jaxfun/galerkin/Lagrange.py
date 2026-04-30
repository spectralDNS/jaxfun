from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.utils.common import Domain, jit_vmap


class Lagrange(OrthogonalSpace):
    def __init__(
        self,
        N: int,
        domain: Domain | None = None,
        system: CoordSys | None = None,
        name: str = "Lagrange",
        fun_str: str = "L",
        **kw,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self.xj: Array = self.quad_points_and_weights(N)[0]
        self.wj: Array = jnp.hstack(
            (
                0.5,
                jnp.array([-1 if i % 2 == 1 else 1 for i in range(1, N - 1)]),
                (-1 if N % 2 == 0 else 1) * 0.5,
            )
        )
        self.cj: Array = jnp.array([jnp.setxor1d(jnp.arange(N), i) for i in range(N)])
        self.weight_factor: float = 2 ** (N - 2) / (N - 1)

    @jit_vmap(in_axes=(0, None))
    def _evaluate(self, X: float, c: Array) -> Array:
        """
        Evaluate a Lagrange series at points X.

        .. math:: p(X) = c_0 * L_0(X) + c_1 * L_1(X) + ... + c_{N-1} * L_{N-1}(X)

        Args:
            X (float): Evaluation point in reference space
            c (Array): Expansion coefficients

        Returns:
            float: Lagrange series evaluated at X.

        """
        ll: Array = self.eval_basis_functions(X)
        return ll @ c

    @property
    def reference_domain(self) -> Domain:
        return Domain(-1, 1)

    def quad_points_and_weights(self, N: int | None = None) -> tuple[Array, Array]:
        N = self.N if N is None else N
        return (
                jnp.cos(jnp.pi * jnp.arange(N) / (N - 1)),
                jnp.ones(N) * jnp.pi / N,
            )

    def eval(self, X: float, c: Array) -> Array:
        N: int = len(c)
        xj: Array = self.xj
        meshpoint: int = jnp.nonzero(xj == X, size=N)[0][0].item()
        if meshpoint != 0:
            return c[meshpoint]
        return self.evaluate(X, c)

    @jit_vmap(in_axes=(0, None))
    def eval_basis_function(self, X: float | Array, i: int) -> Array:
        xj: Array = self.xj

        def inner_loop(_, i: int) -> tuple[Any, Array]:
            return _, X - xj[i]

        _, l0 = jax.lax.scan(inner_loop, 0, jnp.arange(self.N))

        return jnp.take(l0, self.cj[i]).prod() * self.wj[i] * self.weight_factor

    @jit_vmap(in_axes=0)
    def eval_basis_functions(self, X: float) -> Array:
        xj: Array = self.xj
        wj: Array = self.wj

        def inner_loop(_, i: int) -> tuple[Any, Array]:
            return _, (X - xj[i])

        _, l0 = jax.lax.scan(inner_loop, 0, jnp.arange(self.N))

        def inner_loop2(_, i: int):
            return _, jnp.take(l0, self.cj[i]).prod() * wj[i] * self.weight_factor

        _, ll = jax.lax.scan(inner_loop2, 0, jnp.arange(self.N))
        return jnp.array(ll)

    def norm_squared(self) -> Array:
        return jnp.ones(self.N)


def matrices(test: tuple[Lagrange, int], trial: tuple[Lagrange, int]) -> Array | None:
    return None
