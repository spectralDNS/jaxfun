from typing import NamedTuple
from functools import partial
import sympy as sp
from scipy import sparse as scipy_sparse
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO
from jaxfun.utils.common import matmat

# Some functions are borrowed from shenfun for getting a stencil matrix
# for any combination of boundary conditions
from shenfun import BoundaryConditions, get_stencil_matrix, Domain


n = sp.Symbol("n", integer=True, positive=True)


@jax.jit
def apply_stencil_galerkin(S: BCOO, b: Array) -> Array:
    return S @ b @ S.T


@jax.jit
def apply_stencil_petrovgalerkin(S: BCOO, b: Array, P: BCOO) -> Array:
    return S @ b @ P.T


@jax.jit
def apply_stencil_linear(S: BCOO, b: Array) -> Array:
    return S @ b


@jax.jit
def apply_stencil_linearT(a: Array, S: BCOO) -> Array:
    return a @ S


@jax.jit
def to_orthogonal(a: Array, S: BCOO) -> Array:
    return a @ S


class Composite:
    """Space created by combining basis functions

    The stencil matrix is computed from the given boundary conditions, but
    may also be given explicitly.
    """

    def __init__(
        self,
        orthogonal,
        N: int,
        bcs: dict,
        domain: NamedTuple = Domain(-1, 1),
        stencil: dict = None,
        alpha: float = 0,
        beta: float = 0,
    ):
        self.orthogonal = orthogonal(N, domain=domain, alpha=alpha, beta=beta)
        self.bcs = BoundaryConditions(bcs, domain=domain)
        self.N = N
        self.domain = domain
        if stencil is None:
            stencil = get_stencil_matrix(
                self.bcs,
                self.orthogonal.__class__.__name__,
                self.orthogonal.alpha,
                self.orthogonal.beta,
                self.orthogonal.gn,
            )
            assert len(stencil) == self.bcs.num_bcs()
        self.stencil = {(si[0]): si[1] for si in sorted(stencil.items())}
        self.S = BCOO.from_scipy_sparse(self.stencil_to_scipy_sparse())

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        return self.orthogonal.evaluate(x, to_orthogonal(c, self.S))

    def evaluate_basis_derivative(self, x: Array, k: int = 0) -> Array:
        P: Array = self.orthogonal.evaluate_basis_derivative(x, k)
        return apply_stencil_linear(self.S, P)

    def vandermonde(self, x: Array) -> Array:
        P: Array = self.orthogonal.evaluate_basis_derivative(x, 0)
        return apply_stencil_linear(self.S, P)

    def eval_basis_function(self, x: float, i: int) -> float:
        row: Array = self.get_stencil_row(i)
        psi: Array = jnp.array(
            [self.orthogonal.eval_basis_function(x, i + j) for j in self.stencil.keys()]
        )
        return matmat(row, psi)

    def eval_basis_functions(self, x: float) -> Array:
        P: Array = self.orthogonal.eval_basis_functions(x)
        return apply_stencil_linear(self.S, P)

    def get_stencil_row(self, i: int):
        return jnp.array([self.S[i, i + k].data.item() for k in self.stencil.keys()])

    def stencil_to_scipy_sparse(self):
        k = jnp.arange(self.N)
        return scipy_sparse.diags(
            [
                jnp.atleast_1d(sp.lambdify(n, val, modules="jax")(k)).astype(float)
                for val in self.stencil.values()
            ],
            [key for key in self.stencil.keys()],
            shape=(self.N - self.bcs.num_bcs(), self.N),
        )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    from Chebyshev import Chebyshev
    from Legendre import Legendre
    import matplotlib.pyplot as plt
    from jaxfun.inner import inner

    n = sp.Symbol("n", positive=True, integer=True)
    N = 14
    bcB = "u(-1)=0&&u'(-1)=0&&u(1)=0&&u'(1)=0"
    bc = "u(-1)=0&&u(1)=0"
    C = Composite(Chebyshev, N, bc)

    v = jax.random.normal(jax.random.PRNGKey(1), shape=(N, N))
    g = C.S @ v @ C.S.T
    g1 = apply_stencil_galerkin(C.S, v)

    D = C.stencil_to_scipy_sparse()
    vn = v.__array__()
    gn = D @ vn @ D.T

    assert jnp.linalg.norm(gn - g) < 1e-7
    assert jnp.linalg.norm(gn - g1) < 1e-7

    # Galerkin (dense)
    D = inner((C, 0), (C, 2), sparse=True)

    # Petrov-Galerkin method (https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    G = Composite(Chebyshev, N, "u(-1)=0&&u(1)=0")
    PG = Composite(
        Chebyshev,
        N + 2,
        bcB,
        stencil={
            0: 1 / (2 * sp.pi * (n + 1) * (n + 2)),
            2: -1 / (sp.pi * (n**2 + 4 * n + 3)),
            4: 1 / (2 * sp.pi * (n + 2) * (n + 3)),
        },
    )
    L = Composite(Legendre, N, "u(-1)=0&&u(1)=0")
    LG = Composite(
        Legendre,
        N + 2,
        bcB,
        stencil={
            0: 1 / (2 * (2 * n + 3)),
            2: -(2 * n + 5) / (2 * n + 7) / (2 * n + 3),
            4: 1 / (2 * (2 * n + 7)),
        },
    )
    A0 = inner((PG, 0), (G, 2), sparse=True)  # bidiagonal
    A1 = inner((LG, 0), (L, 2), sparse=True)  # bidiagonal

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
    ax0.spy(D.todense())
    ax0.set_title("Galerkin Cheb")
    ax1.spy(A0.todense())
    ax1.set_title("PG Chebyshev")
    ax2.spy(A1.todense())
    ax2.set_title("PG Legendre")
    # plt.show()
