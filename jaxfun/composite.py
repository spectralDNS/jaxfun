from typing import Callable
import sympy as sp
from scipy import sparse as scipy_sparse
import jax
from jax import Array
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Some functions are borrowed from shenfun for getting a stencil matrix
# for any combination of boundary conditions
from shenfun import BoundaryConditions, get_stencil_matrix, Domain


n = sp.Symbol("n", integer=True, positive=True)


@jax.jit
def apply_stencil_galerkin(a: BCOO, b: Array) -> Array:
    return a @ b @ a.T


@jax.jit
def apply_stencil_petrovgalerkin(a: BCOO, b: Array, c: BCOO) -> Array:
    return a @ b @ c.T


@jax.jit
def apply_stencil_linear(a: BCOO, b: Array) -> Array:
    return a @ b


@jax.jit
def matmat(a: Array, b: Array) -> Array:
    return a @ b


def from_dense(a: Array, tol: float = 1e-10) -> BCOO:
    z = jnp.where(jnp.abs(a) < tol, jnp.zeros(a.shape), a)
    return BCOO.from_scipy_sparse(scipy_sparse.csr_matrix(z))


class PN:
    """Space of all polynomials of order less than or equal to N"""

    def __init__(self, family, N: int, domain: tuple = Domain(-1, 1)):
        self.family = family
        self.N = N
        self.bcs = None
        self.stencil = {0: 1}
        self.S = BCOO.from_scipy_sparse(scipy_sparse.diags((1,), 0, (N, N)))


class Composite(PN):
    """Space created by combining basis functions

    The stencil matrix is computed from the given boundary conditions, but
    may also be given explicitly.
    """

    def __init__(self, family, N, bcs, domain=Domain(-1, 1), stencil=None):
        PN.__init__(self, family=family, N=N, domain=domain)
        self.bcs = BoundaryConditions(bcs, domain=domain)
        if stencil is None:
            stencil = get_stencil_matrix(
                self.bcs, family.__name__, family.alpha, family.beta, family.gn
            )
            assert len(stencil) == self.bcs.num_bcs()
        self.stencil = {(si[0]): si[1] for si in sorted(stencil.items())}
        self.S = BCOO.from_scipy_sparse(self.stencil_to_scipy_sparse())

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


def inner(test: tuple[PN, int], trial: tuple[PN, int], sparse: bool = False) -> Array:
    r"""Compute coefficient matrix

    .. math::
        (\frac{\partial^{i}\psi_m}{\partialx^{i}}, \frac{\partial^{j}\phi_n}{\partialx^{j}})

    where :math:`\psi_j` is the j'th basis function of the trial space and
    :math:`\phi_i` is the j'th basis function of the test space
    """
    u, j = trial
    v, i = test
    N = v.N
    x, w = v.family.quad_points_and_weights(N)
    Pi = v.family.evaluate_basis_derivative(x, N, k=i)
    Pj = u.family.evaluate_basis_derivative(x, u.N, k=j)
    z = matmat(Pi.T * w[None, :], Pj)
    if u == v:
        z = apply_stencil_galerkin(v.S, z)
    else:
        z = apply_stencil_petrovgalerkin(v.S, z, u.S)
    return from_dense(z) if sparse else z

def innerlinear(test: tuple[PN, int], u: Callable) -> Array:
    r"""Compute inner product (v, u(x))

    """
    v, i = test
    N = v.N
    x, w = v.family.quad_points_and_weights(N)
    Pi = v.family.evaluate_basis_derivative(x, N, k=i)
    uj = sp.lambdify(s, u, modules=["jax"])(x)
    return apply_stencil_linear(uj * w, Pi)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    import Chebyshev
    import Legendre
    import matplotlib.pyplot as plt

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
    plt.show()
