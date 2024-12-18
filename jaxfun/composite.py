from functools import partial
import sympy as sp
from scipy import sparse as scipy_sparse
import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from jaxfun.Basespace import BaseSpace, n, Domain, BoundaryConditions
from jaxfun.Jacobi import Jacobi
from jaxfun.utils.common import matmat


class Composite(BaseSpace):
    """Space created by combining orthogonal basis functions

    The stencil matrix is computed from the given boundary conditions, but
    may also be given explicitly.
    """

    def __init__(
        self,
        orthogonal,
        N: int,
        bcs: dict,
        domain: Domain = Domain(-1, 1),
        stencil: dict = None,
        alpha: float = 0,
        beta: float = 0,
        scaling: sp.Expr = sp.S(1),
    ):
        BaseSpace.__init__(self, N, domain)
        self.orthogonal = orthogonal(N, domain=domain, alpha=alpha, beta=beta)
        self.bcs = BoundaryConditions(bcs, domain=domain)
        if stencil is None:
            stencil = get_stencil_matrix(self.bcs, self.orthogonal)
        self.stencil = {(si[0]): si[1] / scaling for si in sorted(stencil.items())}
        self.S = BCOO.from_scipy_sparse(self.stencil_to_scipy_sparse())

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: float, c: Array) -> float:
        return self.orthogonal.evaluate(x, self.to_orthogonal(c))

    def evaluate_basis_derivative(self, x: Array, k: int = 0) -> Array:
        P: Array = self.orthogonal.evaluate_basis_derivative(x, k)
        return self.apply_stencil_left(P)

    def vandermonde(self, x: Array) -> Array:
        P: Array = self.orthogonal.evaluate_basis_derivative(x, 0)
        return self.apply_stencil_left(P)

    def eval_basis_function(self, x: float, i: int) -> float:
        row: Array = self.get_stencil_row(i)
        psi: Array = jnp.array(
            [self.orthogonal.eval_basis_function(x, i + j) for j in self.stencil.keys()]
        )
        return matmat(row, psi)

    def eval_basis_functions(self, x: float) -> Array:
        P: Array = self.orthogonal.eval_basis_functions(x)
        return self.apply_stencil_left(P)

    def get_stencil_row(self, i: int):
        return self.S[i].data[: len(self.stencil)]

    def stencil_to_scipy_sparse(self):
        k = jnp.arange(self.N)
        return scipy_sparse.diags(
            [
                jnp.atleast_1d(
                    sp.lambdify(
                        n, val, modules=["jax", {"gamma": jax.scipy.special.gamma}]
                    )(k)
                ).astype(float)
                for val in self.stencil.values()
            ],
            [key for key in self.stencil.keys()],
            shape=(self.N + 1 - self.bcs.num_bcs(), self.N + 1),
        )

    @partial(jax.jit, static_argnums=0)
    def to_orthogonal(self, a: Array) -> Array:
        return a @ self.S

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_galerkin(self, b: Array) -> Array:
        return self.S @ b @ self.S.T

    @partial(jax.jit, static_argnums=0)
    def apply_stencils_petrovgalerkin(self, b: Array, P: BCOO) -> Array:
        return self.S @ b @ P.T

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_left(self, b: Array) -> Array:
        return self.S @ b

    @partial(jax.jit, static_argnums=0)
    def apply_stencil_right(self, a: Array) -> Array:
        return a @ self.S

    def mass_matrix(self):
        P: BCOO = self.orthogonal.mass_matrix()
        T = matmat((self.S * P.data[None, :]), self.S.T).data.reshape(
            (self.N + 1 - self.bcs.num_bcs(), -1)
        )  # sparse @ sparse -> dense (yet sparse format), so need to remove zeros
        return sparse.BCOO.from_scipy_sparse(scipy_sparse.csr_matrix(T))


def get_stencil_matrix(bcs: BoundaryConditions, orthogonal: Jacobi) -> dict:
    r"""Return stencil matrix as dictionary of keys, values being diagonals
    and sympy expressions.

    For example, the Neumann basis functions for Chebyshev polynomials are

    .. math::
        \psi_i = T_i - \frac{i^2}{i^2+4i+4}T_{i+2}

    Hence, we get

    Example
    -------
    >>> from jaxfun import Chebyshev, Composite
    >>> C = Composite(Chebyshev.Chebyshev, 10, {'left': {'N': 0}, 'right': {'N': 0}})
    >>> C.stencil
    {0: 1, 2: -n**2/(n**2 + 4*n + 4)}
    """
    bc = {"D": 0, "N": 1, "N2": 2, "N3": 3, "N4": 4}
    lr = {"L": 0, "R": 1}
    lra = {"L": "left", "R": "right"}
    s = []
    r = []
    for key in bcs.orderednames():
        k, v = key[0], key[1:]
        if v in "WR":  # Robin conditions
            k0 = 0 if v == "R" else 1
            alfa = bcs[lra[k]][v][0]
            f = [
                orthogonal.bnd_values(k=k0)[lr[k]],
                orthogonal.bnd_values(k=k0 + 1)[lr[k]],
            ]
            s.append(
                [
                    sp.simplify(f[0](n + j) + alfa * f[1](n + j))
                    for j in range(1, 1 + bcs.num_bcs())
                ]
            )
            r.append(-sp.simplify(f[0](n) + alfa * f[1](n)))
        else:
            f = orthogonal.bnd_values(k=bc[v])[lr[k]]
            s.append([sp.simplify(f(n + j)) for j in range(1, 1 + bcs.num_bcs())])
            r.append(-sp.simplify(f(n)))
    A = sp.Matrix(s)
    b = sp.Matrix(r)
    M = sp.simplify(A.solve(b))
    d = {0: 1}
    for i, s in enumerate(M):
        if not s == 0:
            d[i + 1] = s
    return d


if __name__ == "__main__":
    from Chebyshev import Chebyshev
    from Legendre import Legendre
    import matplotlib.pyplot as plt
    from jaxfun.inner import inner
    from jaxfun.arguments import TestFunction, TrialFunction, x

    n = sp.Symbol("n", positive=True, integer=True)
    N = 50
    biharmonic = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
    dirichlet = {"left": {"D": 0}, "right": {"D": 0}}
    C = Composite(Chebyshev, N, dirichlet)

    v = jax.random.normal(jax.random.PRNGKey(1), shape=(N + 1, N + 1))
    g = C.S @ v @ C.S.T
    g1 = C.apply_stencil_galerkin(v)

    D = C.stencil_to_scipy_sparse()
    vn = v.__array__()
    gn = D @ vn @ D.T

    assert jnp.linalg.norm(gn - g) < 1e-7
    assert jnp.linalg.norm(gn - g1) < 1e-7

    # Galerkin (dense)
    u = TrialFunction(x, C)
    v = TestFunction(x, C)
    D = inner(v * sp.diff(u, x, 2), sparse=True, sparse_tol=100)

    # Petrov-Galerkin method (https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    G = Composite(Chebyshev, N, dirichlet, scaling=n + 1)
    PG = Composite(
        Chebyshev,
        N + 2,
        biharmonic,
        stencil={
            0: 1 / (2 * sp.pi * (n + 1) * (n + 2)),
            2: -1 / (sp.pi * (n**2 + 4 * n + 3)),
            4: 1 / (2 * sp.pi * (n + 2) * (n + 3)),
        },
    )
    L = Composite(Legendre, N, dirichlet, scaling=n + 1)
    LG = Composite(
        Legendre,
        N + 2,
        biharmonic,
        stencil={
            0: 1 / (2 * (2 * n + 3)),
            2: -(2 * n + 5) / (2 * n + 7) / (2 * n + 3),
            4: 1 / (2 * (2 * n + 7)),
        },
    )
    A0 = inner(
        TestFunction(x, PG) * sp.diff(TrialFunction(x, G), x, 2),
        sparse=True,
        sparse_tol=1000,
    )  # bidiagonal
    A1 = inner(
        TestFunction(x, LG) * sp.diff(TrialFunction(x, L), x, 2),
        sparse=True,
        sparse_tol=1000,
    )  # bidiagonal
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)
    ax0.spy(D.todense())
    ax0.set_title("Galerkin Cheb")
    ax1.spy(A0.todense())
    ax1.set_title("PG Chebyshev")
    ax2.spy(A1.todense())
    ax2.set_title("PG Legendre")
    # plt.show()
