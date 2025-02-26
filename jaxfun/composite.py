from __future__ import annotations

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
from sympy import Number

from jaxfun.Basespace import BaseSpace, Domain, n
from jaxfun.Chebyshev import Chebyshev
from jaxfun.coordinates import CoordSys
from jaxfun.Jacobi import Jacobi
from jaxfun.Legendre import Legendre
from jaxfun.utils.common import matmat

direct_sum_symbol = "\u2295"


class BoundaryConditions(dict):
    """Boundary conditions as a dictionary"""

    def __init__(self, bc: dict, domain: Domain | None = None) -> None:
        if domain is None:
            domain = Domain(-1, 1)
        bcs = {"left": {}, "right": {}}
        bcs.update(copy.deepcopy(bc))
        dict.__init__(self, bcs)

    def orderednames(self) -> list[str]:
        return ["L" + bci for bci in sorted(self["left"].keys())] + [
            "R" + bci for bci in sorted(self["right"].keys())
        ]

    def orderedvals(self) -> list[Number]:
        ls = []
        for lr in ("left", "right"):
            for key in sorted(self[lr].keys()):
                val = self[lr][key]
                ls.append(val[1] if isinstance(val, tuple | list) else val)
        return ls

    def num_bcs(self) -> int:
        return len(self.orderedvals())

    def num_derivatives(self) -> int:
        n = {"D": 0, "R": 0, "N": 1, "N2": 2, "N3": 3, "N4": 4}
        num_diff = 0
        for val in self.values():
            for k in val:
                num_diff += n[k]
        return num_diff

    def is_homogeneous(self) -> bool:
        for val in self.values():
            for v in val.values():
                if v != 0:
                    return False
        return True

    def get_homogeneous(self) -> BoundaryConditions:
        bc = {}
        for k, v in self.items():
            bc[k] = {}
            for s in v:
                bc[k][s] = 0
        return BoundaryConditions(bc)


class Composite(BaseSpace):
    """Space created by combining orthogonal basis functions

    The stencil matrix is computed from the given boundary conditions, but
    may also be given explicitly.
    """

    def __init__(
        self,
        N: int,
        orthogonal: BaseSpace,
        bcs: BoundaryConditions,
        domain: Domain = None,
        name: str = "Composite",
        fun_str: str = "phi",
        system: CoordSys = None,
        stencil: dict = None,
        alpha: Number = 0,
        beta: Number = 0,
        scaling: sp.Expr = sp.S.One,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        BaseSpace.__init__(self, N, domain, system=system, name=name, fun_str=fun_str)
        self.orthogonal = orthogonal(
            N, domain=domain, alpha=alpha, beta=beta, system=system
        )
        self.bcs = BoundaryConditions(bcs)
        if stencil is None:
            stencil = get_stencil_matrix(self.bcs, self.orthogonal)
        self.scaling = scaling
        self.stencil = {(si[0]): si[1] / scaling for si in sorted(stencil.items())}
        self.S = BCOO.from_scipy_sparse(self.stencil_to_scipy_sparse())

    @partial(jax.jit, static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> Array:
        return self.orthogonal.quad_points_and_weights(N)

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        return self.orthogonal.evaluate(X, self.to_orthogonal(c))

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> float:
        return self.orthogonal.backward(self.to_orthogonal(c), kind, N)

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_basis_derivative(self, X: Array, k: int = 0) -> Array:
        P: Array = self.orthogonal.evaluate_basis_derivative(X, k)
        return self.apply_stencil_right(P)

    @partial(jax.jit, static_argnums=0)
    def vandermonde(self, X: Array) -> Array:
        P: Array = self.orthogonal.evaluate_basis_derivative(X, 0)
        return self.apply_stencil_right(P)

    @partial(jax.jit, static_argnums=(0, 2))
    def eval_basis_function(self, X: float, i: int) -> float:
        row: Array = self.get_stencil_row(i)
        psi: Array = jnp.array(
            [self.orthogonal.eval_basis_function(X, i + j) for j in self.stencil]
        )
        return matmat(row, psi)

    @partial(jax.jit, static_argnums=0)
    def eval_basis_functions(self, X: float) -> Array:
        P: Array = self.orthogonal.eval_basis_functions(X)
        return self.apply_stencil_right(P)

    def get_stencil_row(self, i: int) -> Array:
        return self.S[i].data[: len(self.stencil)]

    def stencil_to_scipy_sparse(self) -> scipy_sparse.spmatrix:
        k = jnp.arange(self.N - 1)
        return scipy_sparse.diags(
            [
                jnp.atleast_1d(
                    sp.lambdify(
                        n, val, modules=["jax", {"gamma": jax.scipy.special.gamma}]
                    )(k)
                ).astype(float)
                for val in self.stencil.values()
            ],
            [key for key in self.stencil],
            shape=(self.N - self.bcs.num_bcs(), self.N),
        )

    @property
    def reference_domain(self) -> Domain:
        return self.orthogonal.reference_domain

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
        return a @ self.S.T

    @partial(jax.jit, static_argnums=0)
    def mass_matrix(self) -> BCOO:
        P: BCOO = self.orthogonal.mass_matrix()
        T = matmat((self.S * P.data[None, :]), self.S.T).data.reshape(
            (self.dim, -1)
        )  # sparse @ sparse -> dense (yet sparse format), so need to remove zeros
        return sparse.BCOO.fromdense(T, nse=2 * self.S.nse - self.S.shape[1])

    @partial(jax.jit, static_argnums=0)
    def forward(self, u: Array) -> Array:
        A = self.mass_matrix().todense()
        L = self.scalar_product(u)
        return jnp.linalg.solve(A, L)

    @partial(jax.jit, static_argnums=0)
    def scalar_product(self, u: Array) -> Array:
        P: Array = self.orthogonal.scalar_product(u)
        return self.apply_stencil_right(P)

    @property
    def dim(self) -> int:
        return self.N - self.bcs.num_bcs()

    def get_homogeneous(self) -> Composite:
        bc = self.bcs.get_homogeneous()
        return Composite(
            N=self.N,
            orthogonal=self.orthogonal.__class__,
            bcs=bc,
            domain=self.domain,
            name=self.name + "0",
            fun_str=self.fun_str,
            system=self.system,
            stencil=self.stencil,
            alpha=self.orthogonal.alpha,
            beta=self.orthogonal.beta,
        )

    def get_padded(self, N: int) -> Composite:
        bc = self.bcs.get_homogeneous()
        return Composite(
            N=N,
            orthogonal=self.orthogonal.__class__,
            bcs=bc,
            domain=self.domain,
            name=self.name + "p",
            fun_str=self.fun_str + "p",
            system=self.system,
            stencil=self.stencil,
            alpha=self.orthogonal.alpha,
            beta=self.orthogonal.beta,
        )


class BCGeneric(Composite):
    """Space used to fix nonzero boundary conditions"""

    def __init__(
        self,
        N: int,
        orthogonal: BaseSpace,
        bcs: dict,
        domain: Domain = None,
        name: str = "BCGeneric",
        fun_str: str = "B",
        system: CoordSys = None,
        stencil: dict = None,
        alpha: Number = 0,
        beta: Number = 0,
        M: int = 0,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        bcs = BoundaryConditions(bcs, domain=domain)
        BaseSpace.__init__(self, N, domain, system=system, name=name, fun_str=fun_str)
        self.bcs = bcs
        self.stencil = None
        self.M = M
        self.orthogonal = orthogonal(
            bcs.num_bcs() + bcs.num_derivatives(),
            domain=domain,
            alpha=alpha,
            beta=beta,
            system=system,
        )
        S = get_bc_basis(bcs, self.orthogonal)
        self.orthogonal.N = S.shape[1]
        self.orthogonal.M = M
        self.S = BCOO.fromdense(S.__array__().astype(float))

    @property
    def dim(self) -> int:
        return self.N

    def bnd_vals(self) -> Array:
        return jnp.array(self.bcs.orderedvals())

    @partial(jax.jit, static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> Array:
        N = self.M if N == 0 else N
        return self.orthogonal.quad_points_and_weights(N)


class DirectSum:
    def __init__(self, a: BaseSpace, b: BaseSpace) -> None:
        assert isinstance(b, BCGeneric)
        self.basespaces: list[BaseSpace] = [a, b]
        self.bcs = b.bcs
        self.name = direct_sum_symbol.join([i.name for i in [a, b]])
        self.system = a.system
        
    def __getitem__(self, i: int) -> BaseSpace:
        return self.basespaces[i]

    def __len__(self) -> int:
        return len(self.basespaces)

    def mesh(self, kind: str = "quadrature", N: int = 0) -> Array:
        return self.basespaces[0].mesh(kind=kind, N=N)

    def bnd_vals(self) -> Array:
        return self.basespaces[1].bnd_vals()

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        return self.basespaces[0].evaluate(X, c) + self.basespaces[1].evaluate(
            X, self.bnd_vals()
        )

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        return self.basespaces[0].backward(c, kind, N) + self.basespaces[1].backward(
            self.bnd_vals(), kind, N
        )


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
    >>> C = Composite(Chebyshev.Chebyshev, 10, {"left": {"N": 0}, "right": {"N": 0}})
    >>> C.stencil
    {0: 1, 2: -n**2/(n**2 + 4*n + 4)}
    """
    if "".join(bcs.orderednames()) == "LDRD" and isinstance(
        orthogonal, Chebyshev | Legendre
    ):
        return {0: 1, 2: -1}
    if "".join(bcs.orderednames()) == "LDLNRDRN":
        if orthogonal.name == "Legendre":
            return {
                0: 1,
                2: 2 * (-2 * n - 5) / (2 * n + 7),
                4: (2 * n + 3) / (2 * n + 7),
            }
        elif orthogonal.name == "Chebyshev":
            return {0: 1, 2: 2 * (-n - 2) / (n + 3), 4: (n + 1) / (n + 3)}

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
        if s != 0:
            d[i + 1] = s
    return d


def get_bc_basis(bcs: BoundaryConditions, orthogonal: Jacobi) -> sp.Matrix:
    """Return boundary basis satisfying `bcs`.

    Parameters
    ----------
    bcs : dict
        The boundary conditions in dictionary form, see
        :class:`.BoundaryConditions`.

    family : str
        Choose one of

        - ``Chebyshev``
        - ``Chebyshevu``
        - ``Legendre``
        - ``Ultraspherical``
        - ``Jacobi``
        - ``Laguerre``

    alpha, beta : numbers, optional
        The Jacobi parameters, used only for ``Jacobi`` or ``Ultraspherical``

    gn : Scaling function for Jacobi polynomials

    """
    from sympy.matrices.common import NonInvertibleMatrixError

    bcs = BoundaryConditions(bcs)

    def _computematrix(first):
        bc = {"D": 0, "N": 1, "N2": 2, "N3": 3, "N4": 4}
        lr = {"L": 0, "R": 1}
        lra = {"L": "left", "R": "right"}
        s = []
        for key in bcs.orderednames():
            k, v = key[0], key[1:]
            if v in "WR":
                k0 = 0 if v == "R" else 1
                alfa = bcs[lra[k]][v][0]
                f = [
                    orthogonal.bnd_values(k=k0)[lr[k]],
                    orthogonal.bnd_values(k=k0 + 1)[lr[k]],
                ]
                s.append(
                    [
                        sp.simplify(f[0](j) + alfa * f[1](j))
                        for j in range(first, first + bcs.num_bcs())
                    ]
                )
            else:
                f = orthogonal.bnd_values(k=bc[v])[lr[k]]
                s.append(
                    [sp.simplify(f(j)) for j in range(first, first + bcs.num_bcs())]
                )

        A = sp.Matrix(s)
        s = sp.simplify(A.solve(sp.eye(bcs.num_bcs())).T)
        return s

    first_basis = bcs.num_derivatives()
    first = 0
    for first in range(first_basis + 1):
        try:
            s = _computematrix(first)
            break
        except NonInvertibleMatrixError:
            continue

    sol = sp.Matrix(np.zeros((bcs.num_bcs(), first + bcs.num_bcs())))
    sol[:, first:] = s
    return sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Chebyshev import Chebyshev
    from Legendre import Legendre

    from jaxfun.arguments import TestFunction, TrialFunction
    from jaxfun.composite import Composite
    from jaxfun.inner import inner

    n = sp.Symbol("n", positive=True, integer=True)
    N = 50
    biharmonic = {"left": {"D": 0, "N": 0}, "right": {"D": 0, "N": 0}}
    dirichlet = {"left": {"D": 0}, "right": {"D": 0}}
    C = Composite(N, Chebyshev, dirichlet, name="C")

    v = jax.random.normal(jax.random.PRNGKey(1), shape=(N, N))
    g = C.S @ v @ C.S.T
    g1 = C.apply_stencil_galerkin(v)

    D = C.stencil_to_scipy_sparse()
    vn = v.__array__()
    gn = D @ vn @ D.T

    assert jnp.linalg.norm(gn - g) < 1e-7
    assert jnp.linalg.norm(gn - g1) < 1e-7

    # Galerkin (dense)
    u = TrialFunction(C, name="u")
    v = TestFunction(C, name="v")
    x = C.system.x
    D = inner(v * sp.diff(u, x, 2), sparse=True, sparse_tol=1000)

    # Petrov-Galerkin method (https://www.duo.uio.no/bitstream/handle/10852/99687/1/PGpaper.pdf)
    G = Composite(N, Chebyshev, dirichlet, scaling=n + 1, name="G")
    PG = Composite(
        N + 2,
        Chebyshev,
        biharmonic,
        stencil={
            0: 1 / (2 * sp.pi * (n + 1) * (n + 2)),
            2: -1 / (sp.pi * (n**2 + 4 * n + 3)),
            4: 1 / (2 * sp.pi * (n + 2) * (n + 3)),
        },
        name="PG",
    )
    L = Composite(N, Legendre, dirichlet, scaling=n + 1, name="L")
    LG = Composite(
        N + 2,
        Legendre,
        biharmonic,
        stencil={
            0: 1 / (2 * (2 * n + 3)),
            2: -(2 * n + 5) / (2 * n + 7) / (2 * n + 3),
            4: 1 / (2 * (2 * n + 7)),
        },
        name="LG",
    )
    A0 = inner(
        TestFunction(PG) * sp.diff(TrialFunction(G), x, 2),
        sparse=True,
        sparse_tol=1000,
    )  # bidiagonal
    A1 = inner(
        TestFunction(LG) * sp.diff(TrialFunction(L), x, 2),
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
    plt.show()
