from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from scipy import sparse as scipy_sparse
from sympy import Number

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Jacobi import Jacobi
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.utils.common import Domain, matmat, n

direct_sum_symbol = "\u2295"


class BoundaryConditions(dict):
    """Container for 1D left/right boundary conditions.

    Stores boundary data under the keys "left" and "right". Each side
    maps condition code -> value. Supported codes (examples):

      D  : Dirichlet
      N  : Neumann (1st derivative)
      N2 : 2nd derivative (etc.)
      R  : Robin (tuple (alpha, value))
      W  : Weighted (tuple (alpha, value))

    Values can be:
      * Number (homogeneous / inhomogeneous)
      * Tuple (alpha, Number) for Robin / weighted forms

    Args:
        bc: User dictionary (partial). Missing sides are filled.
        domain: Physical domain (unused here, kept for future features).

    Attributes:
        left/right: Dicts of condition_code -> value.
    """

    def __init__(self, bc: dict, domain: Domain | None = None) -> None:
        if domain is None:
            domain = Domain(-1, 1)
        bcs = {"left": {}, "right": {}}
        bcs.update(copy.deepcopy(bc))
        dict.__init__(self, bcs)

    def orderednames(self) -> list[str]:
        """Return ordered boundary condition codes (prefixed with L/R)."""
        return ["L" + bci for bci in sorted(self["left"].keys())] + [
            "R" + bci for bci in sorted(self["right"].keys())
        ]

    def orderedvals(self) -> list[Number]:
        """Return boundary condition values in same order as orderednames()."""
        ls = []
        for lr in ("left", "right"):
            for key in sorted(self[lr].keys()):
                val = self[lr][key]
                ls.append(val[1] if isinstance(val, tuple | list) else val)
        return ls

    def num_bcs(self) -> int:
        """Return number of scalar boundary conditions."""
        return len(self.orderedvals())

    def num_derivatives(self) -> int:
        """Return total derivative order count (used for basis offset)."""
        n = {"D": 0, "R": 0, "N": 1, "N2": 2, "N3": 3, "N4": 4}
        num_diff = 0
        for val in self.values():
            for k in val:
                num_diff += n[k]
        return num_diff

    def is_homogeneous(self) -> bool:
        """Return True if all boundary values (incl. Robin) equal zero."""
        for val in self.values():
            for v in val.values():
                if v != 0:
                    return False
        return True

    def get_homogeneous(self) -> BoundaryConditions:
        """Return a copy with all boundary values set to zero."""
        bc = {}
        for k, v in self.items():
            bc[k] = {}
            for s in v:
                bc[k][s] = 0
        return BoundaryConditions(bc)


class Composite(OrthogonalSpace):
    """Composite basis enforcing boundary conditions via a stencil.

    Builds a constrained basis φ_i = Σ_j S_{ij} P_{j} where P_j are
    orthogonal polynomials (Chebyshev/Legendre/Jacobi). The stencil is
    selected/derived from BoundaryConditions and converted into a sparse
    matrix S (BCOO). Basis reduction removes degrees constrained by BCs.

    Args:
        N: Target (unconstrained) number of modes of underlying orthogonal.
        orthogonal: Underlying orthogonal basis class (e.g. Chebyshev).
        bcs: BoundaryConditions specification.
        domain: Physical domain (defaults to [-1, 1]).
        name: Space name.
        fun_str: Symbol stem for basis functions.
        system: Optional coordinate system.
        stencil: Optional custom stencil dict {shift: sympy_expr}.
        alpha: Jacobi alpha (for Jacobi-based bases).
        beta: Jacobi beta.
        scaling: SymPy expression scaling the stencil diagonals.

    Attributes:
        orthogonal: Instance of underlying orthogonal basis.
        stencil: Ordered dict of diagonal shift -> expression / scaling.
        S: Sparse (BCOO) stencil matrix.
        scaling: Scaling expression applied to user stencil.
    """

    def __init__(  # noqa: D401  (docstring above)
        self,
        N: int,
        orthogonal: type[Jacobi],
        bcs: BoundaryConditions | dict,
        domain: Domain | None = None,
        name: str = "Composite",
        fun_str: str = "phi",
        system: CoordSys | None = None,
        stencil: dict | None = None,
        alpha: Number | float = 0,
        beta: Number | float = 0,
        scaling: sp.Expr = sp.S.One,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self.orthogonal: Jacobi = orthogonal(
            N, domain=domain, alpha=alpha, beta=beta, system=system
        )
        self.bcs: BoundaryConditions = BoundaryConditions(bcs)
        if stencil is None:
            assert isinstance(self.orthogonal, Jacobi), (
                "Automatic stencil derivation only supported for Jacobi-based "
                "orthogonal bases. Provide custom stencil dict otherwise."
            )
            stencil = get_stencil_matrix(self.bcs, self.orthogonal)
        self.scaling = scaling
        self.stencil = {(si[0]): si[1] / scaling for si in sorted(stencil.items())}
        self.S = BCOO.from_scipy_sparse(self.stencil_to_scipy_sparse())

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> tuple[Array, Array]:
        """Return quadrature nodes/weights (delegated to underlying basis)."""
        return self.orthogonal.quad_points_and_weights(N)

    @jax.jit(static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        """Evaluate constrained expansion at X with composite coeffs c."""
        return self.orthogonal.evaluate(X, self.to_orthogonal(c))

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> float:
        """Inverse transform (physical -> coefficients) via underlying basis."""
        return self.orthogonal.backward(self.to_orthogonal(c), kind, N)

    @jax.jit(static_argnums=(0, 2))
    def evaluate_basis_derivative(self, X: Array, k: int = 0) -> Array:
        """Return k-th derivative Vandermonde (constrained)."""
        P: Array = self.orthogonal.evaluate_basis_derivative(X, k)
        return self.apply_stencil_right(P)

    @jax.jit(static_argnums=0)
    def vandermonde(self, X: Array) -> Array:
        """Return (constrained) Vandermonde matrix at sample points X."""
        P: Array = self.orthogonal.evaluate_basis_derivative(X, 0)
        return self.apply_stencil_right(P)

    @jax.jit(static_argnums=(0, 2))
    def eval_basis_function(self, X: float, i: int) -> float:
        """Evaluate single constrained basis function φ_i at X."""
        row: Array = self.get_stencil_row(i)
        psi: Array = jnp.array(
            [self.orthogonal.eval_basis_function(X, i + j) for j in self.stencil]
        )
        return matmat(row, psi)

    @jax.jit(static_argnums=0)
    def eval_basis_functions(self, X: float) -> Array:
        """Evaluate all constrained basis functions at X."""
        P: Array = self.orthogonal.eval_basis_functions(X)
        return self.apply_stencil_right(P)

    def get_stencil_row(self, i: int) -> Array:
        """Return nonzero stencil row data for basis index i."""
        return self.S[i].data[: len(self.stencil)]

    def stencil_width(self) -> int:
        """Return max diagonal shift minus min shift (stencil width)."""
        return max(self.stencil) - min(self.stencil)

    def stencil_to_scipy_sparse(self) -> scipy_sparse.spmatrix:
        """Convert symbolic stencil to scipy sparse diagonal matrix."""
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
            shape=(self.N - self.stencil_width(), self.N),
        )

    @property
    def reference_domain(self) -> Domain:
        """Return reference domain of underlying orthogonal basis."""
        return self.orthogonal.reference_domain

    @jax.jit(static_argnums=0)
    def to_orthogonal(self, a: Array) -> Array:
        """Map composite coefficients -> underlying orthogonal coefficients."""
        return a @ self.S

    @jax.jit(static_argnums=0)
    def apply_stencil_galerkin(self, b: Array) -> Array:
        """Apply stencil on both sides (Galerkin mass-like transform)."""
        return self.S @ b @ self.S.T

    @jax.jit(static_argnums=0)
    def apply_stencils_petrovgalerkin(self, b: Array, P: BCOO) -> Array:
        """Apply test (S) and trial (P) stencils (Petrov-Galerkin)."""
        return self.S @ b @ P.T

    @jax.jit(static_argnums=0)
    def apply_stencil_left(self, b: Array) -> Array:
        """Left-multiply by stencil (test projection)."""
        return self.S @ b

    @jax.jit(static_argnums=0)
    def apply_stencil_right(self, a: Array) -> Array:
        """Right-multiply by stencil transpose (trial projection)."""
        return a @ self.S.T

    @jax.jit(static_argnums=0)
    def mass_matrix(self) -> BCOO:
        """Return constrained mass matrix in sparse BCOO format."""
        P: BCOO = self.orthogonal.mass_matrix()
        T = matmat((self.S * P.data[None, :]), self.S.T).data.reshape(
            (self.dim, -1)
        )  # sparse @ sparse -> dense (yet sparse format), so need to remove zeros
        return sparse.BCOO.fromdense(T, nse=2 * self.S.nse - self.S.shape[1])

    @jax.jit(static_argnums=0)
    def forward(self, u: Array) -> Array:
        """Project physical samples u -> constrained coefficients."""
        A = self.mass_matrix().todense()
        L = self.scalar_product(u)
        return jnp.linalg.solve(A, L)

    @jax.jit(static_argnums=0)
    def scalar_product(self, u: Array) -> Array:
        """Return right-hand side inner product vector <u, φ_i>."""
        P: Array = self.orthogonal.scalar_product(u)
        return self.apply_stencil_right(P)

    @property
    def dim(self) -> int:
        """Return dimension of composite space."""
        return self.orthogonal.dim - self.stencil_width()

    def get_homogeneous(self) -> Composite:
        """Return new Composite with homogeneous boundary values."""
        bc = self.bcs.get_homogeneous()
        return Composite(
            N=self.orthogonal.dim,
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
        """Return Composite enlarged (padded) to size N (same stencil)."""
        return Composite(
            N=N,
            orthogonal=self.orthogonal.__class__,
            bcs=self.bcs,
            domain=self.domain,
            name=self.name + "p",
            fun_str=self.fun_str + "p",
            system=self.system,
            stencil=self.stencil,
            alpha=self.orthogonal.alpha,
            beta=self.orthogonal.beta,
        )


class BCGeneric(Composite):
    """Basis spanning only boundary-constraint enforcing functions.

    All degrees of freedom correspond to boundary modes; num_dofs == 0.
    Used to construct direct sums (solution space ⊕ boundary lift space).
    """

    def __init__(  # noqa: D401
        self,
        N: int,
        orthogonal: type[Jacobi],
        bcs: dict,
        domain: Domain | None = None,
        name: str = "BCGeneric",
        fun_str: str = "B",
        system: CoordSys | None = None,
        stencil: dict | None = None,
        alpha: Number | float = 0,
        beta: Number | float = 0,
        num_quad_points: int = 0,
    ) -> None:
        domain = Domain(-1, 1) if domain is None else domain
        bcs: BoundaryConditions = BoundaryConditions(bcs, domain=domain)
        OrthogonalSpace.__init__(
            self, N, domain=domain, system=system, name=name, fun_str=fun_str
        )
        self.bcs = bcs
        self.stencil = None
        self._num_quad_points = num_quad_points
        self.orthogonal = orthogonal(
            bcs.num_bcs() + bcs.num_derivatives(),
            domain=domain,
            alpha=alpha,
            beta=beta,
            system=system,
        )
        S = get_bc_basis(bcs, self.orthogonal)
        self.orthogonal.N = S.shape[1]
        self.orthogonal._num_quad_points = num_quad_points
        self.S = BCOO.fromdense(S.__array__().astype(float))

    @property
    def dim(self) -> int:
        """Return dimension of boundary space."""
        return self.S.shape[0]

    @property
    def num_dofs(self) -> int:
        """Return number of free DOFs (always zero for pure BC basis)."""
        return 0

    def bnd_vals(self) -> Array:
        """Return ordered boundary values vector."""
        return jnp.array(self.bcs.orderedvals())

    @jax.jit(static_argnums=(0, 1))
    def quad_points_and_weights(self, N: int = 0) -> Array:
        """Quadrature nodes/weights (override to enforce num_quad_points)."""
        N = self.num_quad_points if N == 0 else N
        return self.orthogonal.quad_points_and_weights(N)


class DirectSum:
    """Direct sum V = Composite ⊕ BCGeneric lifting boundary data.

    Evaluation adds the homogeneous solution part and boundary lift
    expansion. This preserves linear independence and imposes BCs.

    Args:
        a: Composite (homogeneous) space.
        b: BCGeneric boundary lifting space.

    Attributes:
        basespaces: [a, b]
        bcs: BoundaryConditions reference.
        num_dofs: Free DOFs (from Composite part).
    """

    def __init__(self, a: Composite | OrthogonalSpace, b: BCGeneric) -> None:
        assert isinstance(b, BCGeneric)
        self.basespaces: tuple[Composite, BCGeneric] = (a, b)
        self.bcs = b.bcs
        self.name = direct_sum_symbol.join([i.name for i in [a, b]])
        self.system = a.system
        self.N = a.N
        self._num_quad_points = a._num_quad_points
        self.map_reference_domain = a.map_reference_domain
        self.map_true_domain = a.map_true_domain

    def __getitem__(self, i: int) -> Composite | BCGeneric:
        """Return i-th summand."""
        return self.basespaces[i]

    def __len__(self) -> int:
        """Return number of summands (always 2)."""
        return len(self.basespaces)

    def mesh(self, kind: str = "quadrature", N: int = 0) -> Array:
        """Return mesh from homogeneous Composite summand."""
        return self.basespaces[0].mesh(kind=kind, N=N)

    def bnd_vals(self) -> Array:
        """Return boundary lifting values (from BCGeneric)."""
        return self.basespaces[1].bnd_vals()

    @property
    def dim(self) -> int:
        """Return total dimension (homogeneous + boundary)."""
        return self.basespaces[0].dim + self.basespaces[1].dim

    @property
    def num_dofs(self) -> int:
        """Return free degrees of freedom (Composite part)."""
        return self.basespaces[0].num_dofs

    @jax.jit(static_argnums=0)
    def evaluate(self, X: float, c: Array) -> float:
        """Evaluate direct-sum function at X with composite coeffs c."""
        return self.basespaces[0].evaluate(X, c) + self.basespaces[1].evaluate(
            X, self.bnd_vals()
        )

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        """Backward transform (composite + boundary contribution)."""
        return self.basespaces[0].backward(c, kind, N) + self.basespaces[1].backward(
            self.bnd_vals(), kind, N
        )


def get_stencil_matrix(bcs: BoundaryConditions, orthogonal: Jacobi) -> dict:
    """Derive symbolic stencil mapping orthogonal -> constrained basis.

    For BC set, solve linear relations among boundary traces of shifted
    orthogonal functions to express ψ_i = Σ d_k P_{i+k}. Returns dict
    {shift: sympy_expression}. Special-cases some frequent BC patterns.

    Args:
        bcs: BoundaryConditions object.
        orthogonal: Underlying orthogonal basis instance (Jacobi derived).

    Returns:
        Dictionary shift -> SymPy expression (d_shift).

    Example:
        Neumann Chebyshev:
            ψ_i = T_i - i^2/(i^2+4i+4) T_{i+2}
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
    """Return lifting basis satisfying boundary conditions exactly.

    Constructs a matrix S whose rows span functions meeting bcs. The
    method searches for a starting index producing invertible boundary
    evaluation matrix, then solves for canonical columns.

    Args:
        bcs: BoundaryConditions dictionary or raw dict.
        orthogonal: Underlying orthogonal family instance.

    Returns:
        SymPy Matrix of shape (num_bcs, K) giving expansion coefficients.
    """
    from sympy.matrices.common import NonInvertibleMatrixError

    bcs = BoundaryConditions(bcs)

    def _computematrix(first) -> sp.Matrix:
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
    s = None
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

    from jaxfun.galerkin.arguments import TestFunction, TrialFunction
    from jaxfun.galerkin.Chebyshev import Chebyshev
    from jaxfun.galerkin.inner import inner
    from jaxfun.galerkin.Legendre import Legendre

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
    gn = D @ vn @ D.T  # ty:ignore[unresolved-attribute]

    assert jnp.linalg.norm(gn - g) < 1e-7
    assert jnp.linalg.norm(gn - g1) < 1e-7

    # Galerkin (dense)
    u = TrialFunction(C, name="u")
    v = TestFunction(C, name="v")
    x = C.system.x  # ty:ignore[possibly-missing-attribute]
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
    ax0.spy(D.todense())  # ty:ignore[possibly-missing-attribute]
    ax0.set_title("Galerkin Cheb")
    ax1.spy(A0.todense())  # ty:ignore[possibly-missing-attribute]
    ax1.set_title("PG Chebyshev")
    ax2.spy(A1.todense())  # ty:ignore[possibly-missing-attribute]
    ax2.set_title("PG Legendre")
    plt.show()
