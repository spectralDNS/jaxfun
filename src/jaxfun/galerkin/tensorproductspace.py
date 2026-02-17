from __future__ import annotations

import copy
import itertools
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING, TypeGuard

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array
from scipy import sparse as scipy_sparse

from jaxfun.coordinates import CoordSys

if TYPE_CHECKING:
    from jaxfun.galerkin import JAXFunction
from jaxfun.utils.common import eliminate_near_zeros, jit_vmap, lambdify

from .composite import BCGeneric, BoundaryConditions, Composite, DirectSum
from .orthogonal import OrthogonalSpace

tensor_product_symbol = "\u2297"
multiplication_sign = "\u00d7"


class TensorProductSpace:
    """d-dimensional tensor product of 1D BaseSpace instances.

    Provides:
        * Logical / Cartesian mesh generation
        * Forward / backward spectral transforms (dimension-wise vmap)
        * Series evaluation on tensor-product meshes or scattered points
        * Support for heterogeneous underlying bases (Fourier / polynomial)
        * Automatic mapping between true and reference domains per axis

    Boundary condition handling:
        Each 1D factor may itself be a Composite/DirectSum (BC aware). This
        class itself stays agnostic; non-homogeneous BC logic is handled by
        DirectSumTPS wrapper.

    Attributes:
        basespaces: Ordered list of 1D BaseSpace objects.
        system: Coordinate system (created if None).
        tensorname: Pretty tensor product name (e.g. "V0⊗V1").
        name: User label.

    Notes:
        Returned coefficient/tensor shapes follow the ordering of
        basespaces. Methods vectorize over trailing axes with vmap.
    """

    is_transient = False

    def __init__(
        self,
        basespaces: Sequence[OrthogonalSpace],
        system: CoordSys | None = None,
        name: str = "TPS",
    ) -> None:
        from jaxfun.coordinates import CartCoordSys, x, y, z

        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[len(basespaces)])
            if system is None
            else system
        )
        self.basespaces: list[OrthogonalSpace] = list(basespaces)
        self.name = name
        self.system: CoordSys = system
        self.tensorname = tensor_product_symbol.join([b.name for b in basespaces])

    def __len__(self) -> int:
        """Return number of spatial dimensions."""
        return len(self.basespaces)

    def __iter__(self) -> Iterator[OrthogonalSpace]:
        """Iterate over factor spaces."""
        return iter(self.basespaces)

    def __getitem__(self, i: int) -> OrthogonalSpace:
        """Return i-th factor space."""
        return self.basespaces[i]

    @property
    def dims(self) -> int:
        """Return number of spatial dimensions."""
        return len(self)

    @property
    def rank(self) -> int:
        """Return tensor rank (0 for scalar-valued space)."""
        return 0

    @property
    def is_orthogonal(self) -> bool:
        """Return True if underlying bases are all orthogonal."""
        return all(space.is_orthogonal for space in self.basespaces)

    def shape(self) -> tuple[int, ...]:
        """Return raw modal shape (N0, N1, ...)."""
        return tuple([space.N for space in self.basespaces])

    @property
    def dim(self) -> int:
        """Return total number of modes."""
        return int(jnp.prod(jnp.array([space.dim for space in self.basespaces])))

    @property
    def num_dofs(self) -> tuple[int, ...]:
        """Return tuple of active degrees of freedom per axis."""
        return tuple(space.num_dofs for space in self.basespaces)

    def mesh(
        self,
        kind: str = "quadrature",
        N: tuple[int, ...] | None = None,
        broadcast: bool = True,
    ) -> tuple[Array, ...]:
        """Return tensor mesh (as tuple of arrays) in true domain.

        Args:
            kind: 'quadrature' or 'uniform'.
            N: Optional per-axis counts (defaults each to space.N).
            broadcast: If True broadcast each axis array to nd-grid shape.

        Returns:
            Tuple (X0, X1, ...) each either 1D or broadcasted.
        """
        mesh = []
        if N is None:
            N = self.shape()
        for ax, space in enumerate(self.basespaces):
            X = space.mesh(kind, N[ax])
            mesh.append(self.broadcast_to_ndims(X, ax) if broadcast else X)
        return tuple(mesh)

    def flatmesh(
        self,
        kind: str = "quadrature",
        N: tuple[int] | None = None,
    ) -> Array:
        """Return flattened list of all coordinate tuples.

        Args:
            kind: Sampling kind.
            N: Optional per-axis counts.

        Returns:
            Array (M, dims) with Cartesian products of mesh points.
        """
        mesh = self.mesh(kind, N, broadcast=False)
        return jnp.array(
            list(itertools.product(*[m.flatten() for m in mesh])), dtype=mesh[0].dtype
        )

    def cartesian_mesh(
        self, kind: str = "quadrature", N: tuple[int, ...] | None = None
    ) -> tuple[Array, ...]:
        """Return mapped Cartesian mesh (position vector evaluation)."""
        rv = self.system.position_vector(False)
        assert isinstance(rv, sp.Tuple)
        x = self.system.base_scalars()
        xj = self.mesh(kind, N, True)
        mesh = []
        for r in rv:
            mesh.append(lambdify(x, r, modules="jax")(*xj))
        return tuple(mesh)

    def broadcast_to_ndims(self, x: Array, axis: int = 0) -> Array:
        """Return 1D array x expanded to full tensor-product shape."""
        s = [jnp.newaxis] * len(self)
        s[axis] = slice(None)
        return x[tuple(s)]

    def map_expr_true_domain(self, u: sp.Expr) -> sp.Expr:
        """Map reference variables in expression u to true domain coords."""
        for space in self.basespaces:
            u = space.map_expr_true_domain(u)
        return u

    def map_expr_reference_domain(self, u: sp.Expr) -> sp.Expr:
        """Map true domain variables in expression u to reference coords."""
        for space in self.basespaces:
            u = space.map_expr_reference_domain(u)
        return u

    # Cannot jit_vmap since tensor product mesh.
    @jax.jit(static_argnums=(0, 3))
    def evaluate_mesh(
        self, x: list[Array], c: Array, use_einsum: bool = False
    ) -> Array:
        """Evaluate expansion on provided tensor-product mesh arrays.

        Args:
            x: List of per-axis coordinate arrays (broadcasted or 1D).
            c: Coefficient tensor shaped (N0, N1, ...).
            use_einsum: If True use einsum path; else iterative vmaps.

        Returns:
            Array of evaluated field values with broadcast shape.
        """
        dim: int = len(self)
        if dim == 2:
            if not use_einsum:
                for i, (xi, ax) in enumerate(zip(x, range(dim), strict=False)):
                    axi: int = dim - 1 - ax
                    c = jax.vmap(
                        self.basespaces[i].evaluate, in_axes=(None, axi), out_axes=axi
                    )(
                        jnp.atleast_1d(
                            self.basespaces[i].map_reference_domain(xi).squeeze()
                        ),
                        c,
                    )
            else:
                T0, T1 = self.basespaces
                C0 = T0.eval_basis_functions(
                    jnp.atleast_1d(T0.map_reference_domain(x[0]).squeeze())
                )
                C1 = T1.eval_basis_functions(
                    jnp.atleast_1d(T1.map_reference_domain(x[1]).squeeze())
                )
                return jnp.einsum("ij,jk,lk->il", C0, c, C1)
        else:
            if not use_einsum:
                for i, (xi, ax) in enumerate(zip(x, range(dim), strict=False)):
                    ax0, ax1 = set(range(dim)) - set((ax,))
                    c = jax.vmap(
                        jax.vmap(
                            self.basespaces[i].evaluate,
                            in_axes=(None, ax0),
                            out_axes=ax0,
                        ),
                        in_axes=(None, ax1),
                        out_axes=ax1,
                    )(
                        jnp.atleast_1d(
                            self.basespaces[i].map_reference_domain(xi).squeeze()
                        ),
                        c,
                    )
            else:
                T0, T1, T2 = self.basespaces
                C0 = T0.eval_basis_functions(
                    jnp.atleast_1d(T0.map_reference_domain(x[0]).squeeze())
                )
                C1 = T1.eval_basis_functions(
                    jnp.atleast_1d(T1.map_reference_domain(x[1]).squeeze())
                )
                C2 = T2.eval_basis_functions(
                    jnp.atleast_1d(T2.map_reference_domain(x[2]).squeeze())
                )
                c = jnp.einsum("ik,jl,nm,klm->ijn", C0, C1, C2, c)
        return c

    @jit_vmap(in_axes=(0, None, None), static_argnums=(0, 3), ndim=1)
    def evaluate(self, x: Array, c: Array, use_einsum: bool) -> Array:
        """Evaluate expansion at scattered points (per-axis samples).

        Args:
            x: Array of per-axis coordinates stacked (N, d).
            c: Coefficient tensor.
            use_einsum: Whether to use einsum contraction.

        Returns:
            Evaluated values with shape determined by leading dims of x.
        """
        dim = len(self)
        T = self.basespaces
        C = [
            T[i].eval_basis_functions(T[i].map_reference_domain(x[i]))
            for i in range(dim)
        ]
        if not use_einsum:
            c = C[0] @ c @ C[1] if dim == 2 else C[2] @ (C[1] @ (C[0] @ c))
        else:
            path = "i,j,ij" if dim == 2 else "i,j,k,ijk"
            c = jnp.einsum(path, *C, c)
        return c

    @jax.jit(static_argnums=(0, 3))
    def evaluate_derivative(
        self, x: list[Array], c: Array, k: tuple[int, ...]
    ) -> Array:
        """Evaluate expansion (with derivatives) on provided tensor-product mesh arrays.

        Args:
            x: List of per-axis coordinate arrays (broadcasted or 1D).
            c: Coefficient tensor shaped (N0, N1, ...).
            k: Derivative order for each axis.

        Returns:
            Array of evaluated field values with broadcast shape.
        """
        for i, Ti in enumerate(self.basespaces):
            Ci = Ti.evaluate_basis_derivative(
                Ti.map_reference_domain(x[i]).squeeze(), k[i]
            )
            c = jnp.tensordot(Ci, c, axes=(1, i), precision=jax.lax.Precision.HIGHEST)
            c = jnp.moveaxis(c, 0, i)
        return c

    def get_padded(self, N: tuple[int, ...]) -> TensorProductSpace:
        """Return new tensor space with each axis padded/truncated to N."""
        paddedspaces = [
            s.get_padded(n) for s, n in zip(self.basespaces, N, strict=False)
        ]
        return TensorProductSpace(
            paddedspaces, system=self.system, name=self.name + "p"
        )

    def get_orthogonal(self) -> TensorProductSpace:
        """Return underlying orthogonal basis instance."""
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return TensorProductSpace(
            orthogonal_spaces, system=self.system, name=self.name + "o"
        )

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(
        self, c: Array, kind: str = "quadrature", N: tuple[int] | None = None
    ) -> Array:
        """Jitted backward transform with optional padding."""
        dim: int = len(self)
        if dim == 2:
            for ax in range(dim):
                axi: int = dim - 1 - ax
                backward = partial(
                    self.basespaces[ax].backward,
                    kind=kind,
                    N=self.basespaces[ax].num_quad_points if N is None else N[ax],
                )
                c = jax.vmap(backward, in_axes=axi, out_axes=axi)(c)

        else:
            for ax in range(dim):
                backward = partial(
                    self.basespaces[ax].backward,
                    kind=kind,
                    N=self.basespaces[ax].num_quad_points if N is None else N[ax],
                )
                ax0, ax1 = set(range(dim)) - set((ax,))
                c = jax.vmap(
                    jax.vmap(backward, in_axes=ax0, out_axes=ax0),
                    in_axes=ax1,
                    out_axes=ax1,
                )(c)
        return c

    @jax.jit(static_argnums=0)
    def scalar_product(self, u: Array) -> Array:
        """Return tensor of inner products along each axis (separable)."""
        dim: int = len(self)
        sg = self.system.sg
        if sg != 1:
            sg = lambdify(self.system.base_scalars(), sg)(*self.mesh())
            u = u * sg
        if dim == 2:
            for ax in range(dim):
                axi: int = dim - 1 - ax
                u = jax.vmap(
                    self.basespaces[ax].scalar_product, in_axes=axi, out_axes=axi
                )(u)
        else:
            for ax in range(dim):
                ax0, ax1 = set(range(dim)) - set((ax,))
                u = jax.vmap(
                    jax.vmap(
                        self.basespaces[ax].scalar_product, in_axes=ax0, out_axes=ax0
                    ),
                    in_axes=ax1,
                    out_axes=ax1,
                )(u)
        return u

    @jax.jit(static_argnums=(0, 2))
    def forward(self, u: Array, N: tuple[int, ...] | None = None) -> Array:
        """Forward transform with optional truncation."""
        dim: int = len(self)
        if dim == 2:
            for ax in range(dim):
                axi: int = dim - 1 - ax
                u = jax.vmap(self.basespaces[ax].forward, in_axes=axi, out_axes=axi)(u)
        else:
            for ax in range(dim):
                ax0, ax1 = set(range(dim)) - set((ax,))
                u = jax.vmap(
                    jax.vmap(self.basespaces[ax].forward, in_axes=ax0, out_axes=ax0),
                    in_axes=ax1,
                    out_axes=ax1,
                )(u)
        return u


class VectorTensorProductSpace:
    """Vector-valued tensor product space.

    Represents a tuple of identical (or differing in boundary conditions)
    TensorProductSpace objects corresponding to vector components.

    Attributes:
        tensorspaces: Tuple of component tensor spaces.
        system: Shared coordinate system.
        name: Label.
        tensorname: Joined printable representation.
    """

    is_transient = False

    def __init__(
        self,
        tensorspace: TensorProductSpace | tuple[TensorProductSpace, ...],
        name: str = "VTPS",
    ) -> None:
        if isinstance(tensorspace, TensorProductSpace):
            n = len(tensorspace)
            tensorspaces: tuple[TensorProductSpace, ...] = (tensorspace,) * n
        else:
            tensorspaces = tensorspace
        self.tensorspaces = tensorspaces
        self.system: CoordSys = self.tensorspaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.tensorspaces])
        self.mesh = self.tensorspaces[0].mesh
        self.evaluate_mesh = self.tensorspaces[0].evaluate_mesh

    def __len__(self) -> int:
        """Return number of vector components."""
        return len(self.tensorspaces)

    def __iter__(self) -> Iterator[TensorProductSpace]:
        """Iterate over component tensor spaces."""
        return iter(self.tensorspaces)

    def __getitem__(self, i: int) -> TensorProductSpace:
        """Return component tensor space i."""
        return self.tensorspaces[i]

    @property
    def rank(self) -> int:
        """Return tensor rank (1 for vector fields)."""
        return 1

    @property
    def dims(self) -> int:
        """Return spatial dimension of each component space."""
        return len(self.tensorspaces[0])

    @property
    def dim(self) -> int:
        """Return total number of modes."""
        return sum([space.dim for space in self.tensorspaces])

    @property
    def num_dofs(self) -> tuple[int, ...]:
        """Return tuple of active degrees of freedom per axis."""
        return (self.dims,) + self.tensorspaces[0].num_dofs

    @property
    def is_orthogonal(self) -> bool:
        """Return True if underlying bases are all orthogonal."""
        return all(space.is_orthogonal for space in self.tensorspaces)

    def shape(self) -> tuple[int, ...]:
        """Return raw modal shape (N0, N1, ...)."""
        return (self.dims,) + self.tensorspaces[0].shape()

    @jit_vmap(in_axes=(0, None, None), static_argnums=(0, 3), ndim=1)
    def evaluate(self, x: Array, c: Array, use_einsum: bool) -> Array:
        """Evaluate vector expansion at scattered points.

        Args:
            x: Array of per-axis coordinates stacked (N, d).
            c: Coefficient array shaped (dims, N0, N1, ...).
            use_einsum: Whether to use einsum contraction.

        Returns:
            Evaluated values with shape determined by leading dims of x.
        """
        vals = []
        for i, space in enumerate(self.tensorspaces):
            ci = c[i]
            vi = space.evaluate(x, ci, use_einsum)
            vals.append(vi)
        return jnp.array(vals)

    @jit_vmap(in_axes=(0, None, None), static_argnums=(0, 3), ndim=1)
    def evaluate_derivative(self, x: Array, c: Array, k: tuple[int, ...]) -> Array:
        """Evaluate vector expansion derivatives at scattered points."""
        vals = []
        for i, space in enumerate(self.tensorspaces):
            ci = c[i]
            vi = space.evaluate_derivative(x, ci, k)
            vals.append(vi)
        return jnp.stack(vals)

    def forward(self, u: Array, N: tuple[int, ...] | None = None) -> Array:
        """Forward transform with optional truncation."""
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.forward(u[i], N)
            coeffs.append(ci)
        return jnp.stack(coeffs)


def TensorProduct(
    *basespaces: OrthogonalSpace | DirectSum,
    system: CoordSys | None = None,
    name: str = "T",
) -> TensorProductSpace:
    """Factory returning TensorProductSpace or DirectSumTPS.

    Handles:
      * Deep copy of bases to assign distinct coordinate subsystems
      * Name disambiguation for repeated space names
      * Propagation of subsystem coordinates into Composite / DirectSum

    If any axis is a DirectSum (inhomogeneous BC), returns DirectSumTPS.

    Args:
        *basespaces: 1D BaseSpace / DirectSum instances.
        system: Optional global coordinate system.
        name: Base name for the tensor product space(s).

    Returns:
        TensorProductSpace or DirectSumTPS.
    """
    from jaxfun.coordinates import CartCoordSys, x, y, z

    system = (
        CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[len(basespaces)])
        if system is None
        else system
    )

    basespaces_list: list[OrthogonalSpace | DirectSum] = [
        copy.deepcopy(space) for space in basespaces
    ]
    names = [space.name for space in basespaces_list]

    if jnp.any(jnp.array([name == names[0] for name in names[1:]])):
        for i, space in enumerate(basespaces_list):
            if isinstance(space, DirectSum):
                for spi in space.basespaces:
                    spi.name = spi.name + str(i)
            else:
                space.name = space.name + str(i)

    for i, space in enumerate(basespaces_list):
        # ty does not like this weird duck typing
        space.system = system.sub_system(i)  # ty:ignore[invalid-assignment]
        if isinstance(space, Composite):
            space.orthogonal.system = system.sub_system(i)  # ty:ignore[invalid-assignment]
        if isinstance(space, DirectSum):
            space.basespaces[0].system = system.sub_system(i)  # ty:ignore[invalid-assignment]
            if isinstance(space.basespaces[0], Composite):
                space.basespaces[0].orthogonal.system = system.sub_system(i)  # ty:ignore[invalid-assignment]
            space.basespaces[1].system = system.sub_system(i)  # ty:ignore[invalid-assignment]
            space.basespaces[1].orthogonal.system = system.sub_system(i)  # ty:ignore[invalid-assignment]

    if isinstance(basespaces_list[0], DirectSum) or isinstance(
        basespaces_list[1], DirectSum
    ):
        return DirectSumTPS(basespaces_list, system, name)

    def _all_orthogonal(
        spaces: list[OrthogonalSpace | DirectSum],
    ) -> TypeGuard[list[OrthogonalSpace]]:
        return all(isinstance(s, OrthogonalSpace) for s in spaces)

    assert _all_orthogonal(basespaces_list)
    return TensorProductSpace(basespaces_list, system, name)


class DirectSumTPS(TensorProductSpace):
    """Tensor product where one/both axes are DirectSum spaces (BC lifting).

    Builds a dictionary of homogeneous tensor-product subspaces produced
    by expanding DirectSum components. Also precomputes boundary lifting
    contributions needed to evaluate / transform functions with
    inhomogeneous boundary conditions in one or two dimensions.

    Attributes:
        tpspaces: Mapping from tuples of 1D spaces -> TensorProductSpace.
        bndvals: Dict storing boundary lifting coefficient arrays.
    """

    def __init__(
        self,
        basespaces: list[OrthogonalSpace | DirectSum],
        system: CoordSys,
        name: str = "DSTPS",
    ) -> None:
        from jaxfun.galerkin.inner import project1D

        self.basespaces: list[OrthogonalSpace | DirectSum] = basespaces
        self.system = system
        self.name = name
        self.bndvals: dict[tuple[OrthogonalSpace, ...], Array] = {}
        self.tensorname = tensor_product_symbol.join([b.name for b in basespaces])

        # Normalize symbolic BC expressions to base scalar form
        for space in basespaces:
            if space.bcs is None:
                continue
            if space.bcs.is_homogeneous():
                continue
            if isinstance(space, DirectSum):
                s0 = space.basespaces[1]
                for val in s0.bcs.values():
                    for key, v in val.items():
                        if len(sp.sympify(v).free_symbols) > 0:
                            val[key] = system.expr_psi_to_base_scalar(v)

        two_inhomogeneous = False
        bcall: list[list[BoundaryConditions]] = []
        if isinstance(basespaces[0], DirectSum) and isinstance(
            basespaces[1], DirectSum
        ):
            bcspaces = (basespaces[0].basespaces[1], basespaces[1].basespaces[1])
            two_inhomogeneous = bcspaces
            bc0, bc1 = bcspaces
            bc0bcs = copy.deepcopy(bc0.bcs)
            bc1bcs = copy.deepcopy(bc1.bcs)

            def lr(bcz: BCGeneric, z: str) -> sp.Number | float:
                return {"left": bcz.domain.lower, "right": bcz.domain.upper}[z]

            for bcthis, bcother, zother in zip(
                [bc0bcs, bc1bcs], [bc1bcs, bc0bcs], [bc1, bc0], strict=False
            ):
                assert isinstance(bcthis, BoundaryConditions)
                assert isinstance(bcother, BoundaryConditions)
                assert isinstance(zother, BCGeneric)

                bcall.append([])
                df = 2.0 / (zother.domain.upper - zother.domain.lower)
                for bcval in bcthis.orderedvals():
                    bcs: BoundaryConditions = copy.deepcopy(bcother)
                    for lr_other, bco in bcs.items():
                        z = lr(zother, lr_other)
                        for key in bco:
                            if key == "D":
                                expr0 = system.expr_base_scalar_to_psi(bcval)
                                assert isinstance(expr0, sp.Expr)
                                bco[key] = float(expr0.subs(zother.system._psi[0], z))
                            elif key[0] == "N":
                                var = zother.system._psi[0]
                                nd = 1 if len(key) == 1 else int(key[1])
                                expr1 = system.expr_base_scalar_to_psi(bcval)
                                assert isinstance(expr1, sp.Expr)
                                bco[key] = float(
                                    (expr1.diff(var, nd) / df**nd).subs(var, z)
                                )
                    bcall[-1].append(bcs)
            self.bndvals[bcspaces] = jnp.array([z.orderedvals() for z in bcall[0]])

        self.tpspaces: dict[tuple[OrthogonalSpace, ...], TensorProductSpace] = (
            self.split(basespaces)
        )

        # Precompute lifting coefficients
        for tensorspace in self.tpspaces:
            otherspaces: list[OrthogonalSpace] = [
                p for p in tensorspace if not isinstance(p, BCGeneric)
            ]
            bcspaces: list[BCGeneric] = [
                p for p in tensorspace if isinstance(p, BCGeneric)
            ]
            bcsindex: list[int] = [
                i for i, p in enumerate(tensorspace) if isinstance(p, BCGeneric)
            ]
            if len(otherspaces) == 0:
                continue
            elif len(otherspaces) == 1:
                assert len(bcspaces) == 1
                bcspace = bcspaces[0]
                otherspace = otherspaces[0]
                uh: list[Array] = []
                for j, bc in enumerate(bcspace.bcs.orderedvals()):
                    otherspace: OrthogonalSpace = otherspaces[0]
                    if two_inhomogeneous:
                        bco: BCGeneric = copy.deepcopy(
                            two_inhomogeneous[(bcsindex[0] + 1) % 2]
                        )
                        bco.bcs = bcall[bcsindex[0]][j]
                        otherspace: DirectSum = otherspace + bco
                    uh.append(project1D(bc, otherspace))
                if bcsindex[0] == 0:
                    self.bndvals[tensorspace] = jnp.array(uh)
                else:
                    self.bndvals[tensorspace] = jnp.array(uh).T

    def split(
        self, spaces: list[OrthogonalSpace | DirectSum]
    ) -> dict[tuple[OrthogonalSpace, ...], TensorProductSpace]:
        """Return dict of all homogeneous tensor combinations."""
        f: list[Iterable[OrthogonalSpace]] = []
        for space in spaces:
            if isinstance(space, DirectSum):
                f.append(space)
            else:
                f.append([space])
        tensorspaces = itertools.product(*f)
        return {
            s: TensorProductSpace(s, self.system, f"{self.name}{i}")
            for i, s in enumerate(tensorspaces)
        }

    def get_homogeneous(self):
        """Return tensor space built from homogeneous components only."""
        a0 = (
            self.basespaces[0].basespaces[0]
            if isinstance(self.basespaces[0], DirectSum)
            else self.basespaces[0]
        )
        a1 = (
            self.basespaces[1].basespaces[0]
            if isinstance(self.basespaces[1], DirectSum)
            else self.basespaces[1]
        )
        return self.tpspaces[(a0, a1)]

    def backward(
        self, c: Array, kind: str = "quadrature", N: tuple[int, ...] | None = None
    ) -> Array:
        """Evaluate total (homogeneous + lifting) backward transform."""
        a: list[Array] = []
        for f, v in self.tpspaces.items():
            a.append(v.backward(self.coeff(c, f, v), kind=kind, N=N))
        return jnp.sum(jnp.array(a), axis=0)

    def forward(self, c: Array) -> Array:
        """Solve projection for homogeneous coefficients (lifting removed)."""
        from jaxfun.galerkin import TestFunction, TrialFunction, inner
        from jaxfun.galerkin.arguments import JAXArray

        v = TestFunction(self)
        u = TrialFunction(self)
        c_sym = JAXArray(c, v.functionspace)
        A, b = inner((u - c_sym) * v)
        return jnp.linalg.solve(A[0].mat, b.flatten()).reshape(v.functionspace.num_dofs)

    def scalar_product(self, c: Array):
        """Disabled scalar product (non-homogeneous test space)."""
        raise RuntimeError(
            "Scalar product requires homogeneous test space (call on get_homogeneous())"
        )

    def evaluate(self, x: Array, c: Array, use_einsum: bool = False) -> Array:
        """Evaluate direct sum tensor product expansion at scattered points."""
        a: list[Array] = []
        for f, v in self.tpspaces.items():
            a.append(v.evaluate(x, self.coeff(c, f, v), use_einsum))
        return jnp.sum(jnp.array(a), axis=0)

    def evaluate_derivative(self, x: Array, c: Array, k: int) -> Array:
        """Evaluate direct sum tensor product expansion at scattered points."""
        a: list[Array] = []
        for f, v in self.tpspaces.items():
            a.append(v.evaluate_derivative(x, self.coeff(c, f, v), k))
        return jnp.sum(jnp.array(a), axis=0)

    def evaluate_mesh(
        self, x: list[Array], c: Array, use_einsum: bool = False
    ) -> Array:
        """Evaluate expansion on tensor mesh (summing lifting parts)."""
        a: list[Array] = []
        for f, v in self.tpspaces.items():
            a.append(v.evaluate_mesh(x, self.coeff(c, f, v), use_einsum))
        return jnp.sum(jnp.array(a), axis=0)

    def coeff(
        self,
        c: Array,
        f: tuple[OrthogonalSpace, ...],
        v: TensorProductSpace,
    ) -> Array:
        if jnp.any(jnp.array([isinstance(s, BCGeneric) for s in v.basespaces])):
            return self.bndvals[f]
        return c


class TPMatrices:
    """Container for list of TPMatrix bilinear operator tensors.

    Provides vectorized application (sum of individual tensor products)
    and a combined diagonal preconditioner (sum of per-matrix M(u)).
    """

    def __init__(self, tpmats: list[TPMatrix]):
        self.tpmats: list[TPMatrix] = tpmats

    @jax.jit(static_argnums=0)
    def _apply_array(self, u: Array) -> Array:
        return jnp.sum(jnp.array([mat._matmul_array(u) for mat in self.tpmats]), axis=0)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply summed tensor product operator to u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._apply_array(w)

    @jax.jit(static_argnums=0)
    def _precond_array(self, u: Array) -> Array:
        return jnp.sum(
            jnp.array([mat._precond_array(u) for mat in self.tpmats]), axis=0
        )

    def precond(self, u: Array | JAXFunction) -> Array:
        """Apply summed diagonal preconditioner to u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._precond_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return jnp.sum(
            jnp.array([mat._rmatmul_array(w) for mat in self.tpmats]), axis=0
        )


class precond:
    """Simple element-wise diagonal preconditioner wrapper."""

    def __init__(self, M):
        self.M = M

    @jax.jit(static_argnums=0)
    def __call__(self, u):
        """Return M * u (element-wise scaling)."""
        return self.M * u


class TPMatrix:  # noqa: B903
    """Rank-d separable tensor product operator A = kron(A0, A1, ...).

    Provides efficient matvec via successive multiplications instead of
    forming the full Kronecker product explicitly.

    Attributes:
        mats: List of per-axis sparse/dense matrices.
        scale: Scalar scaling (multiplicative).
        test_space, trial_space: TensorProductSpace descriptors.
        M: Preconditioner (diagonal inverse of kron diagonals).
    """

    def __init__(
        self,
        mats: Sequence[Array],
        scale: complex,
        test_space: TensorProductSpace | VectorTensorProductSpace,
        trial_space: TensorProductSpace | VectorTensorProductSpace,
        global_indices: tuple[int, int] = (0, 0),
    ) -> None:
        self.mats = list(mats)
        self.scale = scale
        self.test_space = test_space
        self.trial_space = trial_space
        self.global_indices = global_indices
        self.M = precond(
            (1.0 / self.mats[0].diagonal())[:, None]
            * (1.0 / self.mats[1].diagonal())[None, :]
        )

    @property
    def dims(self) -> int:
        """Return number of factor matrices."""
        return len(self.mats)

    @property
    def mat(self) -> Array:
        """Return explicit Kronecker product (dense)."""
        return jnp.kron(*self.mats)

    @jax.jit(static_argnums=0)
    def _matmul_array(self, w: Array) -> Array:
        return self.mats[0] @ w @ self.mats[1].T

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply matrix to rank-2 coefficient array u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._matmul_array(w)

    @jax.jit(static_argnums=0)
    def _precond_array(self, w: Array) -> Array:
        return self.M(w)

    def precond(self, u: Array | JAXFunction) -> Array:
        """Apply diagonal preconditioner to u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._precond_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    @jax.jit(static_argnums=0)
    def _rmatmul_array(self, w: Array) -> Array:
        return self.mats[0].T @ w @ self.mats[1]

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._rmatmul_array(w)


class TensorMatrix:  # noqa: B903
    """Non-separable tensor with dims * 2 indices.

    For test function v_{ij} and trial function u_{kl}, the tensor
    represents  A_{ikjl}.

    Matrix vector product

    .. math::

        v_{ij} = \\sum_{k,l} A_{ikjl} u_{kl}

    Stored when coefficient is non-separable in coordinates so Kron
    factorization is unavailable.

    Attributes:
        mat: Dense (or sparse) global matrix.
        test_space, trial_space: TensorProductSpace descriptors.
    """

    def __init__(
        self,
        mat: Array,
        test_space: TensorProductSpace,
        trial_space: TensorProductSpace,
    ) -> None:
        self.mat = mat  # mat is A_ikjl
        self.test_space = test_space
        self.trial_space = trial_space

    @jax.jit(static_argnums=0)
    def _matmul_array(self, w: Array) -> Array:
        return jnp.einsum("ikjl,kl->ij", self.mat, w)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply matrix to coefficient array u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._matmul_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    @jax.jit(static_argnums=0)
    def _rmatmul_array(self, w: Array) -> Array:
        return jnp.einsum("ij,ikjl->kl", w, self.mat)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._rmatmul_array(w)


def tpmats_to_scipy_sparse(
    A: list[TPMatrix], tol: int = 100
) -> list[tuple[scipy_sparse.csc_array, ...]]:
    """Convert list of separable TPMatrix to scipy CSC factors.

    Args:
        A: List of TPMatrix objects.
        tol: Near-zero elimination tolerance.

    Returns:
        List of tuples of per-axis scipy csc_array matrices.
    """
    return [
        tuple(
            scipy_sparse.csc_array(
                eliminate_near_zeros(mat, tol)
                if isinstance(mat, Array)
                else mat.todense()
            )
            for mat in a.mats
        )
        for a in A
    ]


def tpmats_to_scipy_kron(A: list[TPMatrix], tol: int = 100) -> scipy_sparse.csc_matrix:
    """Return summed global scipy sparse matrix (Kronecker expansion).

    Args:
        A: List of TPMatrix objects.
        tol: Near-zero elimination tolerance.

    Returns:
        scipy.sparse.csc_matrix representing Σ kron(factors).
    """
    a = tpmats_to_scipy_sparse(A)
    if len(a[0]) == 2:
        return np.sum([scipy_sparse.kron(b[0], b[1], format="csc") for b in a])
    else:
        return np.sum(
            [
                scipy_sparse.kron(
                    scipy_sparse.kron(b[0], b[1], format="csc"), b[2], format="csc"
                )
                for b in a
            ]
        )
