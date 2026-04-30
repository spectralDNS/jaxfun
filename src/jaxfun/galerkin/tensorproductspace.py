from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from flax import nnx
from jax import Array
from scipy import sparse as scipy_sparse

from jaxfun.coordinates import CoordSys
from jaxfun.la import DiaMatrix, Matrix, MatrixProtocol, diakron
from jaxfun.typing import MeshKind

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
        return int(
            jnp.prod(
                jnp.array([space.dim for space in self.basespaces], dtype=int),
                dtype=int,
            )
        )

    @property
    def num_dofs(self) -> tuple[int, ...]:
        """Return tuple of active degrees of freedom per axis."""
        return tuple(space.num_dofs for space in self.basespaces)

    @property
    def num_quad_points(self) -> tuple[int, ...]:
        """Return tuple of quadrature points per axis."""
        return tuple(space.num_quad_points for space in self.basespaces)

    def mesh(
        self,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int, ...] | None = None,
        broadcast: bool = True,
    ) -> tuple[Array, ...]:
        """Return tensor mesh (as tuple of arrays) in true domain.

        Args:
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
            MeshKind.UNIFORM).
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
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int, ...] | None = None,
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
        self,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int, ...] | None = None,
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
                    )(jnp.atleast_1d(xi.squeeze()), c)
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
                    )(jnp.atleast_1d((xi).squeeze()), c)
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
        df = 1
        for i, Ti in enumerate(self.basespaces):
            Ci = Ti.evaluate_basis_derivative(
                Ti.map_reference_domain(x[i]).squeeze(), k[i]
            )
            c = jnp.tensordot(Ci, c, axes=(1, i), precision=jax.lax.Precision.HIGHEST)
            c = jnp.moveaxis(c, 0, i)
            df = df * (float(Ti.domain_factor ** k[i]))
        return c * df

    def get_orthogonal(self) -> TensorProductSpace:
        """Return underlying orthogonal basis instance."""
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return TensorProductSpace(
            orthogonal_spaces, system=self.system, name=self.name + "o"
        )

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(
        self,
        c: Array,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
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

    @jax.jit(static_argnums=0)
    def forward(self, u: Array) -> Array:
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

    @jax.jit(static_argnums=(0, 2, 3, 4))
    def backward_primitive(
        self,
        c: Array,
        k: tuple[int, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate the field or mixed derivatives on a tensor-product mesh."""
        dim: int = len(self)
        if dim == 2:
            for ax in range(dim):
                axi: int = dim - 1 - ax
                backward_p = partial(
                    self.basespaces[ax].backward_primitive,
                    k=k[ax],
                    kind=kind,
                    N=self.basespaces[ax].num_quad_points if N is None else N[ax],
                )
                c = jax.vmap(backward_p, in_axes=axi, out_axes=axi)(c)
        else:
            for ax in range(dim):
                backward_p = partial(
                    self.basespaces[ax].backward_primitive,
                    k=k[ax],
                    kind=kind,
                    N=self.basespaces[ax].num_quad_points if N is None else N[ax],
                )
                ax0, ax1 = set(range(dim)) - set((ax,))
                c = jax.vmap(
                    jax.vmap(backward_p, in_axes=ax0, out_axes=ax0),
                    in_axes=ax1,
                    out_axes=ax1,
                )(c)
        return c

    @jax.jit(static_argnums=0)
    def to_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped to underlying orthogonal basis."""
        S = [s.S.todense() for s in self.basespaces]
        dim = len(self)
        if dim == 2:
            return S[0].T @ c @ S[1]
        return jnp.einsum("is,jp,kl,ijk->spl", *S, c)

    @jax.jit(static_argnums=0)
    def from_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped from underlying orthogonal basis."""
        S = [s.get_inverse_stencil() for s in self.basespaces]
        dim = len(self)
        if dim == 2:
            return S[0].T @ c @ S[1]
        return jnp.einsum("is,jp,kl,ijk->spl", *S, c)


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
        self.num_quad_points = self.tensorspaces[0].num_quad_points

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

    @jax.jit(static_argnums=0)
    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation."""
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.forward(u[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    @jax.jit(static_argnums=0)
    def scalar_product(self, u: Array) -> Array:
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.scalar_product(u[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(
        self,
        u: Array,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Backward transform with optional padding."""
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.backward(u[i], kind=kind, N=N[i] if N is not None else None)
            coeffs.append(ci)
        return jnp.stack(coeffs)

    @jax.jit(static_argnums=(0, 2, 3, 4))
    def backward_primitive(
        self,
        u: Array,
        k: tuple[int, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.backward_primitive(
                u[i], k=k, kind=kind, N=N[i] if N is not None else None
            )
            coeffs.append(ci)
        return jnp.stack(coeffs)

    @jax.jit(static_argnums=0)
    def to_orthogonal(self, c: Array) -> Array:
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.to_orthogonal(c[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def get_orthogonal(self) -> VectorTensorProductSpace:
        orthogonal_spaces = [space.get_orthogonal() for space in self.tensorspaces]
        return VectorTensorProductSpace(tuple(orthogonal_spaces), name=self.name + "o")


def TensorProduct(
    *basespaces: OrthogonalSpace | DirectSum,
    system: CoordSys | None = None,
    name: str = "T",
) -> TensorProductSpace | DirectSumTPS:
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
                                bco[key] = float(
                                    sp.sympify(bcval).subs(
                                        zother.system.base_scalars()[0], z
                                    )
                                )
                            elif key[0] == "N":
                                nd = 1 if len(key) == 1 else int(key[1])
                                var = zother.system.base_scalars()[0]
                                bco[key] = float(
                                    (sp.sympify(bcval).diff(var, nd) / df**nd).subs(
                                        var, z
                                    )
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
                        otherspace: DirectSum = cast(Composite, otherspace) + bco
                    uh.append(project1D(bc, otherspace))
                if bcsindex[0] == 0:
                    self.bndvals[tensorspace] = jnp.array(uh)
                else:
                    self.bndvals[tensorspace] = jnp.array(uh).T
        self.orthogonal = self.get_orthogonal()

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

    def get_homogeneous(self) -> TensorProductSpace:
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

    @jax.jit(static_argnums=(0, 2, 3))
    def backward(
        self,
        c: Array,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: tuple[int, ...] | None = None,
    ) -> Array:
        """Evaluate total (homogeneous + lifting) backward transform."""
        return self.orthogonal.backward(self.to_orthogonal(c), kind=kind, N=N)

    @jax.jit(static_argnums=0)
    def forward(self, c: Array) -> Array:
        """Solve projection for homogeneous coefficients (lifting removed)."""
        d = self.orthogonal.forward(c)
        return self.from_orthogonal(d)

    def scalar_product(self, c: Array) -> NoReturn:
        """Disabled scalar product (non-homogeneous test space)."""
        raise RuntimeError(
            "Scalar product requires homogeneous test space (call on get_homogeneous())"
        )

    @jax.jit(static_argnums=(0, 3))
    def evaluate(self, x: Array, c: Array, use_einsum: bool = False) -> Array:
        """Evaluate direct sum tensor product expansion at scattered points."""
        return self.orthogonal.evaluate(x, self.to_orthogonal(c), use_einsum)

    @jax.jit(static_argnums=(0, 3))
    def evaluate_derivative(self, x: Array, c: Array, k: tuple[int, ...]) -> Array:
        """Evaluate direct sum tensor product expansion at scattered points."""
        return self.orthogonal.evaluate_derivative(x, self.to_orthogonal(c), k)

    @jax.jit(static_argnums=(0, 3))
    def evaluate_mesh(
        self, x: list[Array], c: Array, use_einsum: bool = False
    ) -> Array:
        """Evaluate expansion on tensor mesh (summing lifting parts)."""
        return self.orthogonal.evaluate_mesh(x, self.to_orthogonal(c), use_einsum)

    @jax.jit(static_argnums=(0, 2, 3, 4))
    def backward_primitive(
        self,
        c: Array,
        k: int | tuple[int, ...] = 0,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: tuple[int, ...] | None = None,
    ) -> Array:
        """Evaluate total (homogeneous + lifting) backward transform."""
        return self.orthogonal.backward_primitive(
            self.to_orthogonal(c), k=k, kind=kind, N=N
        )

    @jax.jit(static_argnums=0)
    def to_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped to underlying orthogonal basis."""
        a: list[Array] = []
        for f, v in self.tpspaces.items():
            a.append(v.to_orthogonal(self.bndvals.get(f, c)))
        z = [a[0]]
        for ai in a[1:]:
            z.append(
                jnp.pad(
                    ai,
                    [
                        (0, z[0].shape[0] - ai.shape[0]),
                        (0, z[0].shape[1] - ai.shape[1]),
                    ],
                )
            )
        return jnp.array(z).sum(axis=0)

    @jax.jit(static_argnums=0)
    def from_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped from underlying orthogonal basis."""
        a: list[Array] = []
        for f, v in self.tpspaces.items():
            if f in self.bndvals:
                a.append(-v.to_orthogonal(self.bndvals[f]))
            else:
                a.append(c)
        z = [a[0]]
        for ai in a[1:]:
            z.append(
                jnp.pad(
                    ai,
                    [
                        (0, z[0].shape[0] - ai.shape[0]),
                        (0, z[0].shape[1] - ai.shape[1]),
                    ],
                )
            )
        v = self.get_homogeneous()
        return v.from_orthogonal(jnp.array(z).sum(axis=0))


class TPMatrix(nnx.Pytree):  # noqa: B903
    """Rank-d separable tensor product operator A = kron(A0, A1, ...).

    Provides efficient matvec via successive multiplications instead of
    forming the full Kronecker product explicitly.

    Attributes:
        mats: List of per-axis sparse/dense matrices.
        scale: Scalar scaling (multiplicative).
        global_indices: Tuple of global index into vectorized expansions.
    """

    def __init__(
        self,
        mats: Sequence[MatrixProtocol],
        scale: complex,
        global_indices: tuple[int, int] = (0, 0),
    ) -> None:
        self.mats = nnx.List(mats)
        self.scale = scale
        self.global_indices = global_indices

    @property
    def dims(self) -> int:
        return len(self.mats)

    @property
    def mat(self) -> DiaMatrix | Matrix:
        """Return explicit Kronecker product.

        Returns a :class:`~jaxfun.la.DiaMatrix` when all factor matrices are
        DIA sparse; otherwise falls back to a dense :class:`jax.Array` via
        :func:`jnp.kron`.
        """
        if all(isinstance(m, DiaMatrix) for m in self.mats):
            result: DiaMatrix = self.mats[0]  # type: ignore[assignment]
            for m in self.mats[1:]:
                result = diakron(result, m)  # type: ignore[arg-type]
            return result
        arrays = [m.todense() for m in self.mats]
        out = arrays[0]
        for a in arrays[1:]:
            out = jnp.kron(out, a)
        return Matrix(out)

    def _matmul_array(self, w: Array) -> Array:
        result = w
        for i, mat in enumerate(self.mats):
            result = mat.matvec(result, axis=i)
        return result * jnp.asarray(self.scale)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply matrix to rank-2 coefficient array u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._matmul_array(w)

    def __matmul__(self, u: Array | JAXFunction) -> Array:
        """Alias to __call__ for @ operator."""
        return self.__call__(u)

    def _rmatmul_array(self, w: Array) -> Array:
        result = w
        for i, mat in enumerate(self.mats):
            result = mat.T.matvec(result, axis=i)
        return result * jnp.asarray(self.scale)

    def __rmatmul__(self, u: Array | JAXFunction) -> Array:
        """Right matmul (u @ A) treating u as left factor."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._rmatmul_array(w)

    def solve(self, rhs: Array) -> Array:
        """Solve ``(scale * A0 ⊗ A1 ⊗ …) x = rhs`` using Kronecker-factored LU.

        Exploits the mixed-product property

        .. math::

            (A_0 \\otimes A_1 \\otimes \\cdots)^{-1}
            = A_0^{-1} \\otimes A_1^{-1} \\otimes \\cdots

        to avoid forming the full Kronecker product.  Each factor's LU is
        computed once and cached on the factor matrix itself, so repeated
        ``solve`` calls pay only the substitution cost.

        Args:
            rhs: Right-hand side array.  May be flat ``(n,)`` or have the
                multidimensional shape ``(n0, n1, …)``.

        Returns:
            Solution array with the same shape as ``rhs``.
        """
        return self.lu_factor().solve(rhs)

    def lu_factor(self) -> TPLUFactors:
        """Pre-compute LU factors for every Kronecker factor.

        Returns a :class:`TPLUFactors` whose :meth:`~TPLUFactors.solve` method
        solves the Kronecker system without rebuilding the factorisation.
        """
        lu_factors = [mat.lu_factor() for mat in self.mats]
        shape = tuple(int(m.shape[0]) for m in self.mats)
        return TPLUFactors(lu_factors=lu_factors, scale=self.scale, shape=shape)


class TPLUFactors:
    """LU factorisation of a :class:`TPMatrix` (Kronecker product).

    Holds the per-factor LU objects and applies them sequentially on their
    respective axes to solve the full tensor-product system.

    Attributes:
        lu_factors: Per-axis LU factorisation objects (DiaMatrix or Matrix).
        scale: Scalar from the parent :class:`TPMatrix`.
        shape: Tuple of per-factor sizes ``(n0, n1, …)``.
    """

    def __init__(
        self, lu_factors: list, scale: complex, shape: tuple[int, ...]
    ) -> None:
        self.lu_factors = lu_factors
        self.scale = scale
        self.shape = shape

    def solve(self, rhs: Array) -> Array:
        """Solve ``(scale * A0 ⊗ A1 ⊗ …) x = rhs``.

        Args:
            rhs: Right-hand side.  Flat ``(n,)`` or shaped ``(n0, n1, …)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        y = rhs.reshape(self.shape)
        for i, lu in enumerate(self.lu_factors):
            y = lu.solve(y, axis=i)
        return (y / jnp.asarray(self.scale)).reshape(rhs.shape)


class TPMatrices(nnx.Pytree):
    """Container for list of TPMatrix bilinear operator tensors."""

    def __init__(self, tpmats: list[TPMatrix]) -> None:
        self.tpmats = nnx.List(tpmats)

    @jax.jit
    def _apply_array(self, u: Array) -> Array:
        return jnp.sum(jnp.array([mat._matmul_array(u) for mat in self.tpmats]), axis=0)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply summed tensor product operator to u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._apply_array(w)

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

    def lu_factor(self) -> TPMatricesLUFactors | TPMatricesWavenumberSolver:
        """Pre-compute factors for repeated fast solves.

        Tries :func:`tpmats_wavenumber_factor` first (efficient for Fourier ×
        polynomial problems), then falls back to :func:`tpmats_lu_factor`
        (diagonalization).

        Returns:
            :class:`TPMatricesWavenumberSolver` or :class:`TPMatricesLUFactors`
            for repeated fast solves.
        """
        cached: TPMatricesLUFactors | TPMatricesWavenumberSolver | None = getattr(
            self, "_lu_cache", None
        )
        if cached is not None:
            return cached
        try:
            result: TPMatricesLUFactors | TPMatricesWavenumberSolver = (
                tpmats_wavenumber_factor(list(self.tpmats))
            )
        except ValueError:
            result = tpmats_lu_factor(list(self.tpmats))
        object.__setattr__(self, "_lu_cache", result)
        return result

    def solve(self, rhs: Array) -> Array:
        """Solve the summed tensor-product system.

        Uses diagonalization (:meth:`lu_factor`) when all factor matrices on
        each axis are simultaneously diagonalizable (e.g. 2D/3D Poisson).
        Falls back to an explicit Kronecker product solve otherwise.
        """
        try:
            return self.lu_factor().solve(rhs)
        except ValueError:
            A = tpmats_to_kron(list(self.tpmats))
            return A.solve(rhs.flatten()).reshape(rhs.shape)


class TensorMatrix(nnx.Pytree):  # noqa: B903
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
    """

    def __init__(self, mat: Array) -> None:
        self.mat = mat  # mat is A_ikjl

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


class BlockTPMatrix:
    """Block matrix of TPMatrix objects.

    Attributes:
        blocks: list of TPMatrix objects.
        test_space, trial_space: VectorTensorProductSpace descriptors.
    """

    def __init__(
        self,
        tpmats: list[TPMatrix],
        test_space: VectorTensorProductSpace,
        trial_space: VectorTensorProductSpace,
    ) -> None:
        self.tpmats = tpmats
        self.test_space = test_space
        self.trial_space = trial_space
        self.shape = (self.test_space.dim, self.trial_space.dim)
        self.test_block_sizes = jnp.array(
            [self.test_space[i].dim for i in range(self.test_space.dims)]
        )
        self.trial_block_sizes = jnp.array(
            [self.trial_space[i].dim for i in range(self.trial_space.dims)]
        )

    @jax.jit(static_argnums=0)
    def _matmul_array(self, w: Array) -> Array:
        out = jnp.zeros_like(w)
        for mat in self.tpmats:
            indices = mat.global_indices
            out = out.at[indices[0]].add(mat @ w[indices[1]])
        return out

    @jax.jit(static_argnums=0)
    def mat(self) -> Array:
        """Return explicit block matrix (dense)."""
        out = jnp.zeros(self.shape)
        for m in self.tpmats:
            indices = m.global_indices
            out = out.at[self.slice(indices)].add(m.mat)
        return out

    def slice(self, indices: tuple[int, ...]) -> tuple[slice, ...]:  # ty:ignore[invalid-type-form]
        """Return slice object for block matrix indices."""
        N = self.test_block_sizes
        M = self.trial_block_sizes
        return (
            slice(jnp.sum(N[: indices[0]]), jnp.sum(N[: indices[0] + 1])),
            slice(jnp.sum(M[: indices[1]]), jnp.sum(M[: indices[1] + 1])),
        )

    def block_array(self) -> Matrix:
        """Return dense block matrix assembled from Kronecker products."""
        out = jnp.zeros(self.shape)
        for m in self.tpmats:
            indices = m.global_indices
            block = m.mat
            dense = block.todense()
            out = out.at[self.slice(indices)].add(dense)
        return Matrix(out)

    def __call__(self, u: Array | JAXFunction) -> Array:
        """Apply block matrix to coefficient array u."""
        from jaxfun.galerkin import JAXFunction

        w = u.array if isinstance(u, JAXFunction) else u
        return self._matmul_array(w)

    def solve(self, b: Array) -> Array:
        """Solve M x = b using a dense factorisation of the block matrix."""
        M = self.block_array()
        return M.solve(b.ravel()).reshape(b.shape)


class TPMatricesLUFactors:
    """Diagonalization-based solver for a sum of tensor-product operators.

    Solves

    .. math::

        \\sum_k s_k \\, (A_k^{(0)} \\otimes A_k^{(1)} \\otimes \\cdots)\\, x = f

    by simultaneously diagonalizing the factor matrices on each axis.

    Given a shared eigenbasis :math:`V` satisfying
    :math:`V^T A V = \\Lambda` (diagonal) and :math:`V^T B V = I`, the system
    reduces to element-wise division in the transformed space — :math:`O(n^d)`
    work after the :math:`O(n^3)` per-axis factorisation.

    For 2D Poisson (``K⊗M + M⊗K``): the denominator is
    :math:`D_{ij} = \\lambda_i + \\lambda_j` and the back-transform is
    :math:`U = V \\tilde{U} V^T`.
    """

    def __init__(
        self,
        eigvecs: list,
        per_term_eigenvalues: list,
        scales: list,
        shape: tuple[int, ...],
    ) -> None:
        self.eigvecs = eigvecs  # list of (n_i, n_i) eigenvector matrices
        self.per_term_eigenvalues = per_term_eigenvalues  # [term][axis] -> (n_axis,)
        self.scales = scales
        self.shape = shape

    @jax.jit(static_argnums=(0,))
    def solve(self, rhs: Array) -> Array:
        """Solve the summed tensor-product system for RHS ``rhs``.

        Args:
            rhs: Right-hand side, flat ``(n,)`` or shaped ``(n0, n1, ...)``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        shape = self.shape
        ndim = len(shape)
        F = rhs.reshape(shape)

        # Forward transform: apply V_i^T along each axis i.
        # jnp.tensordot(V.T, X, axes=[[1],[i]]) contracts V^T with axis i of X,
        # placing the result at position 0; moveaxis restores it to position i.
        Ftilde = F
        for i, V in enumerate(self.eigvecs):
            Ftilde = jnp.tensordot(V.T, Ftilde, axes=[[1], [i]])
            Ftilde = jnp.moveaxis(Ftilde, 0, i)

        # Denominator: D[i0,i1,...] = sum_k s_k * Λ_k[0][i0] * Λ_k[1][i1] * ...
        dtype = jnp.result_type(rhs.dtype, jnp.float32)
        D = jnp.zeros(shape, dtype=dtype)
        for evals_k, s_k in zip(self.per_term_eigenvalues, self.scales):
            term = jnp.ones(shape, dtype=dtype)
            for i, ev in enumerate(evals_k):
                idx: list = [None] * ndim
                idx[i] = slice(None)
                term = term * ev[tuple(idx)]
            D = D + jnp.asarray(s_k, dtype=dtype) * term

        # Solve in the transformed space (element-wise division).
        Utilde = Ftilde / D

        # Back-transform: apply V_i along each axis i.
        U = Utilde
        for i, V in enumerate(self.eigvecs):
            U = jnp.tensordot(V, U, axes=[[1], [i]])
            U = jnp.moveaxis(U, 0, i)

        return U.reshape(rhs.shape)


def _make_wavenumber_vmap_solve(
    L_offsets: tuple[int, ...],
    U_offsets: tuple[int, ...],
    n_P: int,
    dtype: Any,
) -> Callable[..., Array]:
    """Build a ``jax.vmap``-compiled batch solver for the wavenumber loop.

    Returns a function ``f(L_data_batch, U_data_batch, rhs_2d) -> sol_2d``
    that solves each 1-D banded system ``B_k x = b_k`` using forward and
    backward substitution compiled via :func:`jax.lax.scan`.  DIA offsets are
    captured as static Python values in the closure so :func:`jax.vmap` only
    traces over the array data — avoiding any pytree-metadata issues that
    would arise from constructing :class:`~jaxfun.la.DiaMatrix` instances
    with traced arrays.

    Args:
        L_offsets: Sub-diagonal offsets of the L factor (shared across all k).
        U_offsets: Super-diagonal offsets of the U factor (shared across all k).
        n_P: Length of each 1-D polynomial system.
        dtype: JAX dtype used for zero-padding of missing diagonals.

    Returns:
        A vmapped callable ``(L_data_batch, U_data_batch, rhs_2d) -> sol_2d``
        where each batch dimension corresponds to one Fourier wavenumber.
    """
    p = max((-o for o in L_offsets if o < 0), default=0)
    q = max((o for o in U_offsets if o > 0), default=0)

    # Index of each sub/super-diagonal in the data array, or None if absent.
    l_indices: list[int | None] = [
        L_offsets.index(-s) if -s in L_offsets else None for s in range(1, p + 1)
    ]
    U_main_idx: int = U_offsets.index(0)
    u_indices: list[int | None] = [
        U_offsets.index(s) if s in U_offsets else None for s in range(1, q + 1)
    ]

    # Reversal index for backward substitution — static since n_P is fixed.
    rev = jnp.arange(n_P - 1, -1, -1)

    def _fwd_elim(L_data: Array, b: Array) -> Array:
        """Solve L y = b (unit lower-triangular) via forward scan."""
        if p == 0:
            return b
        l_rows: list[Array] = []
        for s, idx in enumerate(l_indices, start=1):
            if idx is not None:
                d = L_data[idx]
                l_rows.append(
                    jnp.concatenate([jnp.zeros(s, dtype=d.dtype), d[: n_P - s]])
                )
            else:
                l_rows.append(jnp.zeros(n_P, dtype=dtype))
        l_mat = jnp.stack(l_rows)  # (p, n_P); l_mat[j, i] = L[i, i-(j+1)]

        def step(window: Array, xs: tuple) -> tuple[Array, Array]:
            bi, l_i = xs  # scalar, (p,)
            yi = bi - jnp.dot(l_i, window)
            return jnp.concatenate([yi[None], window[:-1]]), yi

        _, ys = jax.lax.scan(step, jnp.zeros(p, dtype=b.dtype), (b, l_mat.T))
        return ys

    def _bwd_sub(U_data: Array, y: Array) -> Array:
        """Solve U x = y (upper-triangular) via backward scan."""
        diag_d = U_data[U_main_idx]
        if q == 0:
            return y / diag_d
        u_rows: list[Array] = []
        for s, idx in enumerate(u_indices, start=1):
            if idx is not None:
                d = U_data[idx]
                u_rows.append(jnp.concatenate([d[s:n_P], jnp.zeros(s, dtype=d.dtype)]))
            else:
                u_rows.append(jnp.zeros(n_P, dtype=dtype))
        u_mat = jnp.stack(u_rows)  # (q, n_P)
        y_rev, diag_rev, u_mat_rev = y[rev], diag_d[rev], u_mat[:, rev].T  # (n_P, q)

        def step(window: Array, xs: tuple) -> tuple[Array, Array]:
            yi, u_i, dii = xs  # scalar, (q,), scalar
            xi = (yi - jnp.dot(u_i, window)) / dii
            return jnp.concatenate([xi[None], window[:-1]]), xi

        _, xs_out = jax.lax.scan(
            step, jnp.zeros(q, dtype=y.dtype), (y_rev, u_mat_rev, diag_rev)
        )
        return xs_out[rev]

    def _solve_one(L_data: Array, U_data: Array, b: Array) -> Array:
        return _bwd_sub(U_data, _fwd_elim(L_data, b))

    return jax.jit(jax.vmap(_solve_one))


class TPMatricesWavenumberSolver:
    """Per-wavenumber solver for Fourier × polynomial tensor-product systems.

    Solves

    .. math::

        \\sum_i s_i \\bigl(A_i^{(0)} \\otimes \\cdots\\bigr)\\, x = f

    where all axes except one are *Fourier* (every per-axis matrix is diagonal)
    and exactly one axis is *polynomial* (banded but not purely diagonal).

    For each combination of Fourier wavenumber indices the 1-D polynomial
    problem

    .. math::

        B_k\\, \\hat{u}_k = \\hat{f}_k, \\quad
        B_k = \\sum_i s_i \\Bigl(\\prod_{a \\in \\text{Fourier}}
        F_i^{(a)}[k_a]\\Bigr)\\, P_i

    is assembled using banded :class:`~jaxfun.la.DiaMatrix` arithmetic and
    pre-factorised with :meth:`~jaxfun.la.DiaMatrix.lu_factor` (result
    cached on each matrix).

    Args:
        poly_axis: Index of the polynomial axis in the full tensor.
        B_matrices: Per-wavenumber :class:`~jaxfun.la.DiaMatrix` objects,
            length ``n_F`` (product of all Fourier-axis sizes), each
            carrying a warm :meth:`~jaxfun.la.DiaMatrix.lu_factor` cache.
        shape: Full solution shape ``(n_0, n_1, ...)``.
    """

    def __init__(
        self,
        poly_axis: int,
        B_matrices: list,
        shape: tuple[int, ...],
    ) -> None:
        self.poly_axis = poly_axis
        self.B_matrices = B_matrices
        self.shape = shape
        # Pre-stack L and U data for vmapped solving.
        # lu_factor prunes structurally-zero diagonals per-wavenumber, so
        # different B_k may yield L/U with different offsets.  Compute the
        # union of offsets and pad missing diagonals with zeros so all rows
        # have the same shape before stacking.
        lu_list = [B.lu_factor() for B in B_matrices]
        n_P_local = B_matrices[0].shape[0]

        all_L_offsets: tuple[int, ...] = tuple(
            sorted({off for lu in lu_list for off in lu.L.offsets})
        )
        all_U_offsets: tuple[int, ...] = tuple(
            sorted({off for lu in lu_list for off in lu.U.offsets})
        )

        def _align_data(dia_mat: DiaMatrix, target_offsets: tuple[int, ...]) -> Array:
            rows: list[Array] = []
            for off in target_offsets:
                if off in dia_mat.offsets:
                    rows.append(dia_mat.data[list(dia_mat.offsets).index(off)])
                else:
                    rows.append(jnp.zeros(n_P_local, dtype=dia_mat.data.dtype))
            return jnp.stack(rows)

        self.L_data_batch: Array = jnp.stack(
            [_align_data(lu.L, all_L_offsets) for lu in lu_list]
        )
        self.U_data_batch: Array = jnp.stack(
            [_align_data(lu.U, all_U_offsets) for lu in lu_list]
        )
        self.L_offsets: tuple[int, ...] = all_L_offsets
        self.U_offsets: tuple[int, ...] = all_U_offsets
        self._vmap_solve = _make_wavenumber_vmap_solve(
            all_L_offsets, all_U_offsets, n_P_local, self.L_data_batch.dtype
        )

        # Rebuild LUFactors with aligned offsets (all_L_offsets / all_U_offsets)
        # so every element has the same pytree structure.  Then stack their
        # leaves into a single batched pytree and vmap LUFactors.solve over it.
        from jaxfun.la.diamatrix import LUFactors as _DiaLUFactors

        n = n_P_local
        aligned_lu_list = [
            _DiaLUFactors(
                L=DiaMatrix(
                    data=self.L_data_batch[k],
                    offsets=all_L_offsets,
                    shape=(n, n),
                ),
                U=DiaMatrix(
                    data=self.U_data_batch[k],
                    offsets=all_U_offsets,
                    shape=(n, n),
                ),
                shape=(n, n),
            )
            for k in range(len(lu_list))
        ]
        self._lu_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *aligned_lu_list)
        self._vmap_solve2 = jax.jit(jax.vmap(lambda lu, b: lu.solve(b)))

    @jax.jit(static_argnums=(0,))
    def solve(self, rhs: Array) -> Array:
        """Solve the wavenumber-loop system for RHS ``rhs``.

        All per-wavenumber 1-D banded polynomial solves are executed in a
        single :func:`jax.vmap` call over the stacked ``L`` / ``U`` factor
        data arrays.  The scan kernels are compiled once on the first call
        and reused for subsequent solves.

        Args:
            rhs: Right-hand side shaped ``self.shape``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        shape = self.shape
        ndim = len(shape)
        poly_axis = self.poly_axis
        n_P = shape[poly_axis]

        fourier_axes = [a for a in range(ndim) if a != poly_axis]
        fourier_shape = tuple(shape[a] for a in fourier_axes)
        n_F = int(np.prod(fourier_shape)) if fourier_shape else 1

        # Permute so all Fourier axes come first, polynomial axis last, then
        # flatten to (n_F, n_P) for vectorised solving.
        axes_order = fourier_axes + [poly_axis]
        rhs_2d = jnp.transpose(rhs, axes_order).reshape(n_F, n_P)

        sol_2d = self._vmap_solve(self.L_data_batch, self.U_data_batch, rhs_2d)

        # Un-permute back to the original axis order.
        sol_perm = sol_2d.reshape(fourier_shape + (n_P,))
        inv_perm = [0] * ndim
        for new_pos, old_pos in enumerate(axes_order):
            inv_perm[old_pos] = new_pos
        return jnp.transpose(sol_perm, inv_perm)

    @jax.jit(static_argnums=(0,))
    def solve2(self, rhs: Array) -> Array:
        """Alternative solve that vmaps :meth:`~jaxfun.la.diamatrix.LUFactors.solve`
        directly over a batched :class:`~jaxfun.la.diamatrix.LUFactors` pytree.

        All ``LUFactors`` for the wavenumber batch share the same aligned
        ``L``/``U`` offsets and ``shape``, so their leaves can be stacked once
        (in ``__init__``) and ``jax.vmap`` maps ``lu.solve(b)`` over the
        leading batch axis — no custom scan kernels required.

        Args:
            rhs: Right-hand side shaped ``self.shape``.

        Returns:
            Solution with the same shape as ``rhs``.
        """
        shape = self.shape
        ndim = len(shape)
        poly_axis = self.poly_axis
        n_P = shape[poly_axis]

        fourier_axes = [a for a in range(ndim) if a != poly_axis]
        fourier_shape = tuple(shape[a] for a in fourier_axes)
        n_F = int(np.prod(fourier_shape)) if fourier_shape else 1

        axes_order = fourier_axes + [poly_axis]
        rhs_2d = jnp.transpose(rhs, axes_order).reshape(n_F, n_P)

        sol_2d = self._vmap_solve2(self._lu_stacked, rhs_2d)

        sol_perm = sol_2d.reshape(fourier_shape + (n_P,))
        inv_perm = [0] * ndim
        for new_pos, old_pos in enumerate(axes_order):
            inv_perm[old_pos] = new_pos
        return jnp.transpose(sol_perm, inv_perm)


def tpmats_lu_factor(A: TPMatrix | list[TPMatrix]) -> TPMatricesLUFactors:
    """Compute diagonalization-based LU factors for a sum of :class:`TPMatrix`.

    Simultaneously diagonalizes the factor matrices on each axis so that the
    full Kronecker-sum system reduces to element-wise division in the
    transformed space.

    **Algorithm** (2D, generalises to any number of dims):

    Given a list of TPMatrices representing :math:`\\sum_k s_k A_k \\otimes B_k`,
    find :math:`V` such that :math:`V^T A V = \\Lambda_A` and
    :math:`V^T B V = I` (generalized eigenproblem :math:`A v = \\lambda B v`).
    Then:

    .. math::

        \\tilde{F} = V^T F V, \\quad
        D_{ij} = \\textstyle\\sum_k s_k \\lambda_k^{(0)}{}_i \\lambda_k^{(1)}{}_j,
        \\quad U = V (\\tilde{F} / D) V^T.

    **Requirement**: all factor matrices on each axis must be simultaneously
    diagonalizable — true whenever each axis has at most 2 distinct matrices
    that form a symmetric-definite pair (e.g. stiffness K and mass M from the
    same 1D function space).  Axes that share the same unordered matrix pair
    automatically reuse the same eigenvectors.

    Args:
        A: Single :class:`TPMatrix` or list of :class:`TPMatrix` objects (as
            returned by :func:`~jaxfun.galerkin.inner.inner`).

    Returns:
        :class:`TPMatricesLUFactors` whose :meth:`~TPMatricesLUFactors.solve`
        method solves the system without re-factorising.

    Raises:
        ValueError: if any axis has more than 2 distinct factor matrices.
    """
    if isinstance(A, TPMatrix):
        A = [A]
    tpmats = list(A)
    ndim = tpmats[0].dims

    # --- value-based deduplication of factor matrices ----------------------
    # Matrices that are numerically equal but have different Python ids (e.g.
    # M from K⊗M and M from the M⊗M term in a Helmholtz problem) are treated
    # as the same matrix.  All ids are mapped to a single representative id.
    _mat_by_id: dict[int, object] = {}
    _dense_by_id: dict[int, Array] = {}
    for tp in tpmats:
        for mat in tp.mats:
            mid = id(mat)
            if mid not in _mat_by_id:
                _mat_by_id[mid] = mat
                _dense_by_id[mid] = mat.todense()

    _seen_repr: list[int] = []  # canonical ids in first-seen order
    _id_to_repr: dict[int, int] = {}
    for mid in _mat_by_id:
        for rid in _seen_repr:
            if _dense_by_id[mid].shape == _dense_by_id[rid].shape and jnp.allclose(
                _dense_by_id[mid], _dense_by_id[rid], rtol=1e-5, atol=1e-8
            ):
                _id_to_repr[mid] = rid
                break
        else:
            _id_to_repr[mid] = mid
            _seen_repr.append(mid)

    def _repr(mat) -> int:
        return _id_to_repr[id(mat)]

    # --- per-axis pair → (eigvecs, {repr_id: eigenvalues}) ----------------
    # Axes that share the same unordered pair of matrices reuse eigenvectors.
    pair_cache: dict[frozenset, tuple[Array, dict[int, Array]]] = {}

    for i in range(ndim):
        mats_i = list(
            {_repr(tp.mats[i]): _mat_by_id[_repr(tp.mats[i])] for tp in tpmats}.values()
        )
        pair_key = frozenset(_repr(m) for m in mats_i)
        if pair_key in pair_cache:
            continue
        if len(mats_i) == 1:
            A_dense = cast(MatrixProtocol, mats_i[0]).todense()
            evals, evecs = jnp.linalg.eigh(A_dense)
            pair_cache[pair_key] = (evecs, {_repr(mats_i[0]): evals})
        elif len(mats_i) == 2:
            import numpy as _np
            import scipy.linalg as _scipy_linalg

            A0_np = _np.array(cast(MatrixProtocol, mats_i[0]).todense())
            A1_np = _np.array(cast(MatrixProtocol, mats_i[1]).todense())
            # Generalized eigenproblem: try A0 v = λ A1 v (A1 must be PD).
            # If that fails (A1 not PD), swap to A1 v = λ A0 v.
            try:
                evals_np, evecs_np = _scipy_linalg.eigh(A0_np, A1_np)
                evals = jnp.array(evals_np)
                evecs = jnp.array(evecs_np)
                pair_cache[pair_key] = (
                    evecs,
                    {
                        _repr(mats_i[0]): evals,
                        _repr(mats_i[1]): jnp.ones_like(evals),
                    },
                )
            except Exception:
                evals_np, evecs_np = _scipy_linalg.eigh(A1_np, A0_np)
                evals = jnp.array(evals_np)
                evecs = jnp.array(evecs_np)
                pair_cache[pair_key] = (
                    evecs,
                    {
                        _repr(mats_i[1]): evals,
                        _repr(mats_i[0]): jnp.ones_like(evals),
                    },
                )
        else:
            raise ValueError(
                f"Axis {i} has {len(mats_i)} distinct factor matrices; "
                "simultaneous diagonalization requires ≤ 2 distinct matrices per axis."
            )

    # Build per-axis eigenvector list and global repr_id→eigenvalues map.
    eigvecs: list[Array] = []
    axis_eigenvalues: dict[int, Array] = {}
    for i in range(ndim):
        mats_i = list(
            {_repr(tp.mats[i]): _mat_by_id[_repr(tp.mats[i])] for tp in tpmats}.values()
        )
        pair_key = frozenset(_repr(m) for m in mats_i)
        evecs, evals_map = pair_cache[pair_key]
        eigvecs.append(evecs)
        axis_eigenvalues.update(evals_map)

    per_term_eigenvalues = [
        [axis_eigenvalues[_repr(tp.mats[i])] for i in range(ndim)] for tp in tpmats
    ]
    scales = [tp.scale for tp in tpmats]
    shape = tuple(int(tpmats[0].mats[i].shape[0]) for i in range(ndim))
    return TPMatricesLUFactors(
        eigvecs=eigvecs,
        per_term_eigenvalues=per_term_eigenvalues,
        scales=scales,
        shape=shape,
    )


def tpmats_wavenumber_factor(
    A: list[TPMatrix] | TPMatrices,
) -> TPMatricesWavenumberSolver:
    """Pre-factorize a Fourier × polynomial :class:`TPMatrices` system.

    Detects which axes are Fourier (every term has a purely diagonal
    :class:`~jaxfun.la.DiaMatrix` — ``offsets == (0,)`` — on that axis) and
    which is the polynomial axis (banded but not purely diagonal).

    For each Fourier wavenumber index ``k`` assembles the 1-D banded
    polynomial system

    .. math::

        B_k = \\sum_i s_i \\Bigl(\\prod_{a \\in \\text{Fourier}}
        F_i^{(a)}[k_a]\\Bigr)\\, P_i

    as a :class:`~jaxfun.la.DiaMatrix` (preserving the banded sparsity
    pattern of the polynomial matrices) and warms its
    :meth:`~jaxfun.la.DiaMatrix.lu_factor` cache.

    Args:
        A: :class:`list` of :class:`TPMatrix` (as returned by
            :func:`~jaxfun.galerkin.inner.inner`) or a
            :class:`TPMatrices` instance.

    Returns:
        :class:`TPMatricesWavenumberSolver` for repeated fast solves.

    Raises:
        TypeError: If ``A`` is not a ``list[TPMatrix]`` or
            :class:`TPMatrices`.
        ValueError: If the structure does not have exactly one non-diagonal
            (polynomial) axis, e.g. for fully symmetric problems where
            :func:`tpmats_lu_factor` should be used instead.
    """
    if isinstance(A, TPMatrices):
        tpmats: list[TPMatrix] = list(A.tpmats)
    elif isinstance(A, list):
        tpmats = A
    else:
        raise TypeError(
            f"tpmats_wavenumber_factor expects a list[TPMatrix] or TPMatrices, "
            f"got {type(A).__name__!r}."
        )
    ndim: int = tpmats[0].dims

    def _is_diagonal_axis(axis: int) -> bool:
        return all(set(cast(DiaMatrix, tp.mats[axis]).offsets) == {0} for tp in tpmats)

    fourier_axes = [a for a in range(ndim) if _is_diagonal_axis(a)]
    poly_axes = [a for a in range(ndim) if not _is_diagonal_axis(a)]

    if len(poly_axes) != 1:
        raise ValueError(
            f"tpmats_wavenumber_factor requires exactly 1 polynomial "
            f"(non-diagonal) axis; found {len(poly_axes)}: {poly_axes}. "
            f"Use tpmats_lu_factor for fully-symmetric problems."
        )

    poly_axis = poly_axes[0]
    shape = tuple(int(tpmats[0].mats[a].shape[0]) for a in range(ndim))
    n_P = shape[poly_axis]

    # Build weight matrix W[i, k] = scale_i * prod_a(diag(F_i^(a))[k_a]).
    # The flat Fourier index k varies in C-order (last Fourier axis fastest),
    # matching the transposed layout used in TPMatricesWavenumberSolver.solve.
    W_list: list[Array] = []
    for tp in tpmats:
        w: Array = jnp.asarray(tp.scale, dtype=jnp.float32).reshape(1)
        for a in fourier_axes:
            # Diagonal DiaMatrix: data has shape (1, n_a); data[0] is the diagonal.
            diag_a = jnp.asarray(tp.mats[a].data[0], dtype=jnp.float32)  # (n_a,)
            w = jnp.outer(w, diag_a).flatten()  # 1 → n_{a0} → n_{a0}*n_{a1} → …
        W_list.append(w)  # (n_F,)

    W = jnp.stack(W_list)  # (n_terms, n_F)

    # Union of offsets across all polynomial matrices, in sorted order.
    poly_offsets: tuple[int, ...] = tuple(
        sorted(
            {
                int(off)
                for tp in tpmats
                for off in cast(DiaMatrix, tp.mats[poly_axis]).offsets
            }
        )
    )

    # Stack polynomial DIA data aligned to poly_offsets.
    # P_data_stack[i, d, :] = data of term i for offset poly_offsets[d].
    P_data_rows: list[Array] = []
    for tp in tpmats:
        mat = cast(DiaMatrix, tp.mats[poly_axis])
        rows: list[Array] = []
        for off in poly_offsets:
            if off in mat.offsets:
                idx = list(mat.offsets).index(off)
                rows.append(jnp.asarray(mat.data[idx], dtype=jnp.float32))
            else:
                rows.append(jnp.zeros(n_P, dtype=jnp.float32))
        P_data_rows.append(jnp.stack(rows))  # (n_diags, n_P)

    P_data_stack = jnp.stack(P_data_rows)  # (n_terms, n_diags, n_P)

    # Assemble per-wavenumber DIA data:
    # B_data_batch[k, d, :] = sum_i W[i,k] * P_data_stack[i, d, :].
    B_data_batch = jnp.einsum("tf,tdp->fdp", W, P_data_stack)  # (n_F, n_diags, n_P)
    n_F = B_data_batch.shape[0]

    # Build per-wavenumber DiaMatrix objects and warm their lu_factor caches.
    B_matrices: list[DiaMatrix] = []
    for k in range(n_F):
        B_k = DiaMatrix(
            data=B_data_batch[k],
            offsets=poly_offsets,
            shape=(n_P, n_P),
        )
        B_k.lu_factor()  # warms the cache stored on B_k
        B_matrices.append(B_k)

    return TPMatricesWavenumberSolver(
        poly_axis=poly_axis,
        B_matrices=B_matrices,
        shape=shape,
    )


def tpmats_to_kron(A: TPMatrix | list[TPMatrix], tol: int = 100) -> Matrix | DiaMatrix:
    """Return summed Kronecker expansion of a (list of) TPMatrix.

    Args:
        A: :class:`TPMatrix` or list of :class:`TPMatrix` objects with identical
            result shape.
        tol: Near-zero elimination tolerance applied to dense factor matrices
            before Kronecker expansion.

    Returns:
        :class:`~jaxfun.la.DiaMatrix` or :class:`~jaxfun.la.Matrix` representing
            the summed Kronecker expansion of the input TPMatrix objects.
    """
    if isinstance(A, TPMatrix):
        A = [A]

    if not A:
        raise ValueError("tpmats_to_kron requires a non-empty argument.")

    if isinstance(A[0].mats[0], Matrix):
        result: Array | None = None
        for tpm in A:
            a0 = eliminate_near_zeros(tpm.mats[0].todense(), tol)
            a0 = a0 * jnp.asarray(tpm.scale)
            for m in tpm.mats[1:]:
                a0 = jnp.kron(a0, eliminate_near_zeros(m.todense(), tol))
            result = a0 if result is None else result + a0
        assert result is not None
        return Matrix(result)

    def _get_dia(mat: MatrixProtocol) -> DiaMatrix:
        if isinstance(mat, Matrix):
            return DiaMatrix.from_dense(eliminate_near_zeros(mat, tol))
        assert isinstance(mat, DiaMatrix)
        return mat

    result: DiaMatrix | None = None
    for tpm in A:
        dmat: DiaMatrix = _get_dia(tpm.mats[0]) * jnp.asarray(tpm.scale)
        for m in tpm.mats[1:]:
            dmat = diakron(dmat, _get_dia(m))
        dmat = dmat
        result = dmat if result is None else result + dmat
    assert result is not None
    return result


@overload
def vec(A: Array, tol: int = 100) -> Array: ...
@overload
def vec(A: TPMatrix, tol: int = 100) -> Matrix | DiaMatrix: ...
@overload
def vec(A: list[TPMatrix], tol: int = 100) -> Matrix | DiaMatrix: ...
def vec(
    A: Array | TPMatrix | list[TPMatrix], tol: int = 100
) -> Array | Matrix | DiaMatrix:
    """Vectorize array or TPMatrix objects.

    Args:
        A: Dense :class:`jax.Array`, :class:`TPMatrix`, or list of :class:`TPMatrix`
            objects.
        tol: Near-zero elimination tolerance (only used for TPMatrix objects).

    Returns:
        Flattened :class:`jax.Array` or the summed Kronecker expansion as a
        :class:`~jaxfun.la.DiaMatrix`.
    """
    if not isinstance(A, Array):
        return tpmats_to_kron(A, tol=tol)

    return A.flatten()


def tpmats_to_scipy_sparse(
    A: list[TPMatrix], tol: int = 100
) -> list[tuple[scipy_sparse.csc_array, ...]]:
    """Convert list of separable TPMatrix to scipy CSC factors.

    The :attr:`~TPMatrix.scale` is folded into the first factor matrix.

    Args:
        A: List of TPMatrix objects.
        tol: Near-zero elimination tolerance.

    Returns:
        List of tuples of per-axis scipy csc_array matrices.
    """
    result = []
    for a in A:
        scale = a.scale
        factors = []
        for i, mat in enumerate(a.mats):
            dense = eliminate_near_zeros(mat.todense(), tol)
            if i == 0:
                dense = dense * scale
            factors.append(scipy_sparse.csc_array(dense))
        result.append(tuple(factors))
    return result


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
