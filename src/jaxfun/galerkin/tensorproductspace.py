from __future__ import annotations

import copy
import itertools
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from typing import NoReturn, cast

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array, shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxfun.coordinates import CartCoordSys, CoordSys
from jaxfun.sharding import (
    _apply_separable_spmd_shard_map,
    _build_local_apply_fn,
    physical_sharding,
    spectral_sharding,
    spmd_mesh,
)
from jaxfun.typing import MeshKind
from jaxfun.utils.common import jit_vmap, lambdify

from .composite import BCGeneric, BoundaryConditions, Composite, DirectSum
from .orthogonal import OrthogonalSpace

tensor_product_symbol = "\u2297"
multiplication_sign = "\u00d7"

IndivisibleError = ValueError


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
        self._spectral_sharding = spectral_sharding if len(jax.devices()) > 1 else None
        self._physical_sharding = physical_sharding if len(jax.devices()) > 1 else None
        self._spmd_local_fn_cache: dict = {}

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
        N: tuple[int | None, ...] | None = None,
        broadcast: bool = True,
    ) -> tuple[Array, ...]:
        """Return tensor mesh (as tuple of arrays) in true domain.

        Args:
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
            MeshKind.UNIFORM).
            N: Optional per-axis counts (defaults each to space.num_quad_points).
            broadcast: If True broadcast each axis array to nd-grid shape.

        Returns:
            Tuple (X0, X1, ...) each either 1D or broadcasted.
        """
        mesh = []
        N = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        for ax, space in enumerate(self.basespaces):
            X = space.mesh(kind, N[ax])
            mesh.append(self.broadcast_to_ndims(X, ax) if broadcast else X)
        return tuple(mesh)

    def flatmesh(
        self,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
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
        N: tuple[int | None, ...] | None = None,
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

    def evaluate_mesh(
        self,
        c: Array,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate expansion on tensor-product mesh.

        Args:
            c: Coefficient array
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
                MeshKind.UNIFORM).
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of evaluated field values with broadcast shape.
        """
        kind = MeshKind(kind)
        N = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        cache_key = ("evaluate_mesh", kind, N)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self),
                    ax,
                    partial(self.basespaces[ax].evaluate_mesh, kind=kind, N=N[ax]),
                )
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._spectral_sharding and len(c.devices()) > 1:
            return _apply_separable_spmd_shard_map(
                c, fns, spectral_sharding, self._spmd_local_fn_cache
            )
        for fn in fns:
            c = fn(c)
        return c

    @jit_vmap(in_axes=(0, None), static_argnums=(0,), ndim=1)
    def _evaluate_single_device(self, x: Array, c: Array) -> Array:
        """Evaluate expansion at scattered points — single-device path."""
        dim = len(self)
        T = self.basespaces
        C = [
            T[i].eval_basis_functions(T[i].map_reference_domain(x[i]))
            for i in range(dim)
        ]
        path = "i,j,ij" if dim == 2 else "i,j,k,ijk"
        return jnp.einsum(path, *C, c)

    def evaluate(self, x: Array, c: Array) -> Array:
        """Evaluate expansion at scattered points.

        Args:
            x: Stacked coordinate array, shape (n_pts, d).
            c: Coefficient tensor

        Returns:
            Scalar or (n_pts,) array of evaluated values.
        """
        if self._spectral_sharding and len(c.devices()) > 1:
            dim = len(self)
            T = self.basespaces

            C = [
                T[i].eval_basis_functions(T[i].map_reference_domain(x[:, i]))
                for i in range(dim)
            ]

            cache_key = ("evaluate_spmd",)
            if cache_key not in self._spmd_local_fn_cache:
                dc = "abcdef"[:dim]
                einsum_str = ",".join(f"j{ch}" for ch in dc) + f",{dc}->j"

                c_spec = spectral_sharding.spec
                p_spec = physical_sharding.spec

                def _local_eval(c_loc, C0_loc, *C_rest_loc):
                    return jax.lax.psum(
                        jnp.einsum(einsum_str, C0_loc, *C_rest_loc, c_loc), "k"
                    )

                def _jitted(c, C0, *C_rest):
                    C0_sharded = jax.device_put(C0, physical_sharding)
                    return shard_map(
                        _local_eval,
                        mesh=spectral_sharding.mesh,
                        in_specs=(c_spec, p_spec) + tuple(P() for _ in range(1, dim)),
                        out_specs=P(),
                        check_vma=False,
                    )(c, C0_sharded, *C_rest)

                self._spmd_local_fn_cache[cache_key] = jax.jit(_jitted)

            return self._spmd_local_fn_cache[cache_key](c, C[0], *C[1:])

        return self._evaluate_single_device(x, c)

    def get_orthogonal(self) -> TensorProductSpace:
        """Return underlying orthogonal basis instance."""
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return TensorProductSpace(
            orthogonal_spaces, system=self.system, name=self.name + "o"
        )

    def backward(
        self,
        c: Array,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Backward transform.

        Args:
            c: Coefficient array.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of backward transform values on quadrature mesh.
        """
        N = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        cache_key = ("backward", N)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self),
                    ax,
                    partial(self.basespaces[ax].backward, N=N[ax]),
                )
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._spectral_sharding and len(c.devices()) > 1:
            return _apply_separable_spmd_shard_map(
                c, fns, spectral_sharding, self._spmd_local_fn_cache
            )
        for fn in fns:
            c = fn(c)
        return c

    def scalar_product(self, u: Array) -> Array:
        """Return tensor of inner products along each axis (separable).

        Args:
            u: Input array.

        Returns:
            Array of inner products along each axis.
        """
        sg = self.system.sg
        if sg != 1:
            sg = lambdify(self.system.base_scalars(), sg)(*self.mesh())
            u = u * sg
        cache_key = ("scalar_product",)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(len(self), ax, self.basespaces[ax].scalar_product)
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._physical_sharding and len(u.devices()) > 1:
            return _apply_separable_spmd_shard_map(
                u, fns, physical_sharding, self._spmd_local_fn_cache
            )
        for fn in fns:
            u = fn(u)
        return u

    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation.

        Args:
            u: Input array.

        Returns:
            Array of forward transform values.
        """
        cache_key = ("forward",)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(len(self), ax, self.basespaces[ax].forward)
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._physical_sharding and len(u.devices()) > 1:
            return _apply_separable_spmd_shard_map(
                u, fns, physical_sharding, self._spmd_local_fn_cache
            )
        for fn in fns:
            u = fn(u)
        return u

    def backward_primitive(
        self,
        c: Array,
        k: tuple[int, ...],
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate the field or mixed derivatives on a tensor-product mesh.

        Args:
            c: Coefficient array.
            k: Tuple of derivative orders along each axis.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of backward primitive values on tensor-product mesh.
        """
        N = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        cache_key = ("backward_primitive", k, N)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self),
                    ax,
                    partial(
                        self.basespaces[ax].backward_primitive,
                        k=k[ax],
                        N=N[ax],
                    ),
                )
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._spectral_sharding and len(c.devices()) > 1:
            return _apply_separable_spmd_shard_map(
                c, fns, spectral_sharding, self._spmd_local_fn_cache
            )
        for fn in fns:
            c = fn(c)
        return c

    def to_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped to underlying orthogonal basis.

        Args:
            c: Coefficient array.

        Returns:
            Array of coefficients in the orthogonal basis.
        """
        dim = len(self)
        sharding = self._spectral_sharding
        if dim == 2:
            S = [s.S for s in self.basespaces]
            z = S[0].T @ c @ S[1]
        else:
            S = [s.S for s in self.basespaces]
            # z = jnp.einsum("is,jp,kl,ijk->spl", *S, c)
            z = c
            for i, Si in enumerate(S):
                z = Si.rmatvec(z, axis=i)

        if sharding:  # return sharded if possible, otherwise fallback to replicated
            try:
                return jax.device_put(z, sharding)
            except IndivisibleError:
                pass
        return z

    def from_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped from underlying orthogonal basis.

        Args:
            c: Coefficient array.

        Returns:
            Array of coefficients in the original basis.
        """
        sharding = self._spectral_sharding
        S = [s.get_inverse_stencil() for s in self.basespaces]
        dim = len(self)
        z = S[0].T @ c @ S[1] if dim == 2 else jnp.einsum("is,jp,kl,ijk->spl", *S, c)
        if sharding:  # return sharded if possible, otherwise fallback to replicated
            try:
                return jax.device_put(z, sharding)
            except IndivisibleError:
                pass
        return z


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
        self.num_quad_points = self.tensorspaces[0].num_quad_points
        # Slab decomposition for vector spaces
        # First index is vector component, which is not sharded.
        self._spectral_sharding: NamedSharding | None = (
            None if len(jax.devices()) == 1 else NamedSharding(spmd_mesh, P(None, "k"))
        )
        # Sharding of arrays in physical space.
        self._physical_sharding: NamedSharding | None = (
            None
            if len(jax.devices()) == 1
            else NamedSharding(spmd_mesh, P(None, None, "k"))
        )

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

    def evaluate(self, x: Array, c: Array) -> Array:
        """Evaluate vector expansion at scattered points.

        Args:
            x: Array of per-axis coordinates stacked (N, d).
            c: Coefficient array shaped (dims, N0, N1, ...).

        Returns:
            Evaluated values with shape determined by leading dims of x.
        """
        vals = []
        for i, space in enumerate(self.tensorspaces):
            ci = c[i]
            vi = space.evaluate(x, ci)
            vals.append(vi)
        return jnp.array(vals)

    def evaluate_mesh(
        self,
        u: Array,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Evaluate vector expansion on a mesh with optional padding.

        Args:
            u: Input array.
            kind: Type of mesh to evaluate on.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of evaluated values on the mesh.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.evaluate_mesh(u[i], kind=kind, N=N[i] if N is not None else None)
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation.

        Args:
            u: Input array.

        Returns:
            Array of forward transform values.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.forward(u[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def scalar_product(self, u: Array) -> Array:
        """Return tensor of inner products along each axis (separable).
        Args:
            u: Input array.

        Returns:
            Array of inner products along each axis.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.scalar_product(u[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def backward(
        self,
        u: Array,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Backward transform with optional padding.

        Args:
            u: Input array.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of backward transform values.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.backward(u[i], N=N[i] if N is not None else None)
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def backward_primitive(
        self,
        u: Array,
        k: tuple[int, ...],
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Backward primitive transform with optional padding.

        Args:
            u: Input array.
            k: Tuple of derivative orders.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of backward primitive transform values.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.backward_primitive(u[i], k=k, N=N[i] if N is not None else None)
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def to_orthogonal(self, c: Array) -> Array:
        """Convert coefficients to orthogonal basis.

        Args:
            c: Input array of coefficients.

        Returns:
            Array of coefficients in orthogonal basis.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.to_orthogonal(c[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def from_orthogonal(self, c: Array) -> Array:
        """Convert coefficients from orthogonal basis.

        Args:
            c: Input array of coefficients.

        Returns:
            Array of coefficients in the original basis.
        """
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.from_orthogonal(c[i])
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

    for i, space in enumerate(basespaces_list):
        space.system = system.sub_system(i)  # ty:ignore[invalid-assignment]
        if isinstance(space, Composite):
            space.orthogonal.system = space.system
        if isinstance(space, DirectSum):
            space.basespaces[0].system = space.system
            if isinstance(space.basespaces[0], Composite):
                space.basespaces[0].orthogonal.system = space.system
            space.basespaces[1].system = space.system
            space.basespaces[1].orthogonal.system = space.system

    if jnp.any(jnp.array([isinstance(s, DirectSum) for s in basespaces_list])):
        return DirectSumTPS(basespaces_list, system, name)

    assert all(isinstance(s, OrthogonalSpace | DirectSum) for s in basespaces_list)
    return TensorProductSpace(
        cast(list[OrthogonalSpace], basespaces_list), system, name
    )


class DirectSumTPS(TensorProductSpace):
    """Tensor product space where one or two basespaces are DirectSums.

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
        import numpy as np

        from jaxfun.galerkin.inner import project, project1D

        self.basespaces: list[OrthogonalSpace | DirectSum] = basespaces
        self.system = system
        self.name = name
        self.bndvals: dict[tuple[OrthogonalSpace, ...], Array] = {}
        self.tensorname = tensor_product_symbol.join([b.name for b in basespaces])
        self._spectral_sharding = spectral_sharding if len(jax.devices()) > 1 else None
        self._physical_sharding = physical_sharding if len(jax.devices()) > 1 else None

        # Normalize symbolic BC expressions to base scalar form
        bcindices = [
            i for i, space in enumerate(basespaces) if isinstance(space, DirectSum)
        ]
        if len(basespaces) == 3 and bcindices[0] == 0:
            raise ValueError(
                "DirectSum cannot be the first space in a 3D tensor product."
            )

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
        if len(bcindices) == 2:
            # If there are two DirectSums, we need to project to the other for each.
            # When projecting to the other space, we need to use the BC values
            # corresponding to the current space's BC values.
            bcspaces = (
                cast(DirectSum, basespaces[bcindices[0]]).basespaces[1],
                cast(DirectSum, basespaces[bcindices[1]]).basespaces[1],
            )
            two_inhomogeneous = bcspaces
            bc0, bc1 = bcspaces
            bc0bcs = copy.deepcopy(bc0.bcs)
            bc1bcs = copy.deepcopy(bc1.bcs)

            def lr(bcz: BCGeneric, z: str) -> float:
                return {
                    "left": float(bcz.domain.lower),
                    "right": float(bcz.domain.upper),
                }[z]

            for bcthis, bcother, zother in zip(
                [bc0bcs, bc1bcs], [bc1bcs, bc0bcs], [bc1, bc0], strict=False
            ):
                assert isinstance(bcthis, BoundaryConditions)
                assert isinstance(bcother, BoundaryConditions)
                assert isinstance(zother, BCGeneric)

                bcall.append([])
                df = 2.0 / (zother.domain.upper - zother.domain.lower)
                s = zother.system.base_scalars()[0]
                for bcval in bcthis.orderedvals():
                    bcs: BoundaryConditions = copy.deepcopy(bcother)
                    for lr_other, bco in bcs.items():
                        z = lr(zother, lr_other)
                        for key in bco:
                            if key == "D":
                                f = sp.sympify(bcval).subs(s, z)
                                if len(f.free_symbols) == 0:
                                    bco[key] = complex(f) if f.has(sp.I) else float(f)
                                else:
                                    bco[key] = f
                            elif key[0] == "N":
                                nd = 1 if len(key) == 1 else int(key[1])
                                f = (sp.sympify(bcval).diff(s, nd) / df**nd).subs(s, z)
                                if len(f.free_symbols) == 0:
                                    bco[key] = complex(f) if f.has(sp.I) else float(f)
                                else:
                                    bco[key] = f

                    bcall[-1].append(bcs)

            # Use np because orderedvals may contain sympy expressions.
            bvals = np.array([z.orderedvals() for z in bcall[0]])

            if len(basespaces) == 2:
                self.bndvals[bcspaces] = jnp.array(bvals)

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
            elif len(otherspaces) == 1 and len(bcspaces) == 1:
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

            elif len(otherspaces) == 2 and len(bcspaces) == 1:
                # find BCGeneric index. 1 or 2.
                isbc = [isinstance(space, BCGeneric) for space in tensorspace]
                bcind = isbc.index(True)
                ind_other = 1 if bcind == 2 else 2
                bcspace = bcspaces[0]
                uh: list[Array] = []
                for j, bc in enumerate(bcspace.bcs.orderedvals()):
                    otherbc = tensorspace[ind_other]
                    if two_inhomogeneous:
                        bco: BCGeneric = copy.deepcopy(
                            two_inhomogeneous[0 if bcind == 2 else 1]
                        )
                        bco.bcs = bcall[bcind - 1][j]
                        otherbc: DirectSum = (
                            cast(Composite, tensorspace[ind_other]) + bco
                        )

                    newspaces = [
                        copy.deepcopy(space) for space in [otherspaces[0], otherbc]
                    ]
                    othertpspace = TensorProduct(
                        *newspaces,
                        system=CartCoordSys(
                            "T",
                            (
                                newspaces[0].system.base_scalars()[0],
                                newspaces[1].system.base_scalars()[0],
                            ),
                        ),
                    )
                    uh.append(project(bc, othertpspace))

                if bcind == 2:
                    self.bndvals[tensorspace] = jnp.array(uh).transpose(1, 2, 0)
                else:
                    self.bndvals[tensorspace] = jnp.array(uh).transpose(1, 0, 2)

            elif len(otherspaces) == 1 and len(bcspaces) == 2:
                uh: list[Array] = []
                for bci in bcall[0]:
                    for bc0 in bci.orderedvals():
                        uh.append(project(bc0, otherspaces[0]))
                self.bndvals[tensorspace] = jnp.array(uh).T.reshape(
                    (-1, len(bcall[0]), len(bcall[1]))
                )

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
        ai = [
            space[0] if isinstance(space, DirectSum) else space
            for space in self.basespaces
        ]
        return self.tpspaces[tuple(ai)]

    def backward(
        self,
        c: Array,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        return self.orthogonal.backward(self.to_orthogonal(c), N=N)

    def forward(self, u: Array) -> Array:
        d = self.orthogonal.forward(u)
        return self.from_orthogonal(d)

    def scalar_product(self, c: Array) -> NoReturn:  # ty:ignore[invalid-method-override]
        raise RuntimeError(
            "Scalar product requires homogeneous test space (call on get_homogeneous())"
        )

    def evaluate(self, x: Array, c: Array) -> Array:
        return self.orthogonal.evaluate(x, self.to_orthogonal(c))

    def evaluate_mesh(
        self,
        c: Array,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        return self.orthogonal.evaluate_mesh(self.to_orthogonal(c), kind=kind, N=N)

    def backward_primitive(
        self,
        c: Array,
        k: tuple[int, ...],
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        return self.orthogonal.backward_primitive(self.to_orthogonal(c), k=k, N=N)

    def to_orthogonal(self, c: Array) -> Array:
        result = self.get_homogeneous().to_orthogonal(c)

        for f, v in self.tpspaces.items():
            inp = self.bndvals.get(f, c)
            if inp is c:
                continue
            ai = v.to_orthogonal(inp)  # sharded if possible
            result = result + jnp.pad(
                ai,
                [(0, result.shape[i] - ai.shape[i]) for i in range(c.ndim)],
            )

        return result

    def from_orthogonal(self, c: Array) -> Array:
        # Note that c may be replicated, because the orthogonal space is not the
        # same as the original space, so we can't assume the sharding is compatible.

        result: Array = jnp.zeros(1)

        for f, v in self.tpspaces.items():
            inp = self.bndvals.get(f, c)
            if inp is c:
                continue
            ai = -v.to_orthogonal(inp)  # sharded if possible
            result = result + jnp.pad(
                ai,
                [(0, c.shape[i] - ai.shape[i]) for i in range(c.ndim)],
            )
        # ensure replicated result is on same sharding as c
        try:
            result = c + jax.device_put(result, c.sharding)
        except IndivisibleError:
            result = c + result
        return self.get_homogeneous().from_orthogonal(result)
