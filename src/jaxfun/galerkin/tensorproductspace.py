from __future__ import annotations

import copy
import itertools
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from typing import NoReturn, TypeGuard, cast

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxfun.coordinates import CoordSys
from jaxfun.typing import ArrayFun, MeshKind
from jaxfun.utils.common import jit_vmap, lambdify

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
        # Set to a NamedSharding before the first transform call to enable SPMD.
        # Unsharded axes will be processed first (local work, fully distributed),
        # sharded axes last (allgather deferred; output re-annotated with the same
        # sharding via jax.lax.with_sharding_constraint).  Can be changed after
        # the first JIT'd call: the public methods (backward, forward, …) are
        # plain wrappers that delegate to private *_jitted helpers which include
        # _spmd_sharding as an explicit static argument, so JAX recompiles
        # whenever the sharding is set or updated.
        # For DirectSumTPS, also set the sharding on .orthogonal separately.
        self._spmd_sharding: NamedSharding | None = None

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

    def _apply_separable(
        self,
        c: Array,
        fns: tuple[ArrayFun, ...],
        sharding: NamedSharding | None = None,
    ) -> Array:
        """Apply a per-axis 1D function to a tensor-product array.

        Called from inside already-JIT'd methods; not JIT'd itself.

        When ``sharding`` is set the work is split into two
        communication-free phases separated by a single all-to-all
        redistribution ("global transpose"):

        * **Phase 1 — unsharded axes**: each device holds a complete
          local slice along these axes, so the vmaps require no
          communication.
        * **Global transpose**: ``jax.lax.with_sharding_constraint``
          moves the sharding annotation from the originally-sharded axes
          to the formerly-unsharded axes (an all-to-all, cost
          ``O(N^d / P)`` per device).  After this, every device holds a
          complete slice along the originally-sharded axes.
        * **Phase 2 — originally-sharded axes**: the transforms are now
          also fully local.

        The output retains the transposed sharding
        ``P(None, "k")`` rather than the original ``P("k", None)``.
        This avoids a second all-to-all and keeps the result distributed.
        The caller may apply an explicit ``jax.lax.with_sharding_constraint``
        to change the layout if required.

        When ``sharding`` is ``None`` (the default, single-process
        case), axes are processed in natural order ``0, 1, …`` with no
        communication.

        The 3D case uses ``sorted()`` for the two remaining axes to
        guarantee deterministic vmap nesting.

        Args:
            c: Coefficient / value array of shape ``(N0, N1, ...)``.
            fns: Per-axis callables; ``fns[ax]`` accepts and returns a
                 1-D array.
            sharding: Optional SPMD sharding; pass ``self._spmd_sharding``.
                Must be supplied by the caller so that it is visible as a
                static argument to the enclosing JIT'd function, allowing
                JAX to recompile when the sharding is set or changed.

        Returns:
            Transformed array (sharding may differ from input; see above).
        """
        dim = len(self)

        if sharding is not None:
            spec = sharding.spec
            sharded = [
                ax for ax in range(dim) if ax < len(spec) and spec[ax] is not None
            ]
            unsharded = [ax for ax in range(dim) if ax not in sharded]
        else:
            sharded = []
            unsharded = list(range(dim))

        def _apply_axis(c: Array, ax: int) -> Array:
            if dim == 2:
                axi = dim - 1 - ax
                return jax.vmap(fns[ax], in_axes=axi, out_axes=axi)(c)
            else:
                ax0, ax1 = sorted(set(range(dim)) - {ax})
                return jax.vmap(
                    jax.vmap(fns[ax], in_axes=ax0, out_axes=ax0),
                    in_axes=ax1,
                    out_axes=ax1,
                )(c)

        # Phase 1: non-sharded axes — fully local, no communication.
        for ax in unsharded:
            c = _apply_axis(c, ax)

        # Global all-to-all transpose: move the sharding from the
        # originally-sharded axes to the formerly-unsharded axes.
        # After this every device holds a complete slice along the
        # originally-sharded axes, so Phase 2 is also communication-free.
        if sharding is not None and sharded:
            new_spec: list = [None] * dim
            for s_ax, u_ax in zip(sharded, unsharded):
                new_spec[u_ax] = spec[s_ax]
            c = jax.lax.with_sharding_constraint(
                c, NamedSharding(sharding.mesh, P(*new_spec))
            )

        # Phase 2: originally-sharded axes — local after the transpose.
        for ax in sharded:
            c = _apply_axis(c, ax)

        return c

    def _apply_separable_spmd(
        self,
        c: Array,
        fns: tuple[ArrayFun, ...],
        sharding: NamedSharding,
    ) -> Array:
        """SPMD version of ``_apply_separable`` that operates on addressable data.

        Called **outside** JIT so that ``c.addressable_data(0)`` returns the
        concrete local shard.  Each vmap then sees a plain (non-globally-sharded)
        array — the XLA SPMD partitioner is not involved and generates no
        all-gather operations.

        The only cross-device collective is the single ``jax.device_put`` call
        that performs the all-to-all redistribution between Phase 1 and Phase 2.
        """
        dim = len(self)
        spec = sharding.spec
        sharded = [ax for ax in range(dim) if ax < len(spec) and spec[ax] is not None]
        unsharded = [ax for ax in range(dim) if ax not in sharded]

        def _apply_axis(c_local: Array, ax: int) -> Array:
            if dim == 2:
                axi = dim - 1 - ax
                return jax.vmap(fns[ax], in_axes=axi, out_axes=axi)(c_local)
            else:
                ax0, ax1 = sorted(set(range(dim)) - {ax})
                return jax.vmap(
                    jax.vmap(fns[ax], in_axes=ax0, out_axes=ax0),
                    in_axes=ax1,
                    out_axes=ax1,
                )(c_local)

        # Phase 1 — unsharded axes: operate on the local addressable shard.
        # vmap here sees a plain array, so no all-gather is emitted.
        c_local = c.addressable_data(0)
        for ax in unsharded:
            c_local = _apply_axis(c_local, ax)

        # Reconstruct the global array from the updated local shard.
        # Unsharded axes may have changed size (e.g. Chebyshev zero-padding);
        # sharded axes retain their original global size.
        global_shape_p1 = tuple(
            c_local.shape[ax] if ax in unsharded else c.shape[ax] for ax in range(dim)
        )
        c = jax.make_array_from_single_device_arrays(
            global_shape_p1, sharding, [c_local]
        )

        # All-to-all: transpose the sharding (one collective, O(N^d/P) per device).
        new_spec: list = [None] * dim
        for s_ax, u_ax in zip(sharded, unsharded):
            new_spec[u_ax] = spec[s_ax]
        transposed = NamedSharding(sharding.mesh, P(*new_spec))
        c = jax.device_put(c, transposed)

        # Phase 2 — originally-sharded axes: now fully local after the transpose.
        c_local = c.addressable_data(0)
        for ax in sharded:
            c_local = _apply_axis(c_local, ax)

        # Reconstruct the final global array; sharded-axis sizes may have changed.
        global_shape_p2 = list(global_shape_p1)
        for ax in sharded:
            global_shape_p2[ax] = c_local.shape[ax]
        return jax.make_array_from_single_device_arrays(
            tuple(global_shape_p2), transposed, [c_local]
        )

    def backward(
        self,
        c: Array,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Backward transform.

        In the SPMD case the separable transform runs outside JIT on each
        device's addressable shard, eliminating spurious all-gather ops.
        """
        if self._spmd_sharding is not None:
            fns = tuple(
                partial(
                    self.basespaces[ax].backward,
                    kind=kind,
                    N=self.basespaces[ax].num_quad_points if N is None else N[ax],
                )
                for ax in range(len(self))
            )
            return self._apply_separable_spmd(c, fns, self._spmd_sharding)
        return self._backward_jitted(c, kind, N, None)

    @jax.jit(static_argnums=(0, 2, 3, 4))
    def _backward_jitted(
        self,
        c: Array,
        kind: MeshKind,
        N: tuple[int | None, ...] | None,
        sharding: NamedSharding | None,
    ) -> Array:
        fns = tuple(
            partial(
                self.basespaces[ax].backward,
                kind=kind,
                N=self.basespaces[ax].num_quad_points if N is None else N[ax],
            )
            for ax in range(len(self))
        )
        return self._apply_separable(c, fns, sharding)

    def _real_sharding(self) -> NamedSharding | None:
        """Return the real-space (transposed) sharding derived from ``_spmd_sharding``.

        ``_spmd_sharding`` describes the spectral layout, e.g. ``P("k", None, None)``
        (axis 0 sharded).  The real-space array produced by ``backward`` has the
        transposed layout ``P(None, "k", None)`` (axis 1 sharded).  This is the
        correct sharding to pass to ``_apply_separable`` for ``forward`` and
        ``scalar_product`` so that Phase 1 processes only fully-local axes.
        """
        sharding = self._spmd_sharding
        if sharding is None:
            return None
        dim = len(self)
        spec = sharding.spec
        sharded = [ax for ax in range(dim) if ax < len(spec) and spec[ax] is not None]
        unsharded = [ax for ax in range(dim) if ax not in sharded]
        new_spec: list = [None] * dim
        for s_ax, u_ax in zip(sharded, unsharded):
            new_spec[u_ax] = spec[s_ax]
        return NamedSharding(sharding.mesh, P(*new_spec))

    def scalar_product(self, u: Array) -> Array:
        """Return tensor of inner products along each axis (separable)."""
        if self._spmd_sharding is not None:
            sg = self.system.sg
            if sg != 1:
                sg = lambdify(self.system.base_scalars(), sg)(*self.mesh())
                u = u * sg
            fns = tuple(self.basespaces[ax].scalar_product for ax in range(len(self)))
            sharding = self._real_sharding()
            assert sharding is not None, "SPMD sharding must be set for scalar_product"
            return self._apply_separable_spmd(u, fns, sharding)
        return self._scalar_product_jitted(u, None)

    @jax.jit(static_argnums=(0, 2))
    def _scalar_product_jitted(self, u: Array, sharding: NamedSharding | None) -> Array:
        sg = self.system.sg
        if sg != 1:
            sg = lambdify(self.system.base_scalars(), sg)(*self.mesh())
            u = u * sg
        fns = tuple(self.basespaces[ax].scalar_product for ax in range(len(self)))
        return self._apply_separable(u, fns, sharding)

    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation."""
        if self._spmd_sharding is not None:
            fns = tuple(self.basespaces[ax].forward for ax in range(len(self)))
            sharding = self._real_sharding()
            assert sharding is not None, "SPMD sharding must be set for forward"
            return self._apply_separable_spmd(u, fns, sharding)
        return self._forward_jitted(u, None)

    @jax.jit(static_argnums=(0, 2))
    def _forward_jitted(self, u: Array, sharding: NamedSharding | None) -> Array:
        fns = tuple(self.basespaces[ax].forward for ax in range(len(self)))
        return self._apply_separable(u, fns, sharding)

    def backward_primitive(
        self,
        c: Array,
        k: tuple[int, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate the field or mixed derivatives on a tensor-product mesh."""
        if self._spmd_sharding is not None:
            fns = tuple(
                partial(
                    self.basespaces[ax].backward_primitive,
                    k=k[ax],
                    kind=kind,
                    N=self.basespaces[ax].num_quad_points if N is None else N[ax],
                )
                for ax in range(len(self))
            )
            return self._apply_separable_spmd(c, fns, self._spmd_sharding)
        return self._backward_primitive_jitted(c, k, kind, N, None)

    @jax.jit(static_argnums=(0, 2, 3, 4, 5))
    def _backward_primitive_jitted(
        self,
        c: Array,
        k: tuple[int, ...],
        kind: MeshKind | str,
        N: tuple[int | None, ...] | None,
        sharding: NamedSharding | None,
    ) -> Array:
        fns = tuple(
            partial(
                self.basespaces[ax].backward_primitive,
                k=k[ax],
                kind=kind,
                N=self.basespaces[ax].num_quad_points if N is None else N[ax],
            )
            for ax in range(len(self))
        )
        return self._apply_separable(c, fns, sharding)

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

    def scalar_product(self, c: Array) -> NoReturn:  # ty:ignore[invalid-method-override]
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
        k: tuple[int, ...] = (0,),
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
