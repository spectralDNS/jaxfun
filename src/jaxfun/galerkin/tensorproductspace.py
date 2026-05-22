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
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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
        # Slab decomposition
        self._spmd_mesh = Mesh(jax.devices(), ("k",))
        # Sharding of arrays in spectral coefficient space.
        self._spectral_sharding: NamedSharding | None = (
            None
            if len(jax.devices()) == 1
            else NamedSharding(self._spmd_mesh, P("k", None))
        )
        # Sharding of arrays in physical space.
        self._physical_sharding: NamedSharding | None = (
            None
            if len(jax.devices()) == 1
            else NamedSharding(self._spmd_mesh, P(None, "k"))
        )
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
    def _evaluate_single_device(self, x: Array, c: Array, use_einsum: bool) -> Array:
        """Evaluate expansion at scattered points — single-device path."""
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

    def evaluate(self, x: Array, c: Array, use_einsum: bool = False) -> Array:
        """Evaluate expansion at scattered points.

        For SPMD (multi-device): c carries spectral sharding P("k", None).
        The per-axis evaluation matrices C[i] are computed once on the calling
        process (fully replicated, since x may be as small as one point).
        Each local addressable shard of c contributes a partial contraction
        which is summed locally and then all-reduced across MPI processes so
        that every process holds the same replicated result.

        For single-device: x is a stacked (n_pts, d) array evaluated via the
        jit-vmapped single-device path.

        Args:
            x: Stacked coordinate array, shape (n_pts, d).
            c: Coefficient tensor; expected to carry spectral sharding in the
               SPMD case.
            use_einsum: Passed through to the single-device path (unused in
               the SPMD path, which is using einsum).

        Returns:
            Scalar or (n_pts,) array of evaluated values; replicated in SPMD.
        """
        if self._spectral_sharding is not None and len(c.devices()) > 1:
            dim = len(self)
            T = self.basespaces

            # Per-axis evaluation matrices — fully replicated.
            # eval_basis_functions is @jit_vmap'd: 1-D input (n_pts,) → (n_pts, N).
            # x has shape (n_pts, dim); x[:, i] gives axis-i coordinates.
            C = [
                T[i].eval_basis_functions(T[i].map_reference_domain(x[:, i]))
                for i in range(dim)
            ]

            # Build einsum string for arbitrary dim:
            #   C[0]:"ja", C[1]:"jb", ..., c_loc:"ab..." → "j"
            # Contracts all spectral axes; j is the evaluation-point axis.
            dc = "abcdef"[:dim]
            einsum_str = ",".join(f"j{ch}" for ch in dc) + f",{dc}->j"

            # Shard C[0] along axis 1 (N_0 dim) to match c's axis-0 sharding.
            # C[1], ..., C[dim-1] are replicated so every device sees them in full.
            mesh = self._spectral_sharding.mesh
            C0_sharded = jax.device_put(C[0], NamedSharding(mesh, P(None, "k")))

            c_spec = self._spectral_sharding.spec

            def _local_eval(c_loc, C0_loc, *C_rest_loc):
                return jax.lax.psum(
                    jnp.einsum(einsum_str, C0_loc, *C_rest_loc, c_loc), "k"
                )

            return shard_map(
                _local_eval,
                mesh=mesh,
                in_specs=(c_spec, P(None, "k")) + tuple(P() for _ in range(1, dim)),
                out_specs=P(),
                check_rep=False,
            )(c, C0_sharded, *C[1:])
        return self._evaluate_single_device(x, c, use_einsum)

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

    def _build_local_apply_fn(self, ax: int, fn: ArrayFun) -> ArrayFun:
        """Return a ``jax.jit(jax.vmap(...))`` that applies *fn* along *ax*.

        The resulting callable operates on a plain (non-sharded) local array,
        so JAX compiles it once and reuses the compiled binary on every call.
        """
        dim = len(self)
        if dim == 2:
            axi = dim - 1 - ax
            return jax.jit(jax.vmap(fn, in_axes=axi, out_axes=axi))
        ax0, ax1 = sorted(set(range(dim)) - {ax})
        return jax.jit(
            jax.vmap(
                jax.vmap(fn, in_axes=ax0, out_axes=ax0),
                in_axes=ax1,
                out_axes=ax1,
            )
        )

    def _apply_separable_spmd(
        self,
        c: Array,
        fns: tuple[ArrayFun, ...],
        sharding: NamedSharding,
    ) -> Array:
        """Apply separable per-axis transforms on distributed (SPMD) arrays.

        The transform is split into two fully-local phases separated by a
        single all-to-all redistribution:

        * **Phase 1 — unsharded axes**: each device holds the complete extent
          along these axes, so no communication is needed.
        * **All-to-all**: one ``jax.device_put`` transposes the sharding from
          the originally-sharded axes to the formerly-unsharded axes.
        * **Phase 2 — originally-sharded axes**: now fully local after the
          transpose.

        Note:
        * The input sharding must be either spectral or physical, depending on
          the transform being applied.
        * The provided fns must be in the same order as basespaces and match the
            sharding (e.g. spectral fns applied with spectral sharding).
        * When input sharding is spectral, the output is physical and vice versa.

        """
        dim = len(self)
        spec = sharding.spec
        sharded = [ax for ax in range(dim) if ax < len(spec) and spec[ax] is not None]
        unsharded = [ax for ax in range(dim) if ax not in sharded]
        n_local = jax.local_device_count()

        # Phase 1 — unsharded axes: operate on each local addressable shard.
        # fns[ax] is a pre-jitted vmap; XLA cache is hit on every call.
        local_shards = [c.addressable_data(d) for d in range(n_local)]
        for ax in unsharded:
            local_shards = [fns[ax](shard) for shard in local_shards]

        # Reconstruct the global array from the updated local shards.
        # Unsharded axes may have changed size (e.g. Chebyshev with BCs);
        # sharded axes retain their original global size.
        global_shape_p1 = tuple(
            local_shards[0].shape[ax] if ax in unsharded else c.shape[ax]
            for ax in range(dim)
        )
        c = jax.make_array_from_single_device_arrays(
            global_shape_p1, sharding, local_shards
        )

        # All-to-all: transpose the sharding (one collective, O(N^d/P) per device).
        new_spec: list = [None] * dim
        for s_ax, u_ax in zip(sharded, unsharded):
            new_spec[u_ax] = spec[s_ax]
        transposed = NamedSharding(sharding.mesh, P(*new_spec))
        c = jax.device_put(c, transposed)

        # Phase 2 — originally-sharded axes: now fully local after the transpose.
        local_shards = [c.addressable_data(d) for d in range(n_local)]
        for ax in sharded:
            local_shards = [fns[ax](shard) for shard in local_shards]

        # Reconstruct the final global array; sharded-axis sizes may have changed.
        global_shape_p2 = list(global_shape_p1)
        for ax in sharded:
            global_shape_p2[ax] = local_shards[0].shape[ax]
        return jax.make_array_from_single_device_arrays(
            tuple(global_shape_p2), transposed, local_shards
        )

    def _apply_separable_spmd_shard_map(
        self,
        c: Array,
        fns: tuple[ArrayFun, ...],
        sharding: NamedSharding,
    ) -> Array:
        """Apply separable per-axis transforms using ``shard_map`` + ``lax.all_to_all``.

        JAX-native alternative to :meth:`_apply_separable_spmd`.  The entire
        transform — including the inter-device redistribution — is a single
        compiled XLA computation, allowing XLA to fuse across phase boundaries.

        The algorithm mirrors the three-phase structure of the addressable-data
        approach:

        * **Phase 1**: unsharded-axis transforms applied locally inside the kernel.
        * **All-to-all**: ``lax.all_to_all(tiled=True)`` transposes the sharding.
        * **Phase 2**: originally-sharded-axis transforms applied locally.

        .. note::
            ``lax.all_to_all(tiled=True)`` requires the ``split_axis`` dimension
            (the first unsharded axis, after Phase 1) to be divisible by the
            total number of devices.  This holds for typical spectral sizes
            (powers of two for Fourier, even quadrature counts for Chebyshev).

        """
        # Cache the compiled shard_map function keyed on the (fns, sharding spec)
        # combination.  _kernel is defined inside the method, so each call would
        # produce a new function object and force recompilation.  Storing the
        # shard_map-wrapped callable ensures it is compiled exactly once.
        cache_key = ("shard_map_kernel", id(fns), sharding.spec)
        if cache_key not in self._spmd_local_fn_cache:
            dim = len(self)
            spec = sharding.spec
            sharded = [
                ax for ax in range(dim) if ax < len(spec) and spec[ax] is not None
            ]
            unsharded = [ax for ax in range(dim) if ax not in sharded]

            new_spec: list = [None] * dim
            for s_ax, u_ax in zip(sharded, unsharded):
                new_spec[u_ax] = spec[s_ax]

            def _kernel(c_loc: Array) -> Array:
                # Phase 1 — unsharded axes: fully local, no communication.
                for ax in unsharded:
                    c_loc = fns[ax](c_loc)
                # All-to-all: redistribute sharding from sharded → unsharded axes.
                c_loc = jax.lax.all_to_all(
                    c_loc,
                    axis_name="k",
                    split_axis=unsharded[0],
                    concat_axis=sharded[0],
                    tiled=True,
                )
                # Phase 2 — originally-sharded axes: fully local after the transpose.
                for ax in sharded:
                    c_loc = fns[ax](c_loc)
                return c_loc

            self._spmd_local_fn_cache[cache_key] = shard_map(
                _kernel,
                mesh=sharding.mesh,
                in_specs=(sharding.spec,),
                out_specs=P(*new_spec),
                check_rep=False,
            )

        return self._spmd_local_fn_cache[cache_key](c)

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
        N_resolved = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        cache_key = ("backward", kind, N_resolved)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                self._build_local_apply_fn(
                    ax,
                    partial(self.basespaces[ax].backward, kind=kind, N=N_resolved[ax]),
                )
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._spectral_sharding is not None and len(c.devices()) > 1:
            return self._apply_separable_spmd(c, fns, self._spectral_sharding)
        for fn in fns:
            c = fn(c)
        return c

    def scalar_product(self, u: Array) -> Array:
        """Return tensor of inner products along each axis (separable)."""
        sg = self.system.sg
        if sg != 1:
            sg = lambdify(self.system.base_scalars(), sg)(*self.mesh())
            u = u * sg
        cache_key = ("scalar_product",)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                self._build_local_apply_fn(ax, self.basespaces[ax].scalar_product)
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._physical_sharding is not None and len(u.devices()) > 1:
            return self._apply_separable_spmd(u, fns, self._physical_sharding)
        for fn in fns:
            u = fn(u)
        return u

    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation."""
        cache_key = ("forward",)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                self._build_local_apply_fn(ax, self.basespaces[ax].forward)
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._physical_sharding is not None and len(u.devices()) > 1:
            return self._apply_separable_spmd(u, fns, self._physical_sharding)
        for fn in fns:
            u = fn(u)
        return u

    def backward_primitive(
        self,
        c: Array,
        k: tuple[int, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate the field or mixed derivatives on a tensor-product mesh."""
        N_resolved = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        cache_key = ("backward_primitive", k, kind, N_resolved)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                self._build_local_apply_fn(
                    ax,
                    partial(
                        self.basespaces[ax].backward_primitive,
                        k=k[ax],
                        kind=kind,
                        N=N_resolved[ax],
                    ),
                )
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._spectral_sharding is not None and len(c.devices()) > 1:
            return self._apply_separable_spmd(c, fns, self._spectral_sharding)
        for fn in fns:
            c = fn(c)
        return c

    def to_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped to underlying orthogonal basis."""
        dim = len(self)
        if dim == 2:
            if self._spectral_sharding is not None and len(c.devices()) > 1:
                S = [s.S for s in self.basespaces]
                cache_key = "to_orthogonal"
                if cache_key not in self._spmd_local_fn_cache:
                    self._spmd_local_fn_cache[cache_key] = tuple(
                        self._build_local_apply_fn(ax, S[ax].rmatvec)
                        for ax in range(dim)
                    )
                fns = self._spmd_local_fn_cache[cache_key]
                result = self._apply_separable_spmd(c, fns, self._spectral_sharding)
                return jax.device_put(result, self._spectral_sharding)
            S = [s.S for s in self.basespaces]
            return S[0].T @ c @ S[1]

        if self._spectral_sharding is not None:
            raise NotImplementedError(
                "to_orthogonal with SPMD sharding is not implemented yet for dim > 2"
            )
        S = [s.S.todense() for s in self.basespaces]
        return jnp.einsum("is,jp,kl,ijk->spl", *S, c)

    def from_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped from underlying orthogonal basis."""
        S = [s.get_inverse_stencil() for s in self.basespaces]
        dim = len(self)
        if dim == 2:
            if self._physical_sharding is not None and len(c.devices()) > 1:
                cache_key = "from_orthogonal"
                if cache_key not in self._spmd_local_fn_cache:
                    self._spmd_local_fn_cache[cache_key] = tuple(
                        self._build_local_apply_fn(
                            ax, lambda x, _S=S[ax]: jnp.dot(x, _S)
                        )
                        for ax in range(dim)
                    )
                fns = self._spmd_local_fn_cache[cache_key]
                c = jax.device_put(c, self._physical_sharding)
                return self._apply_separable_spmd(c, fns, self._physical_sharding)
            return S[0].T @ c @ S[1]
        if self._physical_sharding is not None:
            raise NotImplementedError(
                "from_orthogonal with SPMD sharding is not implemented yet for dim > 2"
            )
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
        # Slab decomposition for vector spaces
        # First index is vector component, which is not sharded.
        self._spmd_mesh = Mesh(jax.devices(), ("k",))
        self._spectral_sharding: NamedSharding | None = (
            None
            if len(jax.devices()) == 1
            else NamedSharding(self._spmd_mesh, P(None, "k", None))
        )
        # Sharding of arrays in physical space.
        self._physical_sharding: NamedSharding | None = (
            None
            if len(jax.devices()) == 1
            else NamedSharding(self._spmd_mesh, P(None, None, "k", None))
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

    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation."""
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.forward(u[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

    def scalar_product(self, u: Array) -> Array:
        coeffs = []
        for i, space in enumerate(self.tensorspaces):
            ci = space.scalar_product(u[i])
            coeffs.append(ci)
        return jnp.stack(coeffs)

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

    def backward(
        self,
        c: Array,
        kind: MeshKind = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate total (homogeneous + lifting) backward transform."""
        return self.orthogonal.backward(self.to_orthogonal(c), kind=kind, N=N)

    def forward(self, u: Array) -> Array:
        """Solve projection for homogeneous coefficients (lifting removed)."""
        d = self.orthogonal.forward(u)
        return self.from_orthogonal(d)

    def scalar_product(self, c: Array) -> NoReturn:  # ty:ignore[invalid-method-override]
        """Disabled scalar product (non-homogeneous test space)."""
        raise RuntimeError(
            "Scalar product requires homogeneous test space (call on get_homogeneous())"
        )

    def evaluate(self, x: Array, c: Array, use_einsum: bool = False) -> Array:
        """Evaluate direct sum tensor product expansion at scattered points."""
        return self.orthogonal.evaluate(x, self.to_orthogonal(c), use_einsum)

    def evaluate_derivative(self, x: Array, c: Array, k: tuple[int, ...]) -> Array:
        """Evaluate direct sum tensor product expansion at scattered points."""
        return self.orthogonal.evaluate_derivative(x, self.to_orthogonal(c), k)

    def evaluate_mesh(
        self, x: list[Array], c: Array, use_einsum: bool = False
    ) -> Array:
        """Evaluate expansion on tensor mesh (summing lifting parts)."""
        return self.orthogonal.evaluate_mesh(x, self.to_orthogonal(c), use_einsum)

    def backward_primitive(
        self,
        c: Array,
        k: tuple[int, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate total (homogeneous + lifting) backward transform."""
        return self.orthogonal.backward_primitive(
            self.to_orthogonal(c), k=k, kind=kind, N=N
        )

    def to_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped to underlying orthogonal basis."""
        result: Array | None = None
        sharding = self.orthogonal._spectral_sharding
        for f, v in self.tpspaces.items():
            inp = self.bndvals.get(f, c)
            ai = v.to_orthogonal(inp)
            # bndvals are computed replicated (local).  When inp is bndvals
            # rather than c, ai is a single-device array; shard it to the
            # spectral sharding so it can be combined with the sharded result
            # from c.
            if sharding is not None and inp is not c:
                ai = jax.device_put(ai, sharding)
            if result is None:
                result = ai
            else:
                result = result + jnp.pad(
                    ai,
                    [
                        (0, result.shape[0] - ai.shape[0]),
                        (0, result.shape[1] - ai.shape[1]),
                    ],
                )
        assert result is not None
        return result

    def from_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped from underlying orthogonal basis."""
        result: Array | None = None
        sharding = self.orthogonal._spectral_sharding
        for f, v in self.tpspaces.items():
            if f in self.bndvals:
                ai = -v.to_orthogonal(self.bndvals[f])
                if sharding is not None:
                    ai = jax.device_put(ai, sharding)
            else:
                ai = c
            if result is None:
                result = ai
            else:
                result = result + jnp.pad(
                    ai,
                    [
                        (0, result.shape[0] - ai.shape[0]),
                        (0, result.shape[1] - ai.shape[1]),
                    ],
                )
        assert result is not None
        return self.get_homogeneous().from_orthogonal(result)
