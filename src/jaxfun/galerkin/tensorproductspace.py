from __future__ import annotations

import copy
import itertools
import warnings
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from typing import TYPE_CHECKING, NoReturn, cast

import jax
import jax.core
import jax.numpy as jnp
import sympy as sp
from jax import Array, shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxfun.coordinates import CartCoordSys, CoordSys
from jaxfun.sharding import (
    _apply_separable_spmd_shard_map,
    _build_local_apply_fn,
    batched_physical_sharding,
    batched_spectral_sharding,
    physical_sharding,
    place,
    spectral_sharding,
)
from jaxfun.typing import ArrayFun, MeshKind, RankTag
from jaxfun.utils.common import jit_vmap, lambdify

from .composite import BCGeneric, BoundaryConditions, Composite, DirectSum
from .orthogonal import OrthogonalSpace

tensor_product_symbol = "\u2297"
multiplication_sign = "\u00d7"

if TYPE_CHECKING:
    from jaxfun.galerkin import CartesianTensorProductSpace


IndivisibleError = ValueError

# Sentinel restricting TensorProductSpace/DirectSumTPS creation to the
# TensorProduct() factory (and this module's own derived-space helpers) \u2014
# a tensor product space should only ever result from the tensor product
# operation, not be assembled by hand.
_tensorproduct_token = object()


def _cheapest_first(
    shape: tuple[int, ...], sizes: tuple[int, ...], last: int | None = None
) -> tuple[int, ...]:
    """Return the axes of a separable transform ordered cheapest-first.

    Each axis maps ``shape[ax]`` points to ``sizes[ax]``, and applying one costs
    the work along it times the *current* extent of every other axis. The axes
    commute, so doing the ones that shrink the array before the ones that grow it
    is the same arithmetic on smaller operands.

    It matters whenever the padding is not uniform. With the 3/2 rule along
    Fourier and none along the polynomial direction, evaluating the polynomial
    direction first runs its (dense, and by far the more expensive) matrix
    product on 2/3 as many rows.

    Args:
        shape: Current extent of each axis.
        sizes: Extent each axis is transformed to.
        last: Axis forced to the end regardless of cost. The axes only commute
            while every one of them is complex-linear, which a half-spectrum
            axis is not: its `irfft` returns a real array, so running it before
            the other axes have been transformed discards the imaginary parts
            still carrying their information. The result is silently wrong
            rather than an error, so the pin is not an optimisation.
    """
    order = sorted(range(len(sizes)), key=lambda ax: sizes[ax] / shape[ax])
    if last is not None:
        order = [ax for ax in order if ax != last] + [last]
    return tuple(order)


def _validate_hermitian_axis(
    basespaces: Sequence[OrthogonalSpace | DirectSum],
) -> int | None:
    """Return the index of the half-spectrum axis, checking there is at most one.

    A real field's spectrum is Hermitian under a *joint* reflection of every
    axis, `c[-k, -l] = conj(c[k, l])`, which pairs half the coefficients with
    the other half and lets exactly one axis be stored half. Halving two would
    keep one quadrant of four and drop two that nothing determines.

    The axis is additionally required to be the first, which is what both
    orderings assume: the forward transform runs in axis order and the real
    `rfft` has to consume the array while it is still real, and the sharded
    paths transform the unsharded axes first, `physical_sharding` splitting
    axis 1 and `spectral_sharding` axis 0.
    """
    axes = [i for i, space in enumerate(basespaces) if space.is_hermitian_half]
    if not axes:
        return None
    if len(axes) > 1:
        names = ", ".join(basespaces[i].name for i in axes)
        raise ValueError(
            f"at most one axis may store half a Hermitian spectrum, got {len(axes)} "
            f"({names}). A real field's spectrum is Hermitian under a joint "
            "reflection of all axes, so halving one axis already uses up the "
            "symmetry; halving a second would drop coefficients nothing "
            "determines. Keep one and make the rest ordinary Fourier spaces."
        )
    if axes[0] != 0:
        raise ValueError(
            f"a half-spectrum axis must be the first axis, but "
            f"{basespaces[axes[0]].name} is axis {axes[0]}. The forward transform "
            "has to reach it while the "
            "array is still real, and the sharded paths assume the same order. "
            "Put it first in the tensor product."
        )
    return 0


class DirectInstantiationWarning(UserWarning):
    """Raised when a product space is instantiated directly, bypassing its factory."""


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
        leaf: CartesianTensorProductSpace | None = None,
        global_index: int = 0,
        *,
        _token: object = None,
    ) -> None:
        if _token is not _tensorproduct_token:
            warnings.warn(
                "TensorProductSpace should be created via TensorProduct(), "
                "not instantiated directly — a tensor product space is normally "
                "the result of the tensor product operation.",
                DirectInstantiationWarning,
                stacklevel=2,
            )
        from jaxfun.coordinates import CartCoordSys, x, y, z

        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[len(basespaces)])
            if system is None
            else system
        )
        self.basespaces: list[OrthogonalSpace] = list(basespaces)
        self._hermitian_axis = _validate_hermitian_axis(self.basespaces)
        self.name = name
        self.system: CoordSys = system
        self.tensorname = tensor_product_symbol.join([b.name for b in basespaces])
        self._spectral_sharding = spectral_sharding if len(jax.devices()) > 1 else None
        self._physical_sharding = physical_sharding if len(jax.devices()) > 1 else None
        self._spmd_local_fn_cache: dict = {}
        self.global_index = global_index
        self.leaf = leaf

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
    def rank(self) -> RankTag:
        """Return tensor rank (0 for scalar-valued space)."""
        return RankTag.SCALAR

    @property
    def is_orthogonal(self) -> bool:
        """Return True if underlying bases are all orthogonal."""
        return all(space.is_orthogonal for space in self.basespaces)

    @property
    def hermitian_axis(self) -> int | None:
        """Return the axis storing half of a Hermitian spectrum, or None.

        At most one axis may store half, and it has to be the first -- see the
        check in `__init__`. The index matters because the axes of a separable
        transform stop commuting once one of them is a half spectrum, so both
        the forward and the backward orderings have to place it.
        """
        return self._hermitian_axis

    @property
    def is_hermitian_half(self) -> bool:
        """Return True if an axis stores only half of a Hermitian spectrum.

        One such axis is enough to make the whole field real and its
        reconstruction real-linear: the dropped coefficients are the conjugates
        of kept ones under a *joint* reflection of every axis, so folding them
        back weights that axis alone and the result's real part is the field.
        """
        return self._hermitian_axis is not None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return physical-space shape (number of quadrature points per axis)."""
        return tuple(space.num_quad_points for space in self.basespaces)

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

    def weights(
        self,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
        broadcast: bool = True,
    ) -> tuple[Array, ...]:
        """Return tensor weights (as tuple of arrays).

        Args:
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
            MeshKind.UNIFORM).
            N: Optional per-axis counts (defaults each to space.num_quad_points).
            broadcast: If True broadcast each axis array to nd-grid shape.

        Returns:
            Tuple (W0, W1, ...) each either 1D or broadcasted.
        """
        weights = []
        kind = MeshKind(kind)
        N = tuple(
            self.basespaces[ax].num_quad_points if N is None else N[ax]
            for ax in range(len(self))
        )
        for ax, space in enumerate(self.basespaces):
            n_ax = space.num_quad_points if N[ax] is None else cast(int, N[ax])
            if kind == MeshKind.QUADRATURE:
                X = space.quad_points_and_weights(n_ax)[1]
            else:
                X = jnp.ones(n_ax)
            weights.append(self.broadcast_to_ndims(X, ax) if broadcast else X)
        return tuple(weights)

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
            Array (M,) with the per-point weight (the product of per-axis weights).
        """
        mesh = self.mesh(kind, N, broadcast=False)
        return jnp.array(
            list(itertools.product(*[m.flatten() for m in mesh])), dtype=mesh[0].dtype
        )

    def flatweights(
        self,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Return flattened quadrature weights for the tensor-product grid.

        Args:
            kind: Sampling kind.
            N: Optional per-axis counts.

        Returns:
            Array (M,) with Cartesian products of weights.
        """
        weights = self.weights(kind, N, broadcast=False)
        return jnp.prod(
            jnp.array(
                list(itertools.product(*[m.flatten() for m in weights])),
                dtype=weights[0].dtype,
            ),
            axis=-1,
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

    def _resolve_quad_points(self, N: tuple[int | None, ...] | None) -> tuple[int, ...]:
        """Resolve the per-axis quadrature counts a transform was asked for.

        `None` means "that axis's default", and it may appear either as the whole
        argument or as a single entry of it. Both are filled in here rather than
        left to the 1-D transforms, because the result is what the cache key and
        the axis ordering are built from, and neither can work with a `None`.

        Args:
            N: Per-axis counts, any of which may be None, or None for all of them.

        Returns:
            One int per axis.
        """
        resolved: list[int] = []
        for ax in range(len(self)):
            n = None if N is None else N[ax]
            resolved.append(self.basespaces[ax].num_quad_points if n is None else n)
        return tuple(resolved)

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
        nq = self._resolve_quad_points(N)
        cache_key = ("evaluate_mesh", kind, nq)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self),
                    ax,
                    partial(self.basespaces[ax].evaluate_mesh, kind=kind, N=nq[ax]),
                )
                for ax in range(len(self))
            )
        fns = self._spmd_local_fn_cache[cache_key]
        if self._use_spmd(self._spectral_sharding, c.shape, nq):
            # Orders the axes itself, so `fns` stays indexed by axis.
            return _apply_separable_spmd_shard_map(
                c, fns, spectral_sharding, self._spmd_local_fn_cache
            )
        for ax in _cheapest_first(c.shape, nq, self.hermitian_axis):
            c = fns[ax](c)
        return c

    @jit_vmap(in_axes=(0, None), static_argnums=(0,), ndim=1)
    def _evaluate_single_device(self, x: Array, c: Array) -> Array:
        """Evaluate expansion at scattered points — single-device path."""
        dim = len(self)
        T = self.basespaces
        C = [
            T[i].eval_reconstruction(T[i].map_reference_domain(x[i]))
            for i in range(dim)
        ]
        path = "i,j,ij" if dim == 2 else "i,j,k,ijk"
        z = jnp.einsum(path, *C, c)
        return z.real if self.is_hermitian_half else z

    def evaluate(self, x: Array, c: Array) -> Array:
        """Evaluate expansion at scattered points.

        Args:
            x: Stacked coordinate array, shape (n_pts, d).
            c: Coefficient tensor

        Returns:
            Scalar or (n_pts,) array of evaluated values.
        """
        if self._use_spmd(self._spectral_sharding, c.shape):
            dim = len(self)
            T = self.basespaces

            C = [
                T[i].eval_reconstruction(T[i].map_reference_domain(x[:, i]))
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

            z = self._spmd_local_fn_cache[cache_key](c, C[0], *C[1:])
            return z.real if self.is_hermitian_half else z

        return self._evaluate_single_device(x, c)

    def get_orthogonal(self) -> TensorProductSpace:
        """Return underlying orthogonal basis instance."""
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return TensorProductSpace(
            orthogonal_spaces,
            system=self.system,
            name=self.name + "o",
            _token=_tensorproduct_token,
        )

    def _use_spmd(
        self,
        sharding: NamedSharding | None,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...] | None = None,
    ) -> bool:
        """Whether the sharded transform path applies to arrays of these shapes.

        Decided from the shapes, never from an array's placement. Placement is
        not knowable inside `jit`: `Tracer.devices()` raises, and this build of
        JAX does not carry sharding on avals either. Every transform here has to
        be callable from inside a jitted time step -- `BaseIntegrator` takes a
        scalar product of its nonlinear terms at every stage -- so a test that
        only works on concrete arrays makes the whole class unusable there. A
        shape is static under trace and answers the same question.

        Exactly two extents have to divide by the device count, and they are not
        all of them:

        * the axis actually split across devices, for the sharding itself;
        * the axis `lax.all_to_all` splits to transpose that -- `unsharded[0]`,
          at the extent it has *after* the local phase, which is its extent in
          `out_shape`.

        The others are free. A composite polynomial axis with two boundary
        conditions carries `N - 2` coefficients against `N` quadrature points,
        and only the quadrature count is ever split, so `N - 2` may be any
        number at all.

        Raises rather than quietly running on one device when a required extent
        does not divide. The transform is the last component that used to fall
        back silently; `_check_shardable` in the wavenumber solver and the
        placement in `inner.py` both refuse the same configuration, and a
        transform that halves its own throughput without saying so is a worse
        outcome than an error naming the fix.

        Args:
            sharding: `self._spectral_sharding` or `self._physical_sharding` --
                None on a single device, which is the common early exit.
            in_shape: Shape of the array going in, space axes only.
            out_shape: Shape it comes out with, space axes only. Omitted by the
                paths that do not transpose the split, which then constrain only
                the split axis itself.
        """
        if sharding is None:
            return False
        n = len(jax.devices())
        spec = sharding.spec
        sharded = [
            ax for ax in range(len(in_shape)) if ax < len(spec) and spec[ax] is not None
        ]
        unsharded = [ax for ax in range(len(in_shape)) if ax not in sharded]
        required = {"split across devices": in_shape[sharded[0]]}
        if out_shape is not None:
            required["split by the transpose"] = out_shape[unsharded[0]]

        bad = {role: extent for role, extent in required.items() if extent % n}
        if bad:
            detail = ", ".join(f"{extent} ({role})" for role, extent in bad.items())
            raise IndivisibleError(
                f"{self.name}: {detail} does not divide by {n} devices, so this "
                "transform cannot be distributed. A half spectrum stores "
                "N // 2 + 1 coefficients, which is odd for every power-of-two N "
                "-- pass `n_extra` to RFourier (or to TensorProduct(real=True)) "
                "to pad it up to a multiple of the device count, which is what "
                "it does by default. Otherwise choose sizes that divide."
            )
        return True

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

        See `backward_batch` for transforming several coefficient arrays at once.
        """
        nq = self._resolve_quad_points(N)
        if self._use_spmd(self._spectral_sharding, c.shape, nq):
            # Orders the axes itself, so the transforms stay indexed by axis.
            return _apply_separable_spmd_shard_map(
                c, self._backward_fns(nq), spectral_sharding, self._spmd_local_fn_cache
            )
        return self._apply_backward(c, nq)

    def backward_batch(
        self,
        c: Array,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Backward transform of several coefficient arrays at once.

        Args:
            c: Coefficient arrays stacked along one leading batch axis.
            N: Optional per-axis counts, as for `backward`.

        Returns:
            The transformed fields, batch axis first.

        Worth using whenever several fields are transformed together: `vmap`
        turns each per-axis transform into one batched matrix product, which
        keeps the arithmetic intensity up where a product per field does not.
        The result is identical to transforming them one at a time -- this is
        purely how the same arithmetic is issued.

        Sharded coefficients take the same distributed path `backward` does,
        with the batch axis carried along replicated: the fields differ only in
        their values, so each device holds the same wavenumbers of all of them
        and one `all_to_all` transposes the whole batch at once.
        """
        nq = self._resolve_quad_points(N)
        if self._use_spmd(self._spectral_sharding, c.shape[1:], nq):
            return _apply_separable_spmd_shard_map(
                c,
                self._backward_fns(nq, batch_dims=1),
                batched_spectral_sharding,
                self._spmd_local_fn_cache,
                batch_dims=1,
            )
        return jax.vmap(lambda ci: self._apply_backward(ci, nq))(c)

    def _backward_fns(
        self, nq: tuple[int, ...], batch_dims: int = 0
    ) -> tuple[ArrayFun, ...]:
        """Return the cached per-axis backward transforms, indexed by space axis."""
        cache_key = ("backward", nq, batch_dims)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self) + batch_dims,
                    batch_dims + ax,
                    partial(self.basespaces[ax].backward, N=nq[ax]),
                )
                for ax in range(len(self))
            )
        return self._spmd_local_fn_cache[cache_key]

    def _apply_backward(self, c: Array, nq: tuple[int, ...]) -> Array:
        """Apply every axis's backward transform to a local (unsharded) array.

        Kept apart from `backward` so that `backward_batch` can `vmap` it: the
        sharding test in `backward` reads `c.devices()`, which is not available
        on a traced array.
        """
        fns = self._backward_fns(nq)
        for ax in _cheapest_first(c.shape, nq, self.hermitian_axis):
            c = fns[ax](c)
        return c

    def scalar_product(self, u: Array) -> Array:
        """Return tensor of inner products along each axis (separable).

        Args:
            u: Input array.

        Returns:
            Array of inner products along each axis.

        See `scalar_product_batch` for several arrays at once.
        """
        u = self._weight_by_metric(u)
        if self._use_spmd(self._physical_sharding, u.shape, self.num_dofs):
            return _apply_separable_spmd_shard_map(
                u,
                self._scalar_product_fns(),
                physical_sharding,
                self._spmd_local_fn_cache,
            )
        return self._apply_scalar_product(u)

    def scalar_product_batch(self, u: Array) -> Array:
        """Inner products of several arrays at once.

        Args:
            u: Input arrays stacked along one leading batch axis.

        Returns:
            The inner products, batch axis first.

        The batched counterpart of `scalar_product`; see `backward_batch` for
        what batching buys and how a sharded batch is handled.
        """
        if self._use_spmd(self._physical_sharding, u.shape[1:], self.num_dofs):
            return _apply_separable_spmd_shard_map(
                self._weight_by_metric(u),
                self._scalar_product_fns(batch_dims=1),
                batched_physical_sharding,
                self._spmd_local_fn_cache,
                batch_dims=1,
            )
        return jax.vmap(
            lambda ui: self._apply_scalar_product(self._weight_by_metric(ui))
        )(u)

    def _weight_by_metric(self, u: Array) -> Array:
        """Weight by the metric determinant, when the system has a non-unit one."""
        sg = self.system.sg
        if sg == 1:
            return u
        return u * lambdify(self.system.base_scalars(), sg)(*self.mesh())

    def _scalar_product_fns(self, batch_dims: int = 0) -> tuple[ArrayFun, ...]:
        """Return the cached per-axis scalar products, indexed by space axis."""
        cache_key = ("scalar_product", batch_dims)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self) + batch_dims,
                    batch_dims + ax,
                    self.basespaces[ax].scalar_product,
                )
                for ax in range(len(self))
            )
        return self._spmd_local_fn_cache[cache_key]

    def _apply_scalar_product(self, u: Array) -> Array:
        """Apply every axis's scalar product to a local (unsharded) array.

        Kept apart from `scalar_product` so `scalar_product_batch` can `vmap`
        it: the sharding test reads `u.devices()`, unavailable on a tracer.

        In axis order, for the same reason as `_apply_forward`: a half-spectrum
        axis takes an `rfft` and has to see the array while it is still real.
        """
        for fn in self._scalar_product_fns():
            u = fn(u)
        return u

    def forward(self, u: Array) -> Array:
        """Forward transform with optional truncation.

        Args:
            u: Input array.

        Returns:
            Array of forward transform values.

        See `forward_batch` for transforming several arrays at once.
        """
        if self._use_spmd(self._physical_sharding, u.shape, self.num_dofs):
            return _apply_separable_spmd_shard_map(
                u, self._forward_fns(), physical_sharding, self._spmd_local_fn_cache
            )
        return self._apply_forward(u)

    def forward_batch(self, u: Array) -> Array:
        """Forward transform of several arrays at once.

        Args:
            u: Input arrays stacked along one leading batch axis.

        Returns:
            The transformed arrays, batch axis first.

        The batched counterpart of `forward`; see `backward_batch` for what
        batching buys and how a sharded batch is handled.
        """
        if self._use_spmd(self._physical_sharding, u.shape[1:], self.num_dofs):
            return _apply_separable_spmd_shard_map(
                u,
                self._forward_fns(batch_dims=1),
                batched_physical_sharding,
                self._spmd_local_fn_cache,
                batch_dims=1,
            )
        return jax.vmap(self._apply_forward)(u)

    def _forward_fns(self, batch_dims: int = 0) -> tuple[ArrayFun, ...]:
        """Return the cached per-axis forward transforms, indexed by space axis."""
        cache_key = ("forward", batch_dims)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self) + batch_dims,
                    batch_dims + ax,
                    self.basespaces[ax].forward,
                )
                for ax in range(len(self))
            )
        return self._spmd_local_fn_cache[cache_key]

    def _apply_forward(self, u: Array) -> Array:
        """Apply every axis's forward transform to a local (unsharded) array.

        In axis order, which is not arbitrary when an axis stores half a
        Hermitian spectrum: its `rfft` only accepts a real array, so it has to
        run before any other axis has made the array complex. `__init__` pins
        such an axis to index 0, so iterating forwards is what that requires.
        """
        for fn in self._forward_fns():
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

        See `backward_primitive_batch` for several coefficient arrays at once.
        """
        nq = self._resolve_quad_points(N)
        if self._use_spmd(self._spectral_sharding, c.shape, nq):
            # Orders the axes itself, so the transforms stay indexed by axis.
            return _apply_separable_spmd_shard_map(
                c,
                self._backward_primitive_fns(k, nq),
                spectral_sharding,
                self._spmd_local_fn_cache,
            )
        return self._apply_backward_primitive(c, k, nq)

    def backward_primitive_batch(
        self,
        c: Array,
        k: tuple[int, ...],
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate a derivative of several coefficient arrays at once.

        Args:
            c: Coefficient arrays stacked along one leading batch axis.
            k: Tuple of derivative orders along each axis, shared by the batch.
            N: Optional per-axis counts, as for `backward_primitive`.

        Returns:
            The evaluated fields, batch axis first.

        The batched counterpart of `backward_primitive`; see `backward_batch`
        for what batching buys and how a sharded batch is handled. `k` is
        shared, so fields wanting different derivative orders need separate
        calls -- along a polynomial axis each order is a different Vandermonde,
        which is exactly what a batch has to hold fixed.
        """
        nq = self._resolve_quad_points(N)
        if self._use_spmd(self._spectral_sharding, c.shape[1:], nq):
            return _apply_separable_spmd_shard_map(
                c,
                self._backward_primitive_fns(k, nq, batch_dims=1),
                batched_spectral_sharding,
                self._spmd_local_fn_cache,
                batch_dims=1,
            )
        return jax.vmap(lambda ci: self._apply_backward_primitive(ci, k, nq))(c)

    def _backward_primitive_fns(
        self, k: tuple[int, ...], nq: tuple[int, ...], batch_dims: int = 0
    ) -> tuple[ArrayFun, ...]:
        """Return the cached per-axis derivative transforms, indexed by space axis."""
        cache_key = ("backward_primitive", k, nq, batch_dims)
        if cache_key not in self._spmd_local_fn_cache:
            self._spmd_local_fn_cache[cache_key] = tuple(
                _build_local_apply_fn(
                    len(self) + batch_dims,
                    batch_dims + ax,
                    partial(
                        self.basespaces[ax].backward_primitive,
                        k=k[ax],
                        N=nq[ax],
                    ),
                )
                for ax in range(len(self))
            )
        return self._spmd_local_fn_cache[cache_key]

    def _apply_backward_primitive(
        self, c: Array, k: tuple[int, ...], nq: tuple[int, ...]
    ) -> Array:
        """Apply every axis's derivative transform to a local (unsharded) array."""
        fns = self._backward_primitive_fns(k, nq)
        for ax in _cheapest_first(c.shape, nq, self.hermitian_axis):
            c = fns[ax](c)
        return c

    def to_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped to underlying orthogonal basis.

        Args:
            c: Coefficient array.

        Returns:
            Array of coefficients in the orthogonal basis.
        """
        sharding = self._spectral_sharding
        S = [s.S for s in self.basespaces]
        z = c
        for i, Si in enumerate(S):
            z = Si.rmatvec(z, axis=i)

        # Sharded if possible, otherwise replicated -- and never under a trace.
        return place(z, sharding)

    def from_orthogonal(self, c: Array) -> Array:
        """Return coefficients c mapped from underlying orthogonal basis.

        Args:
            c: Coefficient array in orthogonal basis.

        Returns:
            Array of coefficients in the original basis.
        """
        sharding = self._spectral_sharding
        P = [(s.P, s.S) for s in self.basespaces]
        z = c
        for i, (Pi, Si) in enumerate(P):
            z = Pi.solve(Si.matvec(z, axis=i), axis=i)

        # Sharded if possible, otherwise replicated -- and never under a trace.
        return place(z, sharding)


def _halve_leading_fourier(
    basespaces: Sequence[OrthogonalSpace | DirectSum],
    n_extra: int | None = None,
) -> list[OrthogonalSpace | DirectSum]:
    """Return `basespaces` with the leading Fourier axis stored as a half spectrum.

    `RFourier(N)` and `Fourier(N)` resolve the same field, so the swap is in
    place: same quadrature points, same domain, same name, half the
    coefficients. Only the first axis is swapped even when several are Fourier,
    because a real field's spectrum is Hermitian under a *joint* reflection of
    every axis and halving one spends that symmetry -- see
    `_validate_hermitian_axis`.
    """
    from .Fourier import Fourier, RFourier

    out = list(basespaces)
    head = out[0] if out else None
    if isinstance(head, RFourier):
        return out  # already halved; asking twice is not an error
    if not isinstance(head, Fourier):
        where = [i for i, sp_ in enumerate(out) if isinstance(sp_, Fourier)]
        raise ValueError(
            "real=True stores one Fourier axis as half a Hermitian spectrum, so "
            "the first axis has to be Fourier"
            + (
                f", but it is {type(head).__name__} and the Fourier axis is "
                f"{where[0]}. Put the Fourier space first in the tensor product."
                if where
                else f", but got {type(head).__name__} and no axis is Fourier."
            )
        )
    return [
        RFourier(
            head.num_quad_points,
            domain=head.domain,
            system=head.system,
            name=head.name,
            fun_str=head.fun_str,
            n_extra=n_extra,
        ),
        *out[1:],
    ]


def TensorProduct(
    *basespaces: OrthogonalSpace | DirectSum,
    system: CoordSys | None = None,
    name: str = "T",
    real: bool = False,
    n_extra: int | None = None,
) -> TensorProductSpace | DirectSumTPS:
    """Factory returning TensorProductSpace or DirectSumTPS.

    Handles:
      * Deep copy of bases to assign distinct coordinate subsystems
      * Propagation of subsystem coordinates into Composite / DirectSum

    If any axis is a DirectSum (inhomogeneous BC), returns DirectSumTPS.

    Args:
        *basespaces: 1D BaseSpace / DirectSum instances.
        system: Optional global coordinate system.
        name: Base name for the tensor product space(s).
        real: Whether the fields expanded in this space are real. If they are,
            the leading Fourier axis is stored as `RFourier` -- the
            non-negative half of its Hermitian spectrum -- which halves the
            coefficients, the per-wavenumber linear algebra and the transforms
            along every other axis. Declared rather than detected: the
            coefficient count differs between the two, and it is read at
            construction to size operators and initial conditions, long before
            any field exists to inspect. Passing a complex field to a space
            built with `real=True` raises from the forward transform.
        n_extra: Padding for the half spectrum, forwarded to `RFourier`. Only
            meaningful with `real=True`; defaults to whatever the current device
            count needs, which is nothing on one device.

    Returns:
        Instance of TensorProductSpace or DirectSumTPS.
    """
    from jaxfun.coordinates import CartCoordSys, x, y, z

    if real:
        basespaces = tuple(_halve_leading_fourier(basespaces, n_extra))

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

    if any(isinstance(s, DirectSum) for s in basespaces_list):
        return DirectSumTPS(basespaces_list, system, name, _token=_tensorproduct_token)

    assert all(isinstance(s, OrthogonalSpace) for s in basespaces_list)
    return TensorProductSpace(
        cast(list[OrthogonalSpace], basespaces_list),
        system,
        name,
        _token=_tensorproduct_token,
    )


class DirectSumTPS(TensorProductSpace):
    """Tensor product space where one or two basespaces are DirectSums.

    Builds a dictionary of homogeneous tensor-product subspaces produced
    by expanding DirectSum components. Also precomputes boundary lifting
    contributions needed to evaluate / transform functions with
    inhomogeneous boundary conditions in one or two dimensions.

    Args:
        basespaces: List of 1D spaces, some of which may be DirectSums.
        system: Global coordinate system.
        name: Base name for the tensor product space.

    Attributes:
        tpspaces: Mapping from tuples of 1D spaces -> TensorProductSpace.
        bndvals: Dict storing boundary lifting coefficient arrays.
    """

    def __init__(
        self,
        basespaces: list[OrthogonalSpace | DirectSum],
        system: CoordSys,
        name: str = "DSTPS",
        global_index: int = 0,
        leaf: CartesianTensorProductSpace | None = None,
        *,
        _token: object = None,
    ) -> None:
        if _token is not _tensorproduct_token:
            warnings.warn(
                "DirectSumTPS should be created via TensorProduct(), "
                "not instantiated directly — a tensor product space is normally "
                "the result of the tensor product operation.",
                DirectInstantiationWarning,
                stacklevel=2,
            )
        from jaxfun.galerkin.inner import project, project1D

        self.basespaces: list[OrthogonalSpace | DirectSum] = basespaces
        self._hermitian_axis = _validate_hermitian_axis(self.basespaces)
        self.system = system
        self.name = name
        self.bndvals: dict[tuple[OrthogonalSpace, ...], Array] = {}
        self.tensorname = tensor_product_symbol.join([b.name for b in basespaces])
        self._spectral_sharding = spectral_sharding if len(jax.devices()) > 1 else None
        self._physical_sharding = physical_sharding if len(jax.devices()) > 1 else None
        self.global_index = global_index
        self.leaf = leaf

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

        bcindices = [
            i for i, space in enumerate(basespaces) if isinstance(space, DirectSum)
        ]
        if len(basespaces) == 3 and bcindices[0] == 0:
            raise ValueError(
                "DirectSum cannot be the first space in a 3D tensor product."
            )
        has_two_inhomogeneous = len(bcindices) == 2

        projected_bcs: list[list[BoundaryConditions]] = []
        if has_two_inhomogeneous:
            # If there are two DirectSums, we need to project to the other for each.
            # When projecting to the other space, we need to use the BC values
            # corresponding to the current space's BC values.
            bcspaces = (
                cast(DirectSum, basespaces[bcindices[0]]).basespaces[1],
                cast(DirectSum, basespaces[bcindices[1]]).basespaces[1],
            )
            bc_pair = bcspaces
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
                projected_bcs.append([])
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

                    projected_bcs[-1].append(bcs)

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
                self.bndvals[tensorspace] = jnp.array(
                    [z.orderedvals() for z in projected_bcs[0]], dtype=float
                )

            elif len(otherspaces) == 1 and len(bcspaces) == 1:
                bcspace = bcspaces[0]
                uh: list[Array] = []
                for j, bc in enumerate(bcspace.bcs.orderedvals()):
                    otherspace: OrthogonalSpace = otherspaces[0]
                    if has_two_inhomogeneous:
                        bco: BCGeneric = copy.deepcopy(bc_pair[(bcsindex[0] + 1) % 2])
                        bco.bcs = projected_bcs[bcsindex[0]][j]
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
                    if has_two_inhomogeneous:
                        bco: BCGeneric = copy.deepcopy(bc_pair[0 if bcind == 2 else 1])
                        bco.bcs = projected_bcs[bcind - 1][j]
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
                for bci in projected_bcs[0]:
                    for bc0 in bci.orderedvals():
                        uh.append(project(bc0, otherspaces[0]))
                self.bndvals[tensorspace] = jnp.array(uh).T.reshape(
                    (-1, len(projected_bcs[0]), len(projected_bcs[1]))
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
            s: TensorProductSpace(
                s,
                self.system,
                f"{self.name}{i}",
                leaf=self.leaf,
                global_index=self.global_index,
                _token=_tensorproduct_token,
            )
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

    def _apply_backward(self, c: Array, nq: tuple[int, ...]) -> Array:
        """Lift the boundary values, then transform in the orthogonal space.

        The same redirection `backward` makes, as the hook `backward_batch`
        vmaps over -- a direct sum keeps no transform cache of its own.
        """
        return self.orthogonal._apply_backward(self.to_orthogonal(c), nq)

    def _require_local_batch(self, what: str) -> None:
        """Refuse batching while sharding is active.

        Lifting the boundary values adds the field to a boundary contribution
        that `to_orthogonal` has already placed on the space's sharding, and a
        traced array carries no placement to match it against. A plain tensor
        product space does no such mixing: it batches a sharded array through
        the same `shard_map` its unbatched transforms use, carrying the batch
        axis along replicated. A direct sum cannot, and this is also what keeps
        it out of that inherited path -- which applies the base space's own
        per-axis transforms and would skip the lifting entirely. The two
        conditions are complements, `self._spectral_sharding` being set exactly
        when `_use_spmd` can return True, so neither case is ever reached twice
        or missed.
        """
        if self._spectral_sharding:
            raise NotImplementedError(
                f"{what} on a DirectSum space needs a single-device host: the "
                "boundary lifting is placed on the space's sharding, which a "
                "traced array cannot carry. Transform the fields one at a time."
            )

    def backward_batch(
        self,
        c: Array,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        self._require_local_batch("backward_batch")
        return super().backward_batch(c, N=N)

    def forward_batch(self, u: Array) -> Array:
        self._require_local_batch("forward_batch")
        return super().forward_batch(u)

    def forward(self, u: Array) -> Array:
        d = self.orthogonal.forward(u)
        return self.from_orthogonal(d)

    def _apply_forward(self, u: Array) -> Array:
        """Transform in the orthogonal space, then take the lifting back out.

        The same redirection `forward` makes, as the hook `forward_batch` vmaps
        over -- a direct sum keeps no transform cache of its own.
        """
        return self.from_orthogonal(self.orthogonal._apply_forward(u))

    def scalar_product(self, u: Array) -> NoReturn:
        raise RuntimeError(
            "Scalar product requires homogeneous test space (call on get_homogeneous())"
        )

    def scalar_product_batch(self, u: Array) -> NoReturn:
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

    def backward_primitive_batch(
        self,
        c: Array,
        k: tuple[int, ...],
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        self._require_local_batch("backward_primitive_batch")
        return super().backward_primitive_batch(c, k, N=N)

    def _apply_backward_primitive(
        self, c: Array, k: tuple[int, ...], nq: tuple[int, ...]
    ) -> Array:
        """Lift the boundary values, then differentiate in the orthogonal space.

        The same redirection `backward_primitive` makes, as the hook
        `backward_primitive_batch` vmaps over.
        """
        return self.orthogonal._apply_backward_primitive(self.to_orthogonal(c), k, nq)

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
        # Match c's sharding where there is one to match; a traced c has none.
        target = None if isinstance(c, jax.core.Tracer) else c.sharding
        result = c + place(result, target)
        return self.get_homogeneous().from_orthogonal(result)
