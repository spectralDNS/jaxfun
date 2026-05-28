from __future__ import annotations

import jax
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxfun.typing import Array, ArrayFun

spmd_mesh = Mesh(jax.devices(), ("k",))
spectral_sharding = NamedSharding(spmd_mesh, P("k"))
physical_sharding = NamedSharding(spmd_mesh, P(None, "k"))


def get_transposed_sharding(sharding: NamedSharding) -> NamedSharding:
    """Return the sharding with unsharded and sharded axes transposed."""
    if sharding == spectral_sharding:
        return physical_sharding
    elif sharding == physical_sharding:
        return spectral_sharding
    else:
        raise ValueError(f"Provided {sharding} does not match spectral or physical.")


def _build_local_apply_fn(dim: int, ax: int, fn: ArrayFun) -> ArrayFun:
    """Return a ``jax.jit(jax.vmap(...))`` that applies *fn* along *ax*.

    The resulting callable operates on a plain (non-sharded) local array,
    so JAX compiles it once and reuses the compiled binary on every call.
    """
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


def _apply_separable_spmd_shard_map(
    c: Array, fns: tuple[ArrayFun, ...], sharding: NamedSharding, cache: dict
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
    if cache_key not in cache:
        dim = c.ndim
        spec = sharding.spec
        sharded = [ax for ax in range(dim) if ax < len(spec) and spec[ax] is not None]
        unsharded = [ax for ax in range(dim) if ax not in sharded]
        transposed = get_transposed_sharding(sharding)

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

        cache[cache_key] = jax.jit(
            shard_map(
                _kernel,
                mesh=sharding.mesh,
                in_specs=(sharding.spec,),
                out_specs=transposed.spec,
                check_vma=False,
            )
        )

    return cache[cache_key](c)


# Experimental:
def _apply_separable_spmd(
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
    dim = c.ndim
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
    # The two pre-built shardings are each other's transpose by construction.
    transposed = get_transposed_sharding(sharding)

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
