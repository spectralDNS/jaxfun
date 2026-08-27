from __future__ import annotations

import jax
import jax.core
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, Sharding

from jaxfun.typing import Array, ArrayFun

spmd_mesh = Mesh(jax.devices(), ("k",))
spectral_sharding = NamedSharding(spmd_mesh, P("k"))
physical_sharding = NamedSharding(spmd_mesh, P(None, "k"))


replicated_sharding = NamedSharding(spmd_mesh, P())

# The batched counterparts of `spectral_sharding` / `physical_sharding`, for the
# `(n_fields, k, y)` arrays the transforms carry when several fields share one
# matrix product. The leading axis is the field batch and is never split; which
# of the remaining two is, is exactly the distinction the rank-2 pair makes.
batched_spectral_sharding = NamedSharding(spmd_mesh, P(None, "k", None))
batched_physical_sharding = NamedSharding(spmd_mesh, P(None, None, "k"))


def replicate(x):
    """Return `x` with a full copy on every device, or unchanged if it has none.

    For the small assembled arrays an integrator *stores* -- forcing vectors,
    boundary liftings -- as opposed to the state it is handed each step.

    Assembly places its right-hand side on `spectral_sharding`, which is what a
    top-level `A.solve(b)` wants. Held on the integrator it is the wrong thing:
    the integrator goes into `_advance` as a static argument, so everything
    under it is reached through the *closure*, and JAX refuses to close over an
    array spanning devices this process cannot address. Replicating costs one
    gather, once, at construction, on a vector the size of a single field --
    against a run that would otherwise not start.

    A no-op on one device, and on anything that is not an array (an initial
    condition may still be a SymPy expression at this point).
    """
    if len(jax.devices()) <= 1 or not isinstance(x, jax.Array):
        return x
    return jax.device_put(x, replicated_sharding)


def state_sharding(shape: tuple[int, ...]) -> NamedSharding:
    """Return the sharding a coefficient array of this shape should carry.

    Spectral coefficients split along the leading (Fourier) axis, which is what
    `spectral_sharding` expresses and what the per-wavenumber solvers and the
    separable transforms both assume. Anything the mesh cannot divide evenly
    stays replicated, and so does anything one-dimensional: a rank-1 state is a
    single wavenumber's profile, whose one axis is a non-Fourier direction and
    so is never the split axis.
    """
    n = len(jax.devices())
    if len(shape) >= 2 and shape[0] % n == 0:
        return spectral_sharding
    return replicated_sharding


def pin_state[StateT](state: StateT) -> StateT:
    """Constrain every leaf of a coefficient state to its `state_sharding`.

    A no-op on one device. On more than one it is what keeps a jitted step
    compilable at all, and the reason is indirect: left unconstrained, the state
    gives GSPMD freedom to choose layouts, and it exercises that freedom
    *backwards*, onto the small replicated operator arrays the step closes over.
    The shardings it proposes for them are regularly ones they cannot take --
    `P("k")` on a scalar step size, or on a `(1, n)` array of Fourier diagonals
    -- and the failure surfaces far from the cause: an `IndexError` out of
    `named_sharding_to_xla_hlo_sharding`, or a fatal PJRT abort over an argument
    count when a hoisted constant goes missing on a jit cache hit.

    Pinning the state removes the freedom. Every array whose layout matters is
    then stated rather than inferred, and the operators stay replicated because
    nothing suggests otherwise.
    """
    if len(jax.devices()) <= 1:
        return state
    return jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(x, state_sharding(x.shape)), state
    )


def get_transposed_sharding(sharding: NamedSharding) -> NamedSharding:
    """Return the sharding with unsharded and sharded axes transposed."""
    if sharding == spectral_sharding:
        return physical_sharding
    elif sharding == physical_sharding:
        return spectral_sharding
    elif sharding == batched_spectral_sharding:
        return batched_physical_sharding
    elif sharding == batched_physical_sharding:
        return batched_spectral_sharding
    else:
        raise ValueError(f"Provided {sharding} does not match spectral or physical.")


def place(z: Array, sharding: Sharding | None) -> Array:
    """Place ``z`` on ``sharding``, or return it unchanged when that is not possible.

    A no-op for a traced array: a tracer carries no placement, and adding one to
    a `device_put` result raises `Received incompatible devices`. The batched
    transforms run their bodies under `vmap`, so every placement inside them
    meets a tracer. Placement is a locality optimization, never a correctness
    one, so skipping it there costs nothing but locality.
    """
    if sharding is None or isinstance(z, jax.core.Tracer):
        return z
    try:
        return jax.device_put(z, sharding)
    except ValueError:  # sharding does not divide the array evenly
        return z


def _build_local_apply_fn(dim: int, ax: int, fn: ArrayFun) -> ArrayFun:
    """Return a ``jax.jit(jax.vmap(...))`` that applies *fn* along *ax*.

    The resulting callable operates on a plain (non-sharded) local array,
    so JAX compiles it once and reuses the compiled binary on every call.

    ``fn`` is a 1-D transform, so every other axis is mapped over: one `vmap`
    per axis, wrapped smallest-index innermost. Each level then removes an index
    larger than any the levels below it refer to, which is what keeps their
    `in_axes` valid without renumbering.

    ``dim`` is the rank of the arrays the result will be called with, which need
    not be the dimensionality of the space -- a batched transform passes one
    more, and `ax` is then the axis in the batched array.
    """
    out = fn
    for other in sorted(set(range(dim)) - {ax}):
        out = jax.vmap(out, in_axes=other, out_axes=other)
    return jax.jit(out)


def _apply_separable_spmd_shard_map(
    c: Array,
    fns: tuple[ArrayFun, ...],
    sharding: NamedSharding,
    cache: dict,
    batch_dims: int = 0,
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

    Args:
        c: The array to transform.
        fns: One local transform per *space* axis, in axis order. Each is built
            for the rank ``c`` actually has, so with ``batch_dims`` non-zero
            ``fns[i]`` acts on axis ``batch_dims + i``.
        sharding: How ``c`` is split. The output carries its transpose.
        cache: Where the compiled kernel is kept, keyed so that each
            ``(fns, spec, batch_dims)`` combination compiles once.
        batch_dims: Number of leading axes that are not space axes. They are
            carried along replicated: they take no transform, and they are
            excluded from the choice of ``split_axis``/``concat_axis``, which
            has to fall on a space axis for the transpose to mean anything.

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
    cache_key = ("shard_map_kernel", id(fns), sharding.spec, batch_dims)
    if cache_key not in cache:
        spec = sharding.spec
        space = range(batch_dims, c.ndim)
        sharded = [ax for ax in space if ax < len(spec) and spec[ax] is not None]
        unsharded = [ax for ax in space if ax not in sharded]
        transposed = get_transposed_sharding(sharding)

        def _kernel(c_loc: Array) -> Array:
            # Phase 1 — unsharded axes: fully local, no communication.
            for ax in unsharded:
                c_loc = fns[ax - batch_dims](c_loc)
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
                c_loc = fns[ax - batch_dims](c_loc)
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
