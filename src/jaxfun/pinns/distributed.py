import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree


# Identity function is at the top level so that `process_allmean` doesn't
# recompile on every invocation.
def _identity_fn(x):
    return x


def process_allmean(inp: PyTree) -> PyTree:
    """Reduce across processes by summation.

    Args:
        inp: Pytree to be reduced.

    Returns:
        Reduced Pytree.
    """
    if jax.process_count() == 1:
        return inp
    flat_arr, unravel = jax.flatten_util.ravel_pytree(inp)
    global_shape = (jax.process_count(), flat_arr.shape[0])
    flat_arr = jnp.expand_dims(flat_arr, axis=0)
    global_mesh = np.array(jax.devices()).reshape(
        (jax.process_count(), jax.local_device_count())
    )
    mesh = jax.sharding.Mesh(global_mesh, ("processes", "local_devices"))
    shard_batch = jax.sharding.NamedSharding(mesh, P("processes"))
    global_arr = jax.make_array_from_process_local_data(
        shard_batch, flat_arr, global_shape
    )
    out = jax.jit(_identity_fn, out_shardings=jax.NamedSharding(mesh, P()))(
        global_arr.sum(axis=0) / jax.process_count()
    )
    return unravel(out.addressable_data(0))
