import jax
import jax.numpy as jnp
from jax._src import array
from jax.experimental.multihost_utils import core, pxla
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
    mesh = jax.sharding.Mesh(jax.devices(), ("processes",))
    flat_arr = jnp.expand_dims(flat_arr, axis=0)
    s = jax.sharding.NamedSharding(mesh, P("processes"))
    aval = core.ShapedArray(flat_arr.shape, flat_arr.dtype)
    global_aval = pxla.mesh_local_to_global(
        mesh, pxla.get_array_mapping(P("processes")), aval
    )
    bufs = [jax.device_put(flat_arr, d) for d in jax.local_devices()]
    global_arr = array.make_array_from_single_device_arrays(global_aval.shape, s, bufs)
    out = jax.jit(_identity_fn, out_shardings=jax.NamedSharding(mesh, P()))(
        global_arr.sum(axis=0) / jax.process_count()
    )
    return unravel(out.addressable_data(0))
