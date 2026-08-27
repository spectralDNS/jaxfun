"""Bring up `jax.distributed` from an MPI world, for demos that want `mpirun`.

Not a demo: imported by the ones that are. `tests/test_demos.py` knows it by
name and does not try to run it.

Import it *first*, before anything that reaches `jaxfun`::

    from spmd_bootstrap import initialize_distributed

    initialize_distributed()

    import jax.numpy as jnp
    from jaxfun...

The ordering is not stylistic. `jaxfun/sharding.py` builds its device mesh from
`jax.devices()` at import time, and `jax.distributed.initialize` is what makes
the other processes' devices visible -- so anything importing `jaxfun` first
gets a one-device mesh and silently runs the serial path on every rank. This
module imports nothing from `jaxfun` at module scope, which is also why it
lives here rather than in the package. (`to_host` reaches for one sharding from
it, deferred to call time, long after the ordering above stops mattering.)

None of it is needed for several devices in a *single* process -- several GPUs
on one node, or `JAX_NUM_CPU_DEVICES=2`. JAX sees those without help, and
`initialize_distributed()` is a no-op there, so a demo that calls it runs
unchanged on one device, on eight, and under `mpirun`.
"""

import socket

_initialized = False


def initialize_distributed() -> None:
    """Initialize `jax.distributed` when run under `mpirun`, else do nothing.

    A no-op without `mpi4py` installed and a no-op on a single rank, so the
    demos stay runnable as ordinary scripts.

    Idempotent: JAX refuses a second `initialize`, and the demos import each
    other -- RayleighBenard builds on ChannelFlow2D, and a script driving either
    one calls this itself before importing it. Whoever gets there first wins and
    the rest are no-ops.

    Rank 0 binds port 0 to have the OS pick a free port, then broadcasts the
    address; that avoids having to agree on a port number in advance, which is
    the usual reason a second run on the same machine fails to start.
    """
    global _initialized
    if _initialized:
        return

    try:
        from mpi4py import MPI
    except ImportError:
        return

    import jax.distributed as jdist

    comm = MPI.COMM_WORLD
    world, rank = comm.Get_size(), comm.Get_rank()
    if world == 1:
        return

    if rank == 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        sock.close()
        coordinator = f"{host}:{port}"
    else:
        coordinator = None
    coordinator = comm.bcast(coordinator, root=0)

    jdist.initialize(
        coordinator_address=coordinator, num_processes=world, process_id=rank
    )
    _initialized = True


def is_leader() -> bool:
    """Whether this is the process that should write files and draw plots.

    Every rank runs the same program, so anything with a side effect outside its
    own memory -- a figure, a saved animation, a progress bar -- has to be asked
    for once rather than `world` times.
    """
    import jax

    return jax.process_index() == 0


def echo(*args, **kwargs) -> None:
    """`print`, but only from the process that should be talking.

    Every rank runs the same program, so an unguarded `print` says everything
    `world` times -- interleaved, and with no indication which copy is which.
    Diagnostics computed from a distributed array are global reductions, so the
    copies agree and only one of them is worth reading.

    A drop-in for `print`: same arguments, and on a single process it *is*
    `print`.
    """
    if is_leader():
        print(*args, **kwargs)


def to_host(tree):
    """Return `tree` as plain numpy arrays, gathering across processes if needed.

    An array split over several processes holds only this one's shard, and
    anything that reads it element by element -- matplotlib, a file writer --
    cannot be served from that.

    Call it *before* the `is_leader()` guard, never inside it. The gather is a
    collective: every process has to reach it, so a rank that has already
    returned leaves the others waiting forever.
    """
    import jax
    import numpy as np

    if jax.process_count() > 1:
        from jaxfun.sharding import replicated_sharding

        tree = jax.device_put(tree, replicated_sharding)
    return jax.tree.map(np.asarray, tree)
