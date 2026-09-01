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
`initialize_distributed()` is a no-op there: it looks for a launcher in the
environment and returns before importing anything if there is none. So a demo
that calls it runs unchanged on one device, on eight, under pytest, and under
`mpirun`.
"""

import os
import socket

_initialized = False

# What each launcher calls the world size. Read instead of asking MPI, because
# asking means importing `mpi4py`, and importing it calls `MPI_Init` -- which is
# not something to do to a process that was never launched under a launcher.
# `mpi4py` is a declared dependency, so a `try: import` around it always
# succeeds and guards nothing.
_WORLD_SIZE_VARS = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH, Intel MPI
    "SLURM_NTASKS",  # Slurm, srun
)


def _launched_world_size() -> int:
    """Return the world size the launcher advertises, or 1 if there is none."""
    for var in _WORLD_SIZE_VARS:
        size = os.environ.get(var)
        if size:
            return int(size)
    return 1


def initialize_distributed() -> None:
    """Initialize `jax.distributed` when run under `mpirun`, else do nothing.

    A no-op when the process was not started by an MPI launcher, and a no-op on
    a single rank, so the demos stay runnable as ordinary scripts and under
    pytest.

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

    world = _launched_world_size()
    if world == 1:
        return

    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"Started under an MPI launcher with {world} ranks, but mpi4py is "
            "not importable, so the ranks cannot agree on a coordinator. "
            "Without it every rank would run the whole problem independently. "
            "Install mpi4py, or run without the launcher."
        ) from exc

    import jax.distributed as jdist

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
