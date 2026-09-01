"""Bring up `jax.distributed` across processes, for runs that want more than one.

Two ways in, depending on what launched the job: `initialize_distributed` for an
MPI world (`mpirun`), `initialize_saga` for a Slurm allocation. The demos here
call the first; a cluster job script calls the second from its own program, and
the docstring there says which variables that script has to export.

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
on one node, or `JAX_NUM_CPU_DEVICES=2`. JAX sees those without help, and both
initializers are no-ops there: they look for a launcher in the environment and
return before importing anything if there is none. So a demo that calls one runs
unchanged on one device, on eight, under pytest, and under `mpirun`.
"""

import os
import socket

_initialized = False

# How to tell an MPI launcher put us here, without asking MPI: asking means
# importing `mpi4py`, and importing it calls `MPI_Init`, which is not something
# to do to a process that no launcher started. (`mpi4py` is a declared
# dependency, so a `try: import` around it always succeeds and guards nothing.)
# Only what an MPI launcher sets -- Slurm is `initialize_saga`'s business, and a
# plain `srun` of a non-MPI program is not an MPI world.
_MPI_LAUNCHER_VARS = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH, Intel MPI, MVAPICH
    "PMIX_RANK",  # PMIx, as Open MPI 5 / PRRTE
)


def _under_mpi_launcher() -> bool:
    """Whether an MPI launcher started this process."""
    return any(os.environ.get(var) for var in _MPI_LAUNCHER_VARS)


def initialize_saga() -> None:
    """Initialize `jax.distributed` from a Slurm allocation, as on Saga.

    For clusters where the job script defines the world rather than `mpirun`.
    Call it in place of `initialize_distributed`, and just as early -- the same
    import-ordering rule applies.

    Slurm supplies `SLURM_NTASKS`, `SLURM_PROCID` and `SLURM_LOCALID`, and JAX
    finds the coordinator from the allocation, so no address has to be agreed.
    The job script has to export two more itself:

    * `JAX_PLATFORM=gpu` to take the GPU path; anything else stays on CPU. Read
      here and not by JAX, whose own variable is the similar-looking
      `JAX_PLATFORMS`.
    * `LOCAL_DEVICES`, the number of GPUs each task owns, when the platform is
      `gpu`. `jax.local_device_count()` becomes that, and `jax.device_count()`
      becomes `SLURM_NTASKS * LOCAL_DEVICES`.

    So a script asking for `--nodes=2 --ntasks-per-node=1 --gpus-per-node=2`
    exports `LOCAL_DEVICES=2` and gets 2 processes over 4 GPUs.
    """
    global _initialized
    if _initialized:
        return

    num_processes = int(os.environ.get("SLURM_NTASKS", "1"))
    if num_processes == 1:
        return

    # The coordinator connection is node-to-node inside the allocation; left in
    # place, a cluster's HTTP proxy variables send it out through the proxy and
    # it never arrives.
    for _k in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    ):
        os.environ.pop(_k, None)

    import jax.distributed as jdist

    process_id = int(os.environ.get("SLURM_PROCID", "0"))
    local_ids = None
    if os.environ.get("JAX_PLATFORM", "").lower() == "gpu":
        local_devices = os.environ.get("LOCAL_DEVICES")
        assert local_devices, "LOCAL_DEVICES must be set for GPU distributed runs"
        lid_env = int(os.environ.get("SLURM_LOCALID", "0"))
        local_ids = list(range(lid_env, lid_env + int(local_devices)))

    jdist.initialize(
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_ids,
    )
    _initialized = True


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

    if not _under_mpi_launcher():
        return

    try:
        from mpi4py import MPI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Started under an MPI launcher, but mpi4py is not importable, so "
            "the ranks cannot agree on a coordinator. Without it every rank "
            "would run the whole problem independently. Install mpi4py, or run "
            "without the launcher."
        ) from exc

    # Size and rank from MPI itself, now that asking is free: the environment
    # only had to answer whether there was a launcher at all, and every launcher
    # spells the rest differently.
    comm = MPI.COMM_WORLD
    world, rank = comm.Get_size(), comm.Get_rank()
    if world == 1:
        return

    import jax.distributed as jdist

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
