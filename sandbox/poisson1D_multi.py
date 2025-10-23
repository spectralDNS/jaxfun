# Solve Poisson's equation
# ruff: noqa: E402
import time

import jax

jax.config.update("jax_enable_x64", True)

import socket

import numpy as np
import sympy as sp
from flax import nnx


def initialize_distributed():
    import jax.distributed as jdist

    try:
        from mpi4py import MPI
    except Exception:
        return

    comm = MPI.COMM_WORLD
    world = comm.Get_size()
    rank = comm.Get_rank()

    if world == 1:
        return

    # Rank 0 chooses a free TCP port on localhost
    if rank == 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        host, port = s.getsockname()
        s.close()
        coord = f"{host}:{port}"
    else:
        coord = None
    coord = comm.bcast(coord, root=0)

    # Initialize jax.distributed
    jdist.initialize(
        coordinator_address=coord,
        num_processes=world,
        process_id=rank,
    )


initialize_distributed()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxfun.operators import Div, Grad
from jaxfun.pinns import (
    LSQR,
    FlaxFunction,
    MLPSpace,
    Trainer,
)
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs
from jaxfun.utils.common import Domain, lambdify

rank = jax.process_index()
world = jax.process_count()
num_devices = len(jax.devices())

if rank == 0:
    print(f"JAX distributed: {world} processes")

domain = Domain(-sp.pi, sp.pi)
V = MLPSpace([16], dims=1, rank=0, name="V")
# V = Chebyshev.Chebyshev(32, dims=1, rank=0, name="V", domain=domain)
w = FlaxFunction(V, name="w")
x = V.system.x
ue = sp.sin(x) + x**2

# Create mesh for sharding across devices/processes
global_mesh = np.array(jax.devices()).reshape(
    (jax.process_count(), jax.local_device_count())
)
ms = Mesh(global_mesh, ("processes", "local_devices"))
shard_batch = NamedSharding(ms, P("processes", None))

# Each process gets N // world points, total global shape is (N, 1)
N = 10000
N_PER_PROCESS = N // jax.process_count()
global_shape = (N, 1)

# Each local device get N // jax.device_count() points
N_PER_DEVICE = N // jax.device_count()

# Create local mesh on this process
mesh = Line(
    N_PER_PROCESS, float(domain.lower), float(domain.upper), key=nnx.Rngs(1000 + rank)()
)

# Get local data points of shape (N_PER_PROCESS, 1), that reside on single local device
x_process = mesh.get_points_inside_domain("random")

# Create sharded array from data local to each process. Note, the array will be
# replicated across local devices
x_local = jax.make_array_from_process_local_data(
    shard_batch,
    x_process,
    global_shape,
)

# Alternatively (same result)
# x_local = jax.make_array_from_single_device_arrays(
#   global_shape,
#   shard_batch,
#   [jax.device_put(x_process, d) for d in jax.local_devices()],
# )

xj = x_local

xb = mesh.get_points_on_domain()

if rank == 0:
    print(world, "processes with", num_devices, "devices")
    print(f"Global shape: {xj.shape}, Local shape: {x_local.shape}")
    print("Global sharding:")
    jax.debug.visualize_array_sharding(xj)

# Equations to solve
f = Div(Grad(w)) - w - (Div(Grad(ue)) - ue)
loss_fn = LSQR((f, xj), (w, xb, lambdify(x, ue)(xb)))

opt_adam = adam(w, learning_rate=1e-3)
opt_lbfgs = lbfgs(w, memory_size=20)
opt_hess = GaussNewton(w)

trainer = Trainer(loss_fn)

t0 = time.time()
# Train with periodic allreduce
trainer.train(
    opt_adam,
    1000,
    print_final_loss=True,
    epoch_print=100,
    allreduce_grads_and_loss=True,
    allreduce_module_freq=1,
)
trainer.train(
    opt_lbfgs,
    100,
    epoch_print=10,
    abs_limit_change=0,
    allreduce_grads_and_loss=True,
    allreduce_module_freq=1,
)
# trainer.train(
#    opt_hess, 10, epoch_print=1, abs_limit_change=0, allreduce_gradients_and_loss=True
# )
# trainer.allreduce(w.module)
print(f"Total time {time.time() - t0:.1f}s")

print("loss", loss_fn(w.module))
