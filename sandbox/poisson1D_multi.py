# Solve Poisson's equation
# ruff: noqa: E402
import os

import jax

jax.config.update("jax_enable_x64", True)

import socket

import sympy as sp
from flax import nnx


def initialize_distributed():
    import jax.distributed as jdist

    try:
        from mpi4py import MPI
    except Exception:
        # No MPI -> single process
        return

    comm = MPI.COMM_WORLD
    world = comm.Get_size()
    rank = comm.Get_rank()

    if world == 1:
        return

    # One GPU per local rank (if GPUs present)
    try:
        shm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        local_rank = shm.Get_rank()
    except Exception:
        local_rank = rank
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(local_rank))

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

# Use 1 CPU device per MPI rank (each process = one device)
jax.config.update("jax_num_cpu_devices", 1)

import jax.flatten_util
from jax.experimental import multihost_utils as mh
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

# Simple barrier to align start
mh.sync_global_devices("start")

domain = Domain(-sp.pi, sp.pi)
V = MLPSpace([16], dims=1, rank=0, name="V")
# V = Chebyshev.Chebyshev(12, dims=1, rank=0, name="V", domain=domain)
w = FlaxFunction(V, name="w")
x = V.system.x
ue = sp.sin(x) + x**2

# Create mesh for sharding across devices/processes
ms = Mesh(jax.devices(), ("data",))
sharding = NamedSharding(ms, P("data"))

# Each process gets N points, total global shape is (world * N, 1)
N = 600 // world  # 50 points per process for 2 processes
global_shape = (world * N, 1)  # Global shape: (100, 1)

# Create local mesh for this process
mesh = Line(N, float(domain.lower), float(domain.upper), key=nnx.Rngs(1000 + rank)())

# Get local data points - shape will be (50, 1) on each process
x_local = mesh.get_points_inside_domain("random")

# Create globally sharded array from local data across all processes
xj = jax.make_array_from_process_local_data(
    sharding,
    x_local,
    global_shape,
)

xb = mesh.get_points_on_domain()

if rank == 0:
    print(world, "processes with", num_devices, "devices")
    print(f"Global shape: {xj.shape}, Local shape: {x_local.shape}")
    print("Global sharding:")
    jax.debug.visualize_array_sharding(xj)

# Equations to solve
f = Div(Grad(w)) - w - (Div(Grad(ue)) - ue)
loss_fn = LSQR((f, xj), (w, xb, lambdify(x, ue)(xb)))

opt_adam = adam(w.module, learning_rate=1e-3)
opt_lbfgs = lbfgs(w.module, memory_size=20)
opt_hess = GaussNewton(w.module)

print("loss", loss_fn(w.module))

trainer = Trainer(loss_fn)

# Train with periodic allreduce
trainer.train(
    opt_adam, 1000, print_final_loss=True, epoch_print=10000, allreduce_frequency=10
)
trainer.train(
    opt_lbfgs, 100, epoch_print=10, abs_limit_change=0, allreduce_frequency=10
)
trainer.train(opt_hess, 10, epoch_print=1, abs_limit_change=0)
trainer.allreduce(w.module)

# Final barrier for clean exit
mh.sync_global_devices("end")
