# Solve Poisson's equation
# ruff: noqa: E402
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_num_cpu_devices", 2)


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

    # Initialize jax.distributed
    jdist.initialize(
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

# Note that spaces and modules (FlaxFunctions) are created on local device 0
# on each rank. Since they have the same random seed, they will be initialized
# identically across ranks. All weights will be put on only one local device,
# which is the default device set for the process. The weights will need to be
# syncronized periodically during training.
#
# The mesh is created on each process, and then distributed further to local
# devices. Each local device gets a different portion of the data created on the
# process they belong to.

# Create mesh for sharding the mesh across local devices
local_mesh = Mesh(jax.local_devices(), ("local_batch",))
local_batch = NamedSharding(local_mesh, P("local_batch"))

domain = Domain(-sp.pi, sp.pi)
V = MLPSpace([32], dims=1, rank=0, name="V")
# V = Chebyshev.Chebyshev(32, dims=1, rank=0, name="V", domain=domain)
w = FlaxFunction(V, name="w")
x = V.system.x
ue = sp.sin(x) + x**2

w.module = jax.tree.map(
    lambda f: jax.device_put(f, NamedSharding(local_mesh, P())), w.module
)

# Each process gets N // world points, total global shape is (N, 1)
N = 1000
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

# Shard this local data across local devices
x_device = jax.device_put(x_process, local_batch)

# Create global array just for visualization (not used in computation)
x_global = jax.make_array_from_process_local_data(
    NamedSharding(Mesh(jax.devices(), ("batch",)), P("batch")),
    x_process,
    global_shape,
)

xb = mesh.get_points_on_domain()  # Just two points so never mind the sharding
xb = jax.device_put(xb, NamedSharding(local_mesh, P()))
ub = lambdify(x, ue)(xb)

if rank == 0:
    print(
        f"JAX distributed with {world} processes and {jax.local_device_count()} local devices per process"  # noqa: E501
    )
    print(f"In total using {jax.device_count()} devices")
    print(f"Global shape (x_global): {x_global.shape}")
    print(f"Local shape (x_device): {x_device.shape}")
    print(f"Addressable shape on device 0: {x_device.addressable_data(0).shape}")
    print(f"x_device is fully addressable: {x_device.is_fully_addressable}")
    print(f"x_global is fully addressable: {x_global.is_fully_addressable}")
    print("Sharding of x_device living on rank 0:")
    jax.debug.visualize_array_sharding(x_device)
    print("Sharding of x_global (unused):")
    jax.debug.visualize_array_sharding(x_global)

# Equations to solve. Note that we use the fully addressable x_device
f = Div(Grad(w)) - w - (Div(Grad(ue)) - ue)
loss_fn = LSQR((f, x_device), (w, xb, ub))
opt_adam = adam(w, learning_rate=1e-3)
opt_lbfgs = lbfgs(w, memory_size=20)
opt_hess = GaussNewton(w)
trainer = Trainer(loss_fn)

trainer.global_weights = jax.device_put(
    trainer.global_weights, NamedSharding(local_mesh, P())
)
for res in loss_fn.residuals:
    res.target = jax.device_put(res.target, NamedSharding(local_mesh, P()))
    res.weights = jax.device_put(res.weights, NamedSharding(local_mesh, P()))

t0 = time.time()
# Train with periodic allreduce

trainer.train(
    opt_adam,
    1000,
    print_final_loss=True,
    epoch_print=100,
    allreduce_grads_and_loss=True,
    allreduce_module_freq=1,
    abs_limit_change=0,
)
print(f"Time Adam {time.time() - t0:.1f}s")
trainer.train(
    opt_lbfgs,
    1,
    print_final_loss=True,
    abs_limit_change=0,
    allreduce_grads_and_loss=True,
    allreduce_module_freq=1,
)
t0 = time.time()
trainer.train(
    opt_lbfgs,
    10,
    epoch_print=1,
    abs_limit_change=0,
    allreduce_grads_and_loss=False,
    allreduce_module_freq=10,
)
loss = loss_fn(w.module)
print("loss", loss, loss.sharding)
print("gw", trainer.global_weights.sharding)
print(f"LBFS time {time.time() - t0:.1f}s")
# trainer.train(
#    opt_hess,
#    10,
#    epoch_print=1,
#    abs_limit_change=0,
#    allreduce_grads_and_loss=True,
#    allreduce_module_freq=1,
# )

print(f"Total time {time.time() - t0:.1f}s")
