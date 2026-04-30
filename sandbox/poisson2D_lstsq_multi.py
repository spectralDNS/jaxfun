# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_num_cpu_devices", 1) # For testing on non-parallel machines

import socket

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

import time

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxfun import Div, Grad
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.pinns import Loss, FlaxFunction, MLPSpace, Trainer
from jaxfun.pinns.mesh import Rectangle
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs
from jaxfun.utils import lambdify
from jaxfun.utils.common import ulp

rank = jax.process_index()
world = jax.process_count()
num_devices = len(jax.devices())

V = MLPSpace([20, 20], dims=2, rank=0, name="V")
# V = PirateSpace(
#    [20], dims=2, rank=0, name="V", act_fun=nnx.tanh, act_fun_hidden=nnx.swish
# )
# C = FunctionSpace(10, Chebyshev, domain=(-1, 1), name="C")
# V = TensorProduct(C, C, name="V")
w = FlaxFunction(V, name="w")

# Create mesh for sharding the mesh across local devices
local_mesh = Mesh(jax.local_devices(), ("local_batch",))
local_batch = NamedSharding(local_mesh, P("local_batch"))

N = 50

mesh = Rectangle(N, N, -1, 1, -1, 1, key=nnx.Rngs(100 + rank)())

x_process = mesh.get_points_inside_domain("chebyshev")
xyb = mesh.get_points_on_domain("chebyshev")
wi = mesh.get_weights_inside_domain("chebyshev")
wb = mesh.get_weights_on_domain("chebyshev")

N_PER_PROCESS = x_process.shape[0]
global_shape = (N_PER_PROCESS * world, 2)

x, y = V.system.base_scalars()
ue = (1 - x**2) * (1 - y**2)  # manufactured solution

# Shard this local data across local devices
x_device = jax.device_put(x_process, local_batch)
xb_device = jax.device_put(xyb, local_batch)
wi_device = jax.device_put(wi, local_batch)
wb_device = jax.device_put(wb, local_batch)

# Create global array just for postprocessing (not used in computation)
x_global = jax.make_array_from_process_local_data(
    NamedSharding(Mesh(jax.devices(), ("batch",)), P("batch")),
    x_process,
    global_shape,
)

if rank == 0:
    print(
        f"JAX distributed with {jax.process_count()} processes and {jax.local_device_count()} local devices per process"  # noqa: E501
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

f = Div(Grad(w)) + x * w * w.diff(x) - (Div(Grad(ue)) + x * ue * ue.diff(x))

loss_fn = Loss((f, x_device, 0, wi_device), (w, xb_device, 0, wb_device))
trainer = Trainer(loss_fn)

t0 = time.time()

opt_adam = adam(w, learning_rate=1e-3)
trainer.train(opt_adam, 5000, epoch_print=1000, allreduce_module_freq=-1)

print("Time for Adam:", time.time() - t0)
t1 = time.time()

opt_lbfgs = lbfgs(w, memory_size=20)
trainer.train(opt_lbfgs, 10000, epoch_print=1000, allreduce_module_freq=-1)

print("Time for LBFGS:", time.time() - t1)

#opt_hess = GaussNewton(w, use_lstsq=True)
#trainer.train(opt_hess, 10, epoch_print=1, abs_limit_change=0, allreduce_module_freq=1)

print("time", time.time() - t0)

uj = jax.make_array_from_process_local_data(
   NamedSharding(Mesh(jax.devices(), ("batch",)), P("batch")),
   lambdify((x, y), ue)(*x_device.T),
   (global_shape[0],)
)
error = jnp.linalg.norm(w(x_global) - uj) / jnp.sqrt(N_PER_PROCESS)
print("Error", error)
