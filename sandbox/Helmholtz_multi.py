# ruff: noqa: E402
import os
import socket

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_num_cpu_devices", 1)


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
from functools import partial

import jax.numpy as jnp
import optax
import sympy as sp
from flax import nnx

# from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxfun.galerkin import Chebyshev
from jaxfun.operators import Div, Grad
from jaxfun.pinns import (
    FlaxFunction,
    Loss,
    MLPSpace,
    Trainer,
    sPIKANSpace,
)
from jaxfun.pinns.mesh import Line
from jaxfun.pinns.optimizer import GaussNewton, adam, lbfgs
from jaxfun.utils.common import Domain, jacn, lambdify

rank = jax.process_index()
world = jax.process_count()
num_devices = len(jax.devices())

ai = 10.0
omega = 20.0
alpha = 0.1
phi = jnp.pi / 2
A = 1.0


def dampened_harmonic_oscillator(
    t: float, omega: float, alpha: float, phi: float, A: float
) -> float:
    inner = jnp.sqrt(1 - alpha**2) * omega * t + phi
    return A * jnp.exp(-alpha * omega * t) * jnp.sin(inner)


oscillator = jax.vmap(
    partial(dampened_harmonic_oscillator, omega=omega, alpha=alpha, phi=phi, A=A)
)


# Note that spaces and modules (FlaxFunctions) are created on local device 0
# on each rank. Since they have the same random seed, they will be initialized
# identically across ranks. All weights will be put on only one local device,
# which is the default device set for the process. The weights will need to be
# syncronized periodically during training.
#
# The mesh is created on each process, and then distributed further to local
# devices. Each local device get a different portion of the date created on the
# process they belong to.

# Create mesh for sharding the mesh across local devices
local_mesh = Mesh(jax.local_devices(), ("local_batch",))
local_batch = NamedSharding(local_mesh, P("local_batch"))

domain = Domain(0, 1)
#V = Chebyshev.Chebyshev(64, dims=1, rank=0, name="V", domain=domain)
V = MLPSpace([32, 32], dims=1, rank=0, name="V")
#V = sPIKANSpace(8, [4], dims=1, rank=0, name="V", domains=[domain])
w = FlaxFunction(V, name="w", rngs=nnx.Rngs(1000 + rank), kernel_init=nnx.initializers.glorot_normal())

x = V.system.x

# Each process gets N // world points, total global shape is (N, 1)
N = 10000
N_PER_PROCESS = N // jax.process_count()
global_shape = (N, 1)

# Create local mesh on this process
mesh = Line(domain.lower, domain.upper, key=nnx.Rngs(1000 + rank)())

# Get local data points of shape (N_PER_PROCESS, 1), that reside on single local device
x_process = mesh.get_points_inside_domain(N_PER_PROCESS+2, "random")

# Shard this local data across local devices
x_device = jax.device_put(x_process, local_batch)

# Create global array just for visualization (not used in computation)
x_global = jax.make_array_from_process_local_data(
    NamedSharding(Mesh(jax.devices(), ("batch",)), P("batch")),
    x_process,
    global_shape,
)

xb = mesh.get_points_on_domain(2)
xb = jax.device_put(xb, NamedSharding(local_mesh, P()))

dye = lambda x: jacn(oscillator, 1)(x).reshape((-1, 1))
d2ye = lambda x: jacn(oscillator, 2)(x).reshape((-1, 1))

y = oscillator(x_device)
dy = dye(x_device)
d2y = d2ye(x_device)

uj = d2y + y
ub = oscillator(xb)

if rank == 0:
    print(
        f"JAX distributed with {jax.process_count()} processes and {jax.local_device_count()} local devices per process"  # noqa: E501
    )
    print(f"In total using {jax.device_count()} devices")
    print(f"Global shape (x_global): {x_global.shape}, dtype={x_global.dtype}")
    print(f"Local shape (x_device): {x_device.shape}, dtype={x_device.dtype}")
    print(f"Addressable shape on device 0: {x_device.addressable_data(0).shape}")
    print(f"x_device is fully addressable: {x_device.is_fully_addressable}")
    print(f"x_global is fully addressable: {x_global.is_fully_addressable}")
    print(f"Degrees of freedom in module: {w.module.dim}")
    print("Sharding of x_device living on rank 0:")
    jax.debug.visualize_array_sharding(x_device)
    print("Sharding of x_global (unused):")
    jax.debug.visualize_array_sharding(x_global)

# Equations to solve. Note that we use the fully addressable x_device, uj, xb, ub
#wj = mesh.get_weights_inside_domain(N_PER_PROCESS+2, "random")

f = Div(Grad(w)) + x * w
loss_fn = Loss((f, x_device, uj[:, 0]), (w, xb, ub[:, 0]))
trainer = Trainer(loss_fn)

opt_adam = adam(w, learning_rate=1e-3)
opt_lbfgs = lbfgs(w, memory_size=100, max_linesearch_steps=10)

t0 = time.time()

#loss = loss_fn(w.module)
#print("Initial loss", loss, loss.devices())

trainer.train(
    opt_adam,
    1000,
    module=w.module,
    print_final_loss=True,
    epoch_print=100,
    allreduce_grads_and_loss=True,
    allreduce_module_freq=-1,
    update_global_weights=-1,
    abs_limit_change=0,
)

print(f"Time Adam {time.time() - t0:.1f}s")
raise SystemExit(1)
# trainer.train(
#    opt_lbfgs,
#    10,
#    print_final_loss=True,
#    abs_limit_change=0,
#    allreduce_grads_and_loss=False,
#    allreduce_module_freq=1,
# )
t0 = time.time()
trainer.train(
    opt_lbfgs,
    1000,
    epoch_print=100,
    allreduce_grads_and_loss=False,
    allreduce_module_freq=10,
    update_global_weights=10
)


print(f"Time LBFGS {time.time() - t0:.1f}s")
t0 = time.time()
opt_hess = GaussNewton(w, use_lstsq=True, use_GN=True)
trainer.train(
   opt_hess,
   4,
   epoch_print=1,
   abs_limit_change=0,
   allreduce_grads_and_loss=True,
   allreduce_module_freq=1,
)
print(f"Time Hessian {time.time() - t0:.1f}s")

df = lambda mod, x, k: jacn(mod, k)(x).reshape((-1, 1))


def print_error(t0):
    print(
        "Accuracy f(x)=",
        jnp.linalg.norm((w.module(t0) - oscillator(t0)) / len(t0))
        / jnp.linalg.norm((oscillator(t0)) / len(t0)),
        "f'(x)=",
        jnp.linalg.norm((df(w.module, t0, 1) - dye(t0)) / len(t0))
        / jnp.linalg.norm((dye(t0)) / len(t0)),
        "f''(x)=",
        jnp.linalg.norm((df(w.module, t0, 2) - d2ye(t0)) / len(t0))
        / jnp.linalg.norm((d2ye(t0)) / len(t0)),
    )


t0 = jnp.linspace(0, 0.5, 1000)[:, None]
print_error(t0)

try:
    if jax.process_count() > 1:
        # Best-effort barrier so all processes reach teardown together
        try:
            from jax.experimental import multihost_utils as mu

            mu.sync_global_devices("finalize")
        except Exception:
            pass
        import jax.distributed as jdist

        # Best-effort shutdown; ignore errors on teardown
        jdist.shutdown()
except Exception:
    pass
