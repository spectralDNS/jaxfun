# ruff: noqa: E402
import os

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

    # Initialize jax.distributed
    jdist.initialize(
        num_processes=world,
        process_id=rank,
    )


initialize_distributed()

import time
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import Array, grad, random, vmap
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxfun.pinns.mesh import Line
from jaxfun.utils import Domain, jacn

rank = jax.process_index()
world = jax.process_count()
num_devices = len(jax.devices())

ai = 10.0
omega = 20.0
alpha = 0.1
phi = jnp.pi / 2
A = 1.0

print("JAX running on", jax.devices()[0].platform.upper())


def dampened_harmonic_oscillator(
    t: float, omega: float, alpha: float, phi: float, A: float
) -> float:
    inner = jnp.sqrt(1 - alpha**2) * omega * t + phi
    return A * jnp.exp(-alpha * omega * t) * jnp.sin(inner)


oscillator = vmap(
    partial(dampened_harmonic_oscillator, omega=omega, alpha=alpha, phi=phi, A=A)
)

# Create mesh for sharding the mesh across local devices
local_mesh = Mesh(jax.local_devices(), ("local_batch",))
local_batch = NamedSharding(local_mesh, P("local_batch"))

# Each process gets N // world points, total global shape is (N, 1)
N = 10000
N_PER_PROCESS = N // jax.process_count()
global_shape = (N, 1)

domain = Domain(0, 1)
mesh = Line(domain.lower, domain.upper, key=nnx.Rngs(1000 + rank)())

# Get local data points of shape (N_PER_PROCESS, 1), that reside on single local device
x_process = mesh.get_points(N+2, domain="inside", kind="random")

# Shard this local data across local devices
x_device = jax.device_put(x_process, local_batch)

# Create global array just for visualization (not used in computation)
x_global = jax.make_array_from_process_local_data(
    NamedSharding(Mesh(jax.devices(), ("batch",)), P("batch")),
    x_process,
    global_shape,
)

dye = lambda x: jacn(oscillator, 1)(x).reshape((-1, 1))
d2ye = lambda x: jacn(oscillator, 2)(x).reshape((-1, 1))

y = oscillator(x_device)
dy = dye(x_device)
d2y = d2ye(x_device)

y_data = d2y + y

xb = mesh.get_points(2, domain="boundary")
xb = jax.device_put(xb, NamedSharding(local_mesh, P()))
u0 = oscillator(xb)


class MLP(nnx.Module):
    def __init__(
        self, in_size: int, hidden_size: int, out_size: int, *, rngs: nnx.Rngs
    ) -> None:
        init = nnx.initializers.glorot_normal()
        initb = nnx.initializers.zeros_init()

        self.linear1 = nnx.Linear(
            in_size,
            hidden_size,
            rngs=rngs,
            bias_init=initb,
            kernel_init=init,
            param_dtype=float,
            dtype=float,
        )
        self.linear2 = nnx.Linear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            bias_init=initb,
            kernel_init=init,
            param_dtype=float,
            dtype=float,
        )
        self.linear3 = nnx.Linear(
            hidden_size,
            hidden_size,
            rngs=rngs,
            bias_init=initb,
            kernel_init=init,
            param_dtype=float,
            dtype=float,
        )
        self.linear4 = nnx.Linear(
            hidden_size,
            out_size,
            rngs=rngs,
            bias_init=initb,
            kernel_init=init,
            param_dtype=float,
            dtype=float,
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.tanh(self.linear1(x))
        x = nnx.tanh(self.linear2(x))
        x = nnx.tanh(self.linear3(x))
        return self.linear4(x)


model = MLP(1, 32, 1, rngs=nnx.Rngs(1000 + rank))
model = jax.tree.map(lambda f: jax.device_put(f, NamedSharding(local_mesh, P())), model)

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
    st = nnx.state(model)
    pyt, ret = jax.flatten_util.ravel_pytree(st)
    print(f"Degrees of freedom in module: {pyt.shape[0]}")
    print("Sharding of x_device living on rank 0:")
    jax.debug.visualize_array_sharding(x_device)
    print("Sharding of x_global (unused):")
    jax.debug.visualize_array_sharding(x_global)


opt = optax.adam(learning_rate=1e-3)

optlbfgs = optax.lbfgs(
    memory_size=100,
    linesearch=optax.scale_by_zoom_linesearch(25, max_learning_rate=1.0),
)
# opthess = hess(
#    use_lstsq=False,
#    cg_max_iter=1000,
#    linesearch=optax.scale_by_zoom_linesearch(25, max_learning_rate=1.0),
# )

df = lambda mod, x, k: jacn(mod, k)(x).reshape((-1, 1))

gw = jnp.ones(2)
gw = jax.device_put(gw, NamedSharding(local_mesh, P()))


def get_loss_fn(x, target, x_b, t_b):
    @nnx.jit
    def loss_fn(model: nnx.Module, x, target, x_b, t_b) -> Array:
        d2ypred = df(model, x, 2)
        y = model(x)
        ay = y
        u = model(x_b)
        return (
            jnp.array(
                [
                    ((d2ypred + ay - target) ** 2).mean(),
                    ((u - t_b) ** 2).mean(),
                ]
            )
            @ gw
        )

    #return loss_fn
    return partial(loss_fn, x=x, target=target, x_b=x_b, t_b=t_b)


def get_train_step(x: Array, target: Array, x_b: Array, t_b: Array):

    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x, target, x_b, t_b) -> Array:
        gd, state = nnx.split(model)
        # unravel = jax.flatten_util.ravel_pytree(state)[1]
        loss_fn = get_loss_fn(x, target, x_b, t_b)
        loss, gradients = nnx.value_and_grad(loss_fn)(model)
        loss_fn_split = lambda state: loss_fn(nnx.merge(gd, state))
        # H_loss_fn = lambda flat_weights: loss_fn(nnx.merge(gd, unravel(flat_weights)))
        optimizer.update(
            model,
            gradients,
            grad=gradients,
            value_fn=loss_fn_split,
            value=loss,
            # H_loss_fn=H_loss_fn,
        )

        return loss, gradients

    return partial(train_step, x=x, target=target, x_b=x_b, t_b=t_b)


train_step = get_train_step(x_device, y_data, xb, u0)

def run_optimizer(model, opt, num, name, epoch_print=100):
    #cached_train_step = nnx.cached_partial(train_step, model, opt)
    for epoch in range(1, num+1):
        loss, _ = train_step(model, opt)
        #loss, _ = cached_train_step()
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < 1e-20:
            break


opt_adam = nnx.Optimizer(model, opt, wrt=nnx.Param)

loss_fn = get_loss_fn(x_device, y_data, xb, u0)
loss = loss_fn(model)
print("Initial loss", loss, loss.devices())

t0 = time.time()
run_optimizer(model, opt_adam, 1000, "Adam", epoch_print=100)

print("Final loss", loss_fn(model), time.time() - t0)

raise SystemExit

t1 = time.time()
print("Time Adam", t1 - t0)

loss = loss_fn(model)
print("loss", loss, loss.sharding)
print("kernel", model.linear1.kernel.sharding)

raise SystemExit

# opt_hess = nnx.Optimizer(model, opthess)
opt_lbfgs = nnx.Optimizer(model, optlbfgs, wrt=nnx.Param)

run_optimizer(model, opt_lbfgs, 1000, "LBFGS")
t2 = time.time()
print("Time LBFGS", t2 - t1)

# run_optimizer(model, opt_hess, 10, "Hess", 1)

print("Time", time.time() - t0)

# model = unfreeze_layer(model, 'linear2')


def print_error(t0):
    print(
        "Accuracy f(x)=",
        jnp.linalg.norm((model(t0) - oscillator(t0)) / len(t0))
        / jnp.linalg.norm((oscillator(t0)) / len(t0)),
        "f'(x)=",
        jnp.linalg.norm((df(model, t0, 1) - dye(t0)) / len(t0))
        / jnp.linalg.norm((dye(t0)) / len(t0)),
        "f''(x)=",
        jnp.linalg.norm((df(model, t0, 2) - d2ye(t0)) / len(t0))
        / jnp.linalg.norm((d2ye(t0)) / len(t0)),
    )


t0 = jnp.linspace(0, 0.5, 1000)[:, None]
print_error(t0)
st = nnx.state(model)
dim = jax.flatten_util.ravel_pytree(st)[0].shape[0]
print("dim", dim)
y_pred = model(x_device)

# plt.plot(x_device, y, "*", label="True data")
# plt.plot(x_device, y_pred, "s", color="orange", label="Trained NN")
# plt.axvline(0.5, color="k", linestyle="--", label="Training cutoff")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Predictions of the trained neural network")
# plt.show()
