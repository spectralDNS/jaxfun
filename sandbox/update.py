import jax
import jax.numpy as jnp
import optax
from flax import nnx


class Loss:
    def __init__(self, x: jax.Array):
        self.global_weights = jnp.array([1.0])
        self.x = x

    def update_global_weights(self, model: nnx.Module, gw: jax.Array) -> jax.Array:
        new = jnp.array([1.0])
        gw = gw * 0.5 + new * 0.5
        return gw

    def update_global_weights2(
        self, model: nnx.Module, gw: jax.Array) -> jax.Array:
        new = jnp.array([1.0])
        gw = 0.5 + new * 0.5
        return gw

    def __call__(self, model: nnx.Module) -> float:
        #self.global_weights = self.update_global_weights(model, self.global_weights)
        self.global_weights = self.update_global_weights2(model, self.global_weights)
        return jnp.mean(self.global_weights * (model(self.x) ** 2))


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer) -> float:
    gd, state = nnx.split(model, nnx.Param)
    loss_fn_split = lambda state: loss_fn(nnx.merge(gd, state))
    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(
        gradients, grad=gradients, grads=gradients, value_fn=loss_fn_split, value=loss
    )
    return loss


class MLP(nnx.Module):
    def __init__(self, h: int, rngs: nnx.Rngs) -> None:
        self.linear_in = nnx.Linear(1, h, rngs=rngs)
        self.linear_out = nnx.Linear(h, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.tanh(self.linear_in(x))
        return self.linear_out(x)


opt = optax.lbfgs()

x = jnp.linspace(-1, 1, 5)[:, None]
loss_fn = Loss(x)
module = MLP(8, rngs=nnx.Rngs(11))
optimizer = nnx.Optimizer(module, opt, wrt=nnx.Param)
loss = train_step(module, optimizer)
print(loss)
