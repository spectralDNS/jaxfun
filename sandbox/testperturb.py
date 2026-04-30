from flax import nnx
import jax.numpy as jnp


class Model(nnx.Module):
    def __init__(self, rngs):
        self.linear1 = nnx.Linear(2, 3, rngs=rngs)
        self.linear2 = nnx.Linear(3, 4, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.perturb("xgrad", x)
        x = self.linear2(x)
        return x


x = jnp.ones((1, 2))
y = jnp.ones((1, 4))
model = Model(rngs=nnx.Rngs(0))
assert not hasattr(model, "xgrad")  # perturbation requires a sample input run
_ = model(x)
assert model.xgrad.shape == (1, 3)  # same as the intermediate value
graphdef, params, perturbations = nnx.split(model, nnx.Param, nnx.Perturbation)


# Take gradients on the Param and Perturbation variables
@nnx.grad(argnums=(0, 1))
def grad_loss(params, perturbations, inputs, targets):
    model = nnx.merge(graphdef, params, perturbations)
    return jnp.mean((model(inputs) - targets) ** 2)


(grads, perturbations) = grad_loss(params, perturbations, x, y)
# `perturbations.xgrad[...]` is the intermediate gradient
assert not jnp.array_equal(perturbations.xgrad[...], jnp.zeros((1, 3)))
