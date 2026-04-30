from flax import nnx
import jax.numpy as jnp

class Model(nnx.Module):
  def __init__(self, rngs):
    self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    self.linear2 = nnx.Linear(3, 4, rngs=rngs)
  def __call__(self, x, add=0):
    x = self.linear1(x)
    self.sow(nnx.Intermediate, 'i', x+add)
    x = self.linear2(x)
    return x

x = jnp.ones((1, 2))
model = Model(rngs=nnx.Rngs(0))
assert not hasattr(model, 'i')

y = model(x)
assert hasattr(model, 'i')
assert len(model.i) == 1 # tuple of length 1
assert model.i[0].shape == (1, 3)

y = model(x, add=1)
assert len(model.i) == 2 # tuple of length 2
assert (model.i[0] + 1 == model.i[1]).all()

intermediates = nnx.pop(model, nnx.Intermediate)