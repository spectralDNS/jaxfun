import jax
from flax import nnx

from jaxfun import *
from jaxfun.pinns.module import FlaxFunction, MLPSpace
from jaxfun.utils.common import jacn

mod1 = MLPSpace(1, 8, 1)
mod2 = MLPSpace(1, 8, 2)
mod3 = MLPSpace(1, 8, 3)
v1 = FlaxFunction(mod1, rngs=nnx.Rngs(101), name='v1')
v2 = FlaxFunction(mod2, rngs=nnx.Rngs(101), name='v2')
v3 = FlaxFunction(mod3, rngs=nnx.Rngs(101), name='v3')

xj = jax.random.uniform(jax.random.PRNGKey(202), 4)[:, None]

df1 = jacn(v1, 1)(xj)
df2 = jacn(v2, 1)(xj)
df3 = jacn(v3, 1)(xj)