from functools import partial

import jax


class FrozenLayer:
    def __init__(self, layer):
        self.layer = layer
        
    @partial(jax.jit, static_argnums=0)
    def __call__(self, x):
        return self.layer(x)

def freeze_layer(model, layer, i=None):
    if i is None:
        model.__setattr__(layer, FrozenLayer(model.__getattribute__(layer)))
    else:
        model.__getattribute__(layer)[i] = FrozenLayer(model.__getattribute__(layer)[i]) 
    return model

def unfreeze_layer(model, layer, i=None):
    if i is None:
        model.__setattr__(layer, model.__getattribute__(layer).layer)
    else:
        model.__getattribute__(layer)[i] = model.__getattribute__(layer)[i].layer
    return model
