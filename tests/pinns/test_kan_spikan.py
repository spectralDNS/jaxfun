import jax.numpy as jnp
from flax import nnx

from jaxfun.coordinates import BaseTime
from jaxfun.pinns import module as mod, nnspaces as nns


def test_kanlayer_forward_shape_and_params():
    rngs = nnx.Rngs(0)
    layer = mod.KANLayer(in_features=5, out_features=3, spectral_size=4, rngs=rngs)
    x = jnp.linspace(-1.0, 1.0, 15).reshape(3, 5)
    y = layer(x)
    assert y.shape == (3, 3)
    assert hasattr(layer, "kernel")
    assert layer.kernel.value.shape[-1] == 3


def test_kanmlpspace_basic_attributes():
    V = nns.KANMLPSpace(
        spectral_size=5,
        hidden_size=[8, 4],
        dims=2,
        rank=0,
        transient=False,
        name="KANMLP",
    )
    assert V.dims == 2
    assert V.rank == 0
    assert V.out_size == 1
    assert V.in_size >= 2
    bv = V.base_variables()
    assert len(bv) == 2


def test_kanmlpspace_transient_adds_time():
    V = nns.KANMLPSpace(
        spectral_size=4,
        hidden_size=6,
        dims=1,
        rank=0,
        transient=True,
        name="KANMLP",
    )
    assert V.in_size == 2  # 1 spatial + time
    bv = V.base_variables()
    assert isinstance(bv[-1], BaseTime)


def test_spikanspace_basic_attributes():
    V = nns.sPIKANSpace(
        spectral_size=6,
        hidden_size=[5],
        dims=2,
        rank=1,
        transient=False,
        name="sPIKAN",
    )
    assert V.dims == 2
    assert V.rank == 1
    assert V.out_size == 2
    assert V.in_size == 2
    bv = V.base_variables()
    assert len(bv) == 2


def test_spikanspace_transient():
    V = nns.sPIKANSpace(
        spectral_size=3,
        hidden_size=4,
        dims=3,
        rank=0,
        transient=True,
        name="sPIKAN",
    )
    assert V.in_size == 4  # 3 spatial + time
    bv = V.base_variables()
    assert isinstance(bv[-1], BaseTime)


def test_flaxfunction_with_kanmlpspace_builds_module():
    rngs = nnx.Rngs(5)
    V = nns.KANMLPSpace(
        spectral_size=5,
        hidden_size=[6],
        dims=1,
        rank=0,
        transient=False,
        name="KANMLP",
    )
    f = mod.FlaxFunction(V, "u", rngs=rngs)
    x = jnp.zeros((4, V.in_size))
    y = f(x)
    assert y.shape == (4,)
    assert hasattr(f, "module")


def test_flaxfunction_with_spikanspace_forward():
    rngs = nnx.Rngs(6)
    V = nns.sPIKANSpace(
        spectral_size=4,
        hidden_size=[5],
        dims=2,
        rank=0,
        transient=False,
        name="sPIKAN",
    )
    f = mod.FlaxFunction(V, "phi", rngs=rngs)
    x = jnp.zeros((3, V.in_size))
    y = f(x)
    assert y.shape == (3,)


def test_comp_with_kan_and_spikan_spaces():
    rngs = nnx.Rngs(7)
    V1 = nns.KANMLPSpace(
        spectral_size=4,
        hidden_size=[5],
        dims=1,
        rank=0,
        transient=False,
        name="KANMLP",
    )
    V2 = nns.sPIKANSpace(
        spectral_size=3,
        hidden_size=[4],
        dims=1,
        rank=0,
        transient=False,
        name="sPIKAN",
    )
    f1 = mod.FlaxFunction(V1, "u", rngs=rngs)
    f2 = mod.FlaxFunction(V2, "v", rngs=rngs)
    comp = mod.Comp(f1, f2)
    x = jnp.zeros((5, V1.in_size))  # both rank 0 => scalar each
    y = comp(x)
    assert y.shape == (5, 2)
    assert comp.dim == f1.module.dim + f2.module.dim


def test_comp_with_kan_and_spikan_spaces_different_dims():
    rngs = nnx.Rngs(8)
    V1 = nns.KANMLPSpace(
        spectral_size=4,
        hidden_size=[5],
        dims=2,
        rank=0,
        transient=False,
        name="KANMLP",
    )
    V2 = nns.sPIKANSpace(
        spectral_size=3,
        hidden_size=[4],
        dims=1,
        rank=0,
        transient=False,
        name="sPIKAN",
    )
    f1 = mod.FlaxFunction(V1, "u", rngs=rngs)
    f2 = mod.FlaxFunction(V2, "v", rngs=rngs)
    comp = mod.Comp(f1, f2)
    x = jnp.zeros((5, V1.in_size))  # both rank 0 => scalar each
    y = comp(x)
    assert y.shape == (5, 2)
    assert comp.dim == f1.module.dim + f2.module.dim
