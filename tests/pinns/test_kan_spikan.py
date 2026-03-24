from typing import Any, cast

import jax.numpy as jnp
import pytest
from flax import nnx

from jaxfun.coordinates import BaseTime, CartCoordSys, x
from jaxfun.pinns import module as mod, nnspaces as nns


def test_kanlayer_forward_shape_and_params():
    rngs = nnx.Rngs(0)
    system = CartCoordSys("N", (x,))
    layer = mod.KANLayer(
        in_features=5,
        out_features=3,
        spectral_size=4,
        system=system,
        rngs=rngs,
        hidden=True,
    )
    x_arr = jnp.linspace(-1.0, 1.0, 15).reshape(3, 5)
    y = layer(x_arr)
    assert y.shape == (3, 3)
    assert hasattr(layer, "kernel")
    assert layer.kernel[...].shape[-1] == 3


def test_kanmlpspace_basic_attributes():
    V = nns.KANMLPSpace(spectral_size=5, hidden_size=[8, 4], dims=2)
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


def test_spikanspace_hidden_size_1():
    V = nns.sPIKANSpace(spectral_size=6, hidden_size=1, dims=1, rank=0)
    assert V.dims == 1
    assert V.rank == 0
    assert V.out_size == 1
    assert V.in_size == 1
    assert V.act_fun(1) == 1


def test_kanmplspace_hidden_size_1():
    V = nns.KANMLPSpace(spectral_size=6, hidden_size=1, dims=1, rank=0)
    assert V.dims == 1
    assert V.rank == 0
    assert V.out_size == 1
    assert V.in_size == 1
    assert V.act_fun(1) == 1


def test_spikanspace_hidden_size_1_dim_2():
    with pytest.raises(ValueError):
        nns.sPIKANSpace(spectral_size=6, hidden_size=1, dims=2, rank=0)


def test_kanmlpspace_hidden_size_1_dim_2():
    with pytest.raises(ValueError):
        nns.KANMLPSpace(spectral_size=6, hidden_size=1, dims=2, rank=0)


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


def test_flaxfunction_with_kanmlpspace_hidden_list():
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
    assert isinstance(f.module, mod.KANMLPModule)
    assert isinstance(V.hidden_size, list | tuple)
    assert f.module.layer_in.kernel.shape == (
        V.in_size,
        V.spectral_size,
        V.hidden_size[0],
    )
    assert f.module.hidden[0].kernel.shape == (
        V.hidden_size[0],
        V.hidden_size[0],
    )
    layer_out = f.module.layer_out
    assert hasattr(layer_out, "kernel")
    assert cast(Any, layer_out).kernel.shape == (V.hidden_size[-1], 1)


def test_flaxfunction_with_kanmlpspace_transient():
    V = nns.KANMLPSpace(
        spectral_size=5,
        hidden_size=[6],
        dims=1,
        rank=0,
        transient=True,
        domains=[(-1, 1), (0, 1)],
    )
    f = mod.FlaxFunction(V, name="f")
    x = jnp.zeros((4, V.in_size))
    y = f(x)
    assert y.shape == (4,)
    assert isinstance(f.module, mod.KANMLPModule)
    assert isinstance(V.hidden_size, list | tuple)
    assert f.module.layer_in.kernel.shape == (
        V.in_size,
        V.spectral_size,
        V.hidden_size[0],
    )
    assert f.module.hidden[0].kernel.shape == (
        V.hidden_size[0],
        V.hidden_size[0],
    )
    layer_out = f.module.layer_out
    assert hasattr(layer_out, "kernel")
    assert cast(Any, layer_out).kernel.shape == (V.hidden_size[-1], 1)


def test_flaxfunction_with_kanmlpspace_hidden_int():
    rngs = nnx.Rngs(5)
    V = nns.KANMLPSpace(
        spectral_size=5,
        hidden_size=6,
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
    assert isinstance(f.module, mod.KANMLPModule)
    assert f.module.layer_in.kernel.shape == (
        V.in_size,
        V.spectral_size,
        V.hidden_size,
    )
    layer_out = f.module.layer_out
    assert hasattr(layer_out, "kernel")
    assert cast(Any, layer_out).kernel.shape == (V.hidden_size, 1)


def test_flaxfunction_with_spikanspace_hidden_list():
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
    assert isinstance(f.module, mod.sPIKANModule)
    assert isinstance(V.hidden_size, list | tuple)
    assert f.module.layer_in.kernel.shape == (
        V.in_size,
        V.spectral_size,
        V.hidden_size[0],
    )
    assert f.module.hidden[0].kernel.shape == (
        V.hidden_size[0],
        V.spectral_size,
        V.hidden_size[0],
    )
    layer_out = f.module.layer_out
    assert hasattr(layer_out, "kernel")
    assert cast(Any, layer_out).kernel.shape == (V.hidden_size[-1], V.spectral_size, 1)


def test_flaxfunction_with_spikanspace_hidden_int():
    rngs = nnx.Rngs(6)
    V = nns.sPIKANSpace(
        spectral_size=4,
        hidden_size=5,
        dims=2,
        rank=0,
        transient=False,
        name="sPIKAN",
    )
    f = mod.FlaxFunction(V, "phi", rngs=rngs)
    x = jnp.zeros((3, V.in_size))
    y = f(x)
    assert y.shape == (3,)
    assert isinstance(f.module, mod.sPIKANModule)
    assert f.module.layer_in.kernel.shape == (
        V.in_size,
        V.spectral_size,
        V.hidden_size,
    )
    layer_out = f.module.layer_out
    assert hasattr(layer_out, "kernel")
    assert cast(Any, layer_out).kernel.shape == (V.hidden_size, V.spectral_size, 1)


@pytest.mark.slow
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
