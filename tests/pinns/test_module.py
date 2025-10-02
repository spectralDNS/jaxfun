import jax.numpy as jnp
from flax import nnx

from jaxfun.galerkin import Legendre, TensorProduct
from jaxfun.pinns.module import (
    MLP,
    Comp,
    FlaxFunction,
    PIModifiedBottleneck,
    PirateNet,
    RWFLinear,
    SpectralModule,
)
from jaxfun.pinns.nnspaces import MLPSpace, PirateSpace


def test_rwflinear_forward_shapes_and_bias():
    rngs = nnx.Rngs(0)
    lin = RWFLinear(in_features=3, out_features=5, use_bias=True, rngs=rngs)
    x = jnp.ones((7, 3))
    y = lin(x)
    assert y.shape == (7, 5)
    assert lin.use_bias is True
    assert lin.bias is not None

    lin_nb = RWFLinear(in_features=3, out_features=2, use_bias=False, rngs=nnx.Rngs(1))
    y2 = lin_nb(x)
    assert y2.shape == (7, 2)
    assert lin_nb.bias is None


def test_mlp_forward_and_dim_with_mlpspace():
    rngs = nnx.Rngs(0)
    V = MLPSpace(
        hidden_size=[8, 4],
        dims=2,
        rank=0,
        transient=False,
        act_fun=nnx.tanh,
        name="MLP",
    )
    mlp = MLP(V, rngs=rngs)

    x = jnp.zeros((10, V.in_size))
    y = mlp(x)

    assert y.shape == (10, V.out_size)
    assert mlp.dim > 0


def test_pimodifiedbottleneck_alpha_zero_returns_identity():
    rngs = nnx.Rngs(0)
    pib = PIModifiedBottleneck(
        in_dim=4,
        hidden_dim=6,
        output_dim=4,
        nonlinearity=0.0,
        rngs=rngs,
        act_fun=nnx.tanh,
    )
    x = jnp.linspace(-1.0, 1.0, 12).reshape(3, 4)
    # u and v with compatible shapes
    u = jnp.tanh(jnp.linspace(-0.5, 0.5, 18)).reshape(3, 6)
    v = -u
    out = pib(x, u, v)
    # alpha=0 => out == identity (input x)
    assert out.shape == x.shape
    assert jnp.allclose(out, x)


def test_piratenet_forward_shape_with_piratespace():
    rngs = nnx.Rngs(0)
    V = PirateSpace(
        hidden_size=[4],
        dims=2,
        rank=0,
        transient=False,
        act_fun=nnx.tanh,
        act_fun_hidden=nnx.tanh,
        nonlinearity=0.1,
        periodicity=None,
        fourier_emb=None,
        name="PirateNet",
    )
    net = PirateNet(V, rngs=rngs)
    x = jnp.zeros((6, V.in_size))
    y = net(x)
    assert y.shape == (6, V.out_size)
    assert net.dim > 0


def test_spectralmodule_1d_forward_and_dim_with_legendre():
    rngs = nnx.Rngs(0)
    V1 = Legendre.Legendre(6)  # 1D Legendre basis
    sm = SpectralModule(V1, rngs=rngs)
    x = jnp.array([[0.0], [0.3], [0.7]])
    y = sm(x)
    assert y.shape == x.shape  # 1D branch returns (N,)
    assert sm.dim == V1.dim


def test_spectralmodule_2d_forward_and_dim_with_tensor_product():
    rngs = nnx.Rngs(1)
    Vx = Legendre.Legendre(3)
    Vy = Legendre.Legendre(4)
    V2 = TensorProduct(Vx, Vy)  # 2D space
    sm = SpectralModule(V2, rngs=rngs)
    x = jnp.array([[0.0, 0.0], [0.1, 0.2], [0.7, 0.3]])
    y = sm(x)
    assert y.shape == (3, 1)  # 2D branch returns (N, 1)
    assert sm.dim == V2.dim


def test_flaxfunction_builds_mlp_and_call():
    rngs = nnx.Rngs(0)
    V = MLPSpace(
        hidden_size=[4], dims=2, rank=0, transient=False, act_fun=nnx.tanh, name="MLP"
    )
    f = FlaxFunction(V, "u", rngs=rngs)

    x = jnp.zeros((3, V.in_size))
    y_mod = f.module(x)
    y_call = f(x)  # rank 0 -> returns y[:, 0]

    assert y_mod.shape == (3, V.out_size)
    assert y_call.shape == (3,)
    assert isinstance(f.module, MLP)


def test_flaxfunction_builds_spectral_and_call_legendre():
    rngs = nnx.Rngs(1)
    V1 = Legendre.Legendre(4)
    f = FlaxFunction(V1, "phi", rngs=rngs)

    x = jnp.linspace(0.0, 1.0, 6).reshape(-1, 1)
    y_mod = f.module(x)
    y_call = f(x)

    assert y_mod.shape == (x.shape[0], 1)
    assert y_call.shape == (x.shape[0],)
    assert isinstance(f.module, SpectralModule)


def test_comp_stacks_multiple_flaxfunctions():
    rngs = nnx.Rngs(0)
    V1 = MLPSpace(hidden_size=[4], dims=2, rank=0, name="MLP")
    f1 = FlaxFunction(V1, "u", rngs=rngs)

    V2 = Legendre.Legendre(3)
    f2 = FlaxFunction(V2, "v", rngs=rngs)

    comp = Comp(f1, f2)
    x = jnp.zeros((5, V1.in_size))  # both take same x arity in tests
    y = comp(x)

    # f1.module(x) -> (N, out1) = (N,1)
    # f2.module(x) -> (N,) => expanded by Comp stacking
    assert y.shape[0] == 5
    assert y.ndim == 2
    assert comp.dim == f1.module.dim + f2.module.dim
