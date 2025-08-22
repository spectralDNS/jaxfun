import jax.numpy as jnp
import pytest
from flax import nnx

from jaxfun.pinns.embeddings import Embedding, FourierEmbs, PeriodEmbs


def test_period_embs_values_and_shape_nontrainable():
    # Embed only the first axis -> output is [cos(p*x0), sin(p*x0), x1]
    period = 2.0
    pe = PeriodEmbs(period=(period,), axis=(0,), trainable=(False,))
    x = jnp.array([0.5, 1.0])
    out = pe(x)

    val = x[0] * 2 * jnp.pi / period
    expected = jnp.array([jnp.cos(val), jnp.sin(val), x[1]])
    assert out.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-7)

    # Period is stored as plain array when not trainable
    assert not isinstance(pe._periods["period_0"], nnx.Param)


def test_period_embs_trainable_param_and_call():
    period = 3.0
    pe = PeriodEmbs(period=(period,), axis=(0,), trainable=(True,))
    # Period is stored as nnx.Param when trainable
    p = pe._periods["period_0"]
    assert isinstance(p, nnx.Param)

    # Forward pass works with trainable period
    x = jnp.array([0.25, -2.0])
    out = pe(x)
    val = x[0] * 2 * jnp.pi / period
    expected = jnp.array([jnp.cos(val), jnp.sin(val), x[1]])

    assert out.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-7)


def test_period_embs_is_periodic():
    period = 2.0
    pe = PeriodEmbs(period=(period,), axis=(0,), trainable=(False,))

    x1 = jnp.array([0.5, 1.0])
    x2 = jnp.array([0.5 + period, 1.0])
    x3 = jnp.array([0.5 - period, 1.0])

    out1 = pe(x1)
    out2 = pe(x2)
    out3 = pe(x3)

    assert jnp.allclose(out1, out2, atol=1e-6), "Outputs should be periodic"
    assert jnp.allclose(out1, out3, atol=1e-6), "Outputs should be periodic"


def test_fourier_embs_shape_and_batch():
    rngs = nnx.Rngs(0)
    fe = FourierEmbs(embed_scale=1.0, embed_dim=4, in_dim=2, rngs=rngs)

    x = jnp.array([0.1, -0.2])
    out = fe(x)
    assert out.shape == (4,)

    X = jnp.stack([x, x * 2.0, x * -1.5], axis=0)  # (3, 2)
    out_b = fe(X)
    assert out_b.shape == (3, 4)


def test_fourier_embs_odd_embed_dim_raises():
    rngs = nnx.Rngs(0)
    with pytest.raises(ValueError):
        _ = FourierEmbs(embed_scale=1.0, embed_dim=3, in_dim=2, rngs=rngs)


def test_embedding_only_periodic_matches_period_embs():
    rngs = nnx.Rngs(0)
    periodicity = dict(period=(2.0,), axis=(0,), trainable=(False,))
    emb = Embedding(periodicity=periodicity, fourier_emb=None, rngs=rngs)

    x = jnp.array([0.5, 1.0])
    out = emb(x)

    pe = PeriodEmbs(**periodicity)
    expected = pe(x)
    assert out.shape == expected.shape == (3,)
    assert jnp.allclose(out, expected, atol=1e-7)


def test_embedding_only_fourier_matches_fourier_embs_shape():
    rngs = nnx.Rngs(0)
    fe_conf = dict(embed_scale=1.0, embed_dim=6, in_dim=2)
    emb = Embedding(periodicity=None, fourier_emb=fe_conf, rngs=rngs)

    x = jnp.array([0.3, -0.7])
    out = emb(x)

    fe = FourierEmbs(rngs=rngs, **fe_conf)
    out_fe = fe(x)

    assert out.shape == (6,)
    assert out_fe.shape == (6,)


def test_embedding_periodic_then_fourier_pipeline_shape():
    rngs = nnx.Rngs(0)
    # After periodic with axis=(0,), 2D input becomes 3D -> set in_dim=3 for Fourier
    periodicity = dict(period=(2.0,), axis=(0,), trainable=(False,))
    fe_conf = dict(embed_scale=1.0, embed_dim=8, in_dim=3)

    emb = Embedding(periodicity=periodicity, fourier_emb=fe_conf, rngs=rngs)
    x = jnp.array([0.5, 1.0])
    out = emb(x)

    assert out.shape == (8,)
    assert jnp.all(jnp.isfinite(out))
