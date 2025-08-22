import jax.numpy as jnp
from flax import nnx
from jax import Array


class PeriodEmbs(nnx.Module):
    """Per-axis cosine/sine embeddings with optionally trainable periods."""

    def __init__(
        self,
        *,
        period: tuple[float, ...],
        axis: tuple[int, ...],
        trainable: tuple[bool, ...],
    ) -> None:
        self.axis = tuple(axis)
        # Store trainable periods as nnx.Param, constants as plain arrays.
        store = {}
        for p, idx, is_trainable in zip(period, axis, trainable, strict=True):
            val = jnp.asarray(p)
            store[f"period_{idx}"] = nnx.Param(val) if is_trainable else val
        self._periods = store

    def __call__(self, x: Array) -> Array:
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                p = self._periods[f"period_{idx}"]
                # NOTE: Differs from original implementation, which used xi * p
                period_val = 2 * jnp.pi / p
                vals = [jnp.cos(xi * period_val), jnp.sin(xi * period_val)]
            else:
                vals = [xi]

            y += vals

        return jnp.hstack(y)


class FourierEmbs(nnx.Module):
    """Gaussian RFFs with fixed projection matrix."""

    def __init__(
        self, *, embed_scale: float, embed_dim: int, in_dim: int, rngs: nnx.Rngs
    ) -> None:
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even (cos & sin halves).")

        init = nnx.initializers.normal(embed_scale)
        k = init(rngs(), (in_dim, embed_dim // 2), float)
        self.embed_dim = embed_dim
        self.kernel = nnx.Param(k)

    def __call__(self, x: Array) -> Array:
        proj = x @ self.kernel
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


class Embedding(nnx.Module):
    """Optionally apply PeriodEmbs then FourierEmbs."""

    def __init__(
        self,
        *,
        periodicity: dict | None = None,
        fourier_emb: dict | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        self.periodic = PeriodEmbs(**periodicity) if periodicity else None
        self.fourier = FourierEmbs(**fourier_emb, rngs=rngs) if fourier_emb else None

    def __call__(self, x: Array) -> Array:
        if self.periodic is not None:
            x = self.periodic(x)
        if self.fourier is not None:
            x = self.fourier(x)

        return x
