import jax.numpy as jnp
from flax import nnx
from flax.nnx import Initializer, Param
from jax import Array


class PeriodEmbs(nnx.Module):
    """Per-axis cosine/sine positional embeddings with optional trainable periods.

    For each selected axis i we map scalar coordinate x_i to:
        [cos(2π x_i / p_i), sin(2π x_i / p_i)]
    If an axis is not selected it is passed through unchanged.

    Periods can be fixed (stored as plain arrays) or trainable (stored as
    nnx.Param). This enables learning optimal fundamental frequencies.

    Args:
        period: Tuple of periods p_i (same length as axis / trainable).
        axis: Tuple of axis indices whose coordinates are embedded.
        trainable: Tuple of bool flags indicating if corresponding period
            is trainable.

    Attributes:
        axis: Stored axis indices (tuple).
        _periods: Dict mapping 'period_{ordinal}' -> value or nnx.Param.
                  Note: ordinal is the position inside the axis tuple, not
                  the absolute axis index.
    """

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
        self._periods = nnx.Dict(store)

    def __call__(self, x: Array) -> Array:
        """Apply per-axis periodic embeddings.

        Args:
            x: Input coordinate vector (1D) or batch-unaware array. Expected
                shape (D,) where D is at least max(axis)+1.

        Returns:
            Concatenated embedding vector consisting of:
              - Original coordinates for non-embedded axes
              - Cos/Sin pairs for embedded axes (doubling those dims)
        """
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
    """Gaussian random Fourier feature embeddings (RFF).

    Projects input x ∈ R^{in_dim} using a fixed (non-trainable) random
    Gaussian matrix W then applies cos/sin:
        z = [cos(x W), sin(x W)]  giving dimension embed_dim.

    Args:
        embed_scale: Std deviation of Gaussian initializer for W.
        embed_dim: Final embedding dimension (must be even).
        in_dim: Input feature dimension.
        rngs: RNG container (nnx.Rngs) for deterministic init.

    Attributes:
        kernel: Projection matrix (nnx.Param) of shape (in_dim, embed_dim/2).
        embed_dim: Total output embedding size.
    """

    def __init__(
        self, *, embed_scale: float, embed_dim: int, in_dim: int, rngs: nnx.Rngs
    ) -> None:
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even (cos & sin halves).")

        init: Initializer = nnx.initializers.normal(embed_scale)
        k: Array = init(rngs(), (in_dim, embed_dim // 2), float)
        self.embed_dim = embed_dim
        self.kernel: Param[Array] = nnx.Param(k)

    def __call__(self, x: Array) -> Array:
        """Apply random Fourier feature mapping.

        Args:
            x: Input array of shape (..., in_dim). Broadcasting over leading
                dimensions is supported.

        Returns:
            Array of shape (..., embed_dim) containing concatenated cos/sin
            embeddings.
        """
        kernel = self.kernel[...]
        proj = x @ kernel
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


class Embedding(nnx.Module):
    """Composite embedding applying PeriodEmbs then FourierEmbs (optional).

    Workflow:
        1. (Optional) PeriodEmbs expands selected axes with cos/sin.
        2. (Optional) FourierEmbs maps result to random Fourier features.

    Args:
        periodicity: Dict of kwargs for PeriodEmbs or None.
            Expected keys: period, axis, trainable.
        fourier_emb: Dict of kwargs for FourierEmbs or None.
            Expected keys: embed_scale, embed_dim, in_dim.
        rngs: RNG container for FourierEmbs initialization.

    Attributes:
        periodic: PeriodEmbs module or None.
        fourier: FourierEmbs module or None.
    """

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
        """Apply configured embeddings sequentially.

        Args:
            x: Input array; shape depends on embedding configuration:
                - For PeriodEmbs only: (D,)
                - For FourierEmbs: (..., in_dim) after periodic expansion.

        Returns:
            Embedded array with additional feature dimensions.
        """
        if self.periodic is not None:
            x = self.periodic(x)
        if self.fourier is not None:
            x = self.fourier(x)

        return x
