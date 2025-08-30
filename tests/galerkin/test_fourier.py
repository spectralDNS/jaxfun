import jax.numpy as jnp

from jaxfun.galerkin import Fourier
from jaxfun.utils.common import ulp


def test_fourier_truncation_and_padding():
    F = Fourier.Fourier(8)
    c = jnp.zeros(8, dtype=complex).at[1].set(1 + 0j).at[2].set(2 + 0j)
    u = F.backward(c)
    # Truncate to smaller N
    ut = F.backward(c, N=4)
    assert ut.shape[0] == 4
    # Pad to larger N
    up = F.backward(c, N=12)
    assert up.shape[0] == 12
    # Forward/backward consistency (energy)
    cf = F.forward(u)
    assert jnp.linalg.norm(cf[:8] - c) < ulp(100)
