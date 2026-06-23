"""Tests for CartesianModule: sub-module dispatch, output shapes, and Loss integration.

Covers both spectral (CartesianTensorProductSpace) and NN (CartesianNNSpace) paths,
with VectorTensorProductSpace placed at different positions in the Cartesian product.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from jaxfun.galerkin import CartesianProduct, FunctionSpace, Legendre, TensorProduct
from jaxfun.operators import Div
from jaxfun.pinns import FlaxFunction, Loss
from jaxfun.pinns.module import CartesianModule, SpectralModule
from jaxfun.pinns.nnspaces import CartesianNNSpace, MLPSpace

pytestmark = pytest.mark.pinn

N_PTS = 6


# ---------------------------------------------------------------------------
# Spectral fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spectral_spaces():
    """Small 2-D spectral spaces for Stokes-like coupled problems."""
    D0 = FunctionSpace(4, Legendre.Legendre, name="D0")
    D1 = FunctionSpace(4, Legendre.Legendre, name="D1")
    P0 = FunctionSpace(3, Legendre.Legendre, name="P0")
    T0 = TensorProduct(D0, D1, name="T0")
    T1 = TensorProduct(D0, D0, name="T1")
    Q = TensorProduct(P0, P0, name="Q")
    return T0, T1, Q


@pytest.fixture(scope="module")
def spectral_velocity_first(spectral_spaces):
    """W = CartesianProduct(V, Q): VectorTPS is the first component."""
    T0, T1, Q = spectral_spaces
    V = CartesianProduct(T0, T1, name="V", rank=1)
    W = CartesianProduct(V, Q, name="up")
    up = FlaxFunction(W, "up", rngs=nnx.Rngs(10))
    u, p = up
    return up, u, p


@pytest.fixture(scope="module")
def spectral_pressure_first(spectral_spaces):
    """W = CartesianProduct(Q, V): scalar TPS first, VectorTPS second."""
    T0, T1, Q = spectral_spaces
    V = CartesianProduct(T0, T1, name="V", rank=1)
    W = CartesianProduct(Q, V, name="pu")
    pu = FlaxFunction(W, "pu", rngs=nnx.Rngs(20))
    p, u = pu
    return pu, p, u


# ---------------------------------------------------------------------------
# NN fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nn_velocity_first():
    """CartesianNNSpace(u, p): vector NN first, scalar NN second."""
    u_space = MLPSpace([8], dims=2, rank=1, name="u")
    p_space = MLPSpace([8], dims=2, rank=0, name="p")
    W = CartesianNNSpace(u_space, p_space, name="up")
    up = FlaxFunction(W, "up", rngs=nnx.Rngs(30))
    u, p = up
    return up, u, p


@pytest.fixture(scope="module")
def nn_pressure_first():
    """CartesianNNSpace(p, u): scalar NN first, vector NN second."""
    p_space = MLPSpace([8], dims=2, rank=0, name="p")
    u_space = MLPSpace([8], dims=2, rank=1, name="u")
    W = CartesianNNSpace(p_space, u_space, name="pu")
    pu = FlaxFunction(W, "pu", rngs=nnx.Rngs(40))
    p, u = pu
    return pu, p, u


# ---------------------------------------------------------------------------
# Spectral — module structure
# ---------------------------------------------------------------------------


def test_spectral_vf_creates_cartesian_module(spectral_velocity_first):
    pu, u, p = spectral_velocity_first
    assert isinstance(pu.module, CartesianModule)


def test_spectral_vf_sub_module_identity(spectral_velocity_first):
    pu, u, p = spectral_velocity_first
    assert u.module is pu.module.data[0]
    assert p.module is pu.module.data[1]


def test_spectral_pf_creates_cartesian_module(spectral_pressure_first):
    pu, p, u = spectral_pressure_first
    assert isinstance(pu.module, CartesianModule)


def test_spectral_pf_sub_module_identity(spectral_pressure_first):
    pu, p, u = spectral_pressure_first
    assert p.module is pu.module.data[0]
    assert u.module is pu.module.data[1]


# ---------------------------------------------------------------------------
# Spectral — output shapes
# ---------------------------------------------------------------------------


def test_spectral_vf_output_shapes(spectral_velocity_first):
    pu, u, p = spectral_velocity_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    assert pu(pts).shape == (N_PTS, 3)
    assert u(pts).shape == (N_PTS, 2)
    assert p(pin).shape == (1,)


def test_spectral_pf_output_shapes(spectral_pressure_first):
    """With pressure first the combined output is still (N, 3)."""
    pu, p, u = spectral_pressure_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    assert pu(pts).shape == (N_PTS, 3)
    assert p(pin).shape == (1,)
    assert u(pts).shape == (N_PTS, 2)


# ---------------------------------------------------------------------------
# Spectral — global_offset when VectorTPS is NOT first
# ---------------------------------------------------------------------------


def test_spectral_pf_global_offset(spectral_pressure_first):
    """Q (scalar, 1 col) is at index 0 → u's offset must be 1."""
    pu, p, u = spectral_pressure_first
    assert p.global_offset == 0
    assert u.global_offset == 1


def test_spectral_vf_global_offset(spectral_velocity_first):
    """V (vector, 2 cols) is at index 0 → p's offset must be 2."""
    pu, u, p = spectral_velocity_first
    assert u.global_offset == 0
    assert p.global_offset == 2


# ---------------------------------------------------------------------------
# Standalone VectorTPS must produce SpectralModule, not CartesianModule
# ---------------------------------------------------------------------------


def test_standalone_vector_tps_produces_spectral_module(spectral_spaces):
    """get_flax_module(VectorTPS) must NOT hit the CartesianTPS branch."""
    T0, T1, _ = spectral_spaces
    V = CartesianProduct(T0, T1, name="V", rank=1)
    u = FlaxFunction(V, "u", rngs=nnx.Rngs(0))
    assert isinstance(u.module, SpectralModule)


def test_standalone_vector_tps_output_shape(spectral_spaces):
    T0, T1, _ = spectral_spaces
    V = CartesianProduct(T0, T1, name="V", rank=1)
    u = FlaxFunction(V, "u", rngs=nnx.Rngs(1))
    pts = jnp.zeros((N_PTS, 2))
    assert u(pts).shape == (N_PTS, 2)


# ---------------------------------------------------------------------------
# Spectral — Loss dispatch
# ---------------------------------------------------------------------------


def test_spectral_vf_loss_evaluates(spectral_velocity_first):
    """Loss with divergence constraint + pressure pin (velocity first ordering)."""
    pu, u, p = spectral_velocity_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    loss = Loss((Div(u), pts), (p, pin, 0, 10))
    val = loss(pu.module)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_spectral_pf_loss_evaluates(spectral_pressure_first):
    """Loss with divergence constraint + pressure pin (pressure first ordering)."""
    pu, p, u = spectral_pressure_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    loss = Loss((Div(u), pts), (p, pin, 0, 10))
    val = loss(pu.module)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_spectral_pf_pin_only_dispatches_pressure_sub_module(spectral_pressure_first):
    """The pressure pin residual must resolve to the pressure sub-module."""
    pu, p, u = spectral_pressure_first
    pin = jnp.zeros((1, 2))

    loss = Loss((p, pin, 0, 10))
    # Only the p sub-module is involved; this should not differentiate u's params.
    val = loss(pu.module)
    assert jnp.isfinite(val)


# ---------------------------------------------------------------------------
# NN — module structure and output shapes
# ---------------------------------------------------------------------------


def test_nn_vf_creates_cartesian_module(nn_velocity_first):
    pu, u, p = nn_velocity_first
    assert isinstance(pu.module, CartesianModule)


def test_nn_vf_sub_module_identity(nn_velocity_first):
    pu, u, p = nn_velocity_first
    assert u.module is pu.module.data[0]
    assert p.module is pu.module.data[1]


def test_nn_vf_output_shapes(nn_velocity_first):
    pu, u, p = nn_velocity_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    assert pu(pts).shape == (N_PTS, 3)
    assert u(pts).shape == (N_PTS, 2)
    assert p(pin).shape == (1,)


def test_nn_pf_creates_cartesian_module(nn_pressure_first):
    pu, p, u = nn_pressure_first
    assert isinstance(pu.module, CartesianModule)


def test_nn_pf_sub_module_identity(nn_pressure_first):
    pu, p, u = nn_pressure_first
    assert p.module is pu.module.data[0]
    assert u.module is pu.module.data[1]


def test_nn_pf_output_shapes(nn_pressure_first):
    """Pressure first: p gets 1 column, u gets 2 → combined (N, 3)."""
    pu, p, u = nn_pressure_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    assert pu(pts).shape == (N_PTS, 3)
    assert p(pin).shape == (1,)
    assert u(pts).shape == (N_PTS, 2)


def test_nn_pf_global_offset(nn_pressure_first):
    """Scalar p (1 output col) is first → u.global_offset must be 1."""
    pu, p, u = nn_pressure_first
    assert p.global_offset == 0
    assert u.global_offset == 1


# ---------------------------------------------------------------------------
# NN — Loss dispatch
# ---------------------------------------------------------------------------


def test_nn_vf_loss_evaluates(nn_velocity_first):
    pu, u, p = nn_velocity_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    loss = Loss((Div(u), pts), (p, pin, 0, 10))
    val = loss(pu.module)
    assert val.shape == ()
    assert jnp.isfinite(val)


def test_nn_pf_loss_evaluates(nn_pressure_first):
    """Loss with pressure first ordering (non-zero global_offset for u)."""
    pu, p, u = nn_pressure_first
    pts = jnp.zeros((N_PTS, 2))
    pin = jnp.zeros((1, 2))

    loss = Loss((Div(u), pts), (p, pin, 0, 10))
    val = loss(pu.module)
    assert val.shape == ()
    assert jnp.isfinite(val)
