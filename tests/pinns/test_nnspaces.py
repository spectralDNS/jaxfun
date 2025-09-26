import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.coordinates import BaseScalar, BaseTime
from jaxfun.pinns.nnspaces import MLPSpace, MLPVectorSpace, NNSpace, PirateSpace


def test_nnspace_basic_attributes_and_out_size():
    ns = NNSpace(dims=3, rank=2, transient=False, name="testNN")
    # in_size should equal dims when not transient
    assert ns.in_size == 3
    # out_size computed as dims ** rank
    assert ns.out_size == 3**2
    assert ns.dims == 3
    assert ns.rank == 2
    assert ns.is_transient is False
    # base_variables should contain spatial BaseScalar objects only
    bv = ns.base_variables()
    assert isinstance(bv, sp.Tuple)
    assert len(bv) == ns.dims
    assert all(isinstance(v, BaseScalar) for v in bv)


def test_nnspace_transient_includes_time_variable():
    ns = NNSpace(dims=2, rank=0, transient=True, name="testNNTime")
    # in_size should be dims + 1 when transient
    assert ns.in_size == 3
    bv = ns.base_variables()
    assert len(bv) == 3
    # last entry should be a BaseTime instance
    assert isinstance(bv[-1], BaseTime)


def test_mlpspace_sets_hidden_and_activation_callable():
    def my_act(x):
        return jnp.tanh(x) * 2.0

    mlp = MLPSpace(
        hidden_size=[8, 4], dims=2, rank=0, transient=False, act_fun=my_act, name="mlp"
    )
    assert mlp.hidden_size == [8, 4]
    # act_fun should be the same callable object passed in
    assert mlp.act_fun is my_act
    # inherits properties from NNSpace
    assert mlp.dims == 2
    assert mlp.out_size == 2**0


def test_mlpvectorspace_partial_sets_rank_one():
    vec = MLPVectorSpace(hidden_size=5, dims=2, transient=False, name="vecspace")
    # partial should have applied rank=1
    assert vec.rank == 1
    # hidden_size should be stored as given
    assert vec.hidden_size == 5


def test_piratespace_hidden_size_conversion_and_params():
    # integer hidden_size should be converted to list
    p1 = PirateSpace(hidden_size=7, dims=1, rank=0)
    assert isinstance(p1.hidden_size, list)
    assert p1.hidden_size == [7]

    # list or tuple should be preserved
    p2 = PirateSpace(hidden_size=(3, 5), dims=1, rank=0)
    assert list(p2.hidden_size) == [3, 5]

    # test PirateSpace specific parameters are stored
    pi_arr = jnp.array([1.0, 2.0])
    periodic = {"period": (2.0,)}
    fourier = {"embed_dim": 4}
    p3 = PirateSpace(
        hidden_size=[4],
        dims=1,
        rank=0,
        name="pirate",
        nonlinearity=0.42,
        periodicity=periodic,
        fourier_emb=fourier,
        pi_init=pi_arr,
    )
    assert pytest.approx(p3.nonlinearity) == 0.42
    assert p3.periodicity is periodic
    assert p3.fourier_emb is fourier
