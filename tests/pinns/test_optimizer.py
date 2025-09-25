import jax.numpy as jnp
import pytest
from flax import nnx

from jaxfun.pinns import optimizer as opt_mod


class DummyModule(nnx.Module):
    def __init__(self):
        # single scalar parameter
        self.w = nnx.Param(jnp.array(0.0))


def test_adam_no_decay_name_and_module():
    m = DummyModule()
    opt = opt_mod.adam(m, learning_rate=1e-3)
    assert isinstance(opt, nnx.Optimizer)
    assert opt.module is m
    assert opt.name == "Adam(lr=0.001)"


def test_adam_with_decay_and_end_lr_default_and_custom():
    m = DummyModule()
    # default end lr = lr/10
    opt = opt_mod.adam(m, learning_rate=1e-3, decay_steps=100)
    assert isinstance(opt, nnx.Optimizer)
    assert opt.name == "Adam(lr=0.001->0.0001 in 100 steps)"

    # custom end lr
    opt2 = opt_mod.adam(m, learning_rate=1e-3, end_learning_rate=5e-5, decay_steps=50)
    assert opt2.name == "Adam(lr=0.001->5e-05 in 50 steps)"


@pytest.mark.skipif(
    pytest.importorskip("soap_jax", reason="soap_jax not installed") is None,
    reason="soap_jax not installed",
)
def test_soap_no_decay_and_with_decay():
    m = DummyModule()
    s1 = opt_mod.soap(m, learning_rate=1e-3)
    assert isinstance(s1, nnx.Optimizer)
    assert s1.name == "Soap(lr=0.001)"
    assert s1.module is m

    s2 = opt_mod.soap(m, learning_rate=1e-3, decay_steps=10)
    # default end lr is lr/10
    assert s2.name == "Soap(lr=0.001->0.0001 in 10 steps)"


def test_lbfgs_name_and_memory_size():
    m = DummyModule()
    opt = opt_mod.lbfgs(m, memory_size=32, max_linesearch_steps=17)
    assert isinstance(opt, nnx.Optimizer)
    assert opt.module is m
    assert opt.name == "LBFGS(memory_size=32)"


def test_gaussnewton_uses_fake_hess_and_names():
    m = DummyModule()
    g1 = opt_mod.GaussNewton(m, use_lstsq=True, cg_max_iter=5, max_linesearch_steps=7)
    assert isinstance(g1, nnx.Optimizer)
    assert g1.module is m
    assert g1.name == "Hessian(lstsq=True)"

    g2 = opt_mod.GaussNewton(m, use_lstsq=False)
    assert g2.name == "Hessian(lstsq=False)"


def test_train_returns_callable():
    # simple differentiable loss: (w - 2)^2
    def loss_fn(model: nnx.Module):
        return (model.w - 2.0) ** 2

    step = opt_mod.train(loss_fn)
    assert callable(step)


def test_run_optimizer_requires_module_if_not_nnx_optimizer():
    class NotAnOptimizer:
        pass

    dummy_opt = NotAnOptimizer()
    # Should raise because opt is not nnx.Optimizer and has no module attr
    with pytest.raises(ValueError, match="Module must be provided"):
        opt_mod.run_optimizer(
            loss_fn=lambda m: jnp.array(0.0),
            opt=dummy_opt,  # type: ignore[arg-type]
            num=1,
        )


def test_run_optimizer_early_stop_on_abs_limit_loss(monkeypatch, capsys):
    # Stub train to return a function that yields a tiny loss once
    calls = {"n": 0}

    def fake_train(loss_fn):
        def step(model, optimizer):
            calls["n"] += 1
            return jnp.array(0.0)

        return step

    monkeypatch.setattr(opt_mod, "train", fake_train)

    m = DummyModule()
    opt = opt_mod.adam(m, learning_rate=1e-3)
    opt_mod.run_optimizer(
        loss_fn=lambda m: jnp.array(0.0),
        opt=opt,
        num=10,
        abs_limit_loss=1e-6,
        print_final_loss=True,
    )
    # Should have stopped after first epoch
    assert calls["n"] == 1


def test_run_optimizer_calls_update_global_weights(monkeypatch):
    # Make a train stub that returns decreasing losses to avoid "small change" early stop
    seq = {"vals": [10.0, 9.0, 8.0], "i": 0}

    def fake_train(loss_fn):
        def step(model, optimizer):
            i = seq["i"]
            val = seq["vals"][i if i < len(seq["vals"]) else -1]
            seq["i"] = min(i + 1, len(seq["vals"]) - 1)
            return jnp.array(val)

        return step

    monkeypatch.setattr(opt_mod, "train", fake_train)

    class LossWithUpdate:
        def __init__(self):
            self.updated = 0

        def __call__(self, model):
            # never used by fake_train
            return jnp.array(0.0)

        def update_global_weights(self, module):
            self.updated += 1

    loss_obj = LossWithUpdate()
    m = DummyModule()
    opt = opt_mod.adam(m, learning_rate=1e-3)

    # Trigger update at epoch 2
    opt_mod.run_optimizer(
        loss_fn=loss_obj,
        opt=opt,
        num=3,
        update_global_weights=2,
        abs_limit_loss=0.0,  # don't early stop by absolute loss
        abs_limit_change=1e-12,  # small threshold; our loss changes by 1.0
    )
    assert loss_obj.updated >= 1
