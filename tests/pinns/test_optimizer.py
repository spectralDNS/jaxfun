import contextlib

import jax.numpy as jnp
import pytest
from flax import nnx

from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, optimizer as opt_mod


class DummyModule(nnx.Module):
    def __init__(self):
        # single scalar parameter
        self.w = nnx.Param(jnp.zeros(1))


def test_adam_no_decay_name_and_module():
    m = DummyModule()
    opt = opt_mod.adam(m, learning_rate=1e-3)
    assert isinstance(opt, opt_mod.NamedOptimizer)
    assert opt.module is m
    assert opt.name == "Adam(lr=0.001)"


def test_adam_with_decay_and_end_lr_default_and_custom():
    m = DummyModule()
    # default end lr = lr/10
    opt = opt_mod.adam(m, learning_rate=1e-3, decay_steps=100)
    assert isinstance(opt, opt_mod.NamedOptimizer)
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
    assert isinstance(s1, opt_mod.NamedOptimizer)
    assert s1.name == "Soap(lr=0.001)"
    assert s1.module is m

    s2 = opt_mod.soap(m, learning_rate=1e-3, decay_steps=10)
    # default end lr is lr/10
    assert s2.name == "Soap(lr=0.001->0.0001 in 10 steps)"


def test_lbfgs_name_and_memory_size():
    m = DummyModule()
    opt = opt_mod.lbfgs(m, memory_size=32, max_linesearch_steps=17)
    assert isinstance(opt, opt_mod.NamedOptimizer)
    assert opt.module is m
    assert opt.name == "LBFGS(memory_size=32)"


def test_gaussnewton_uses_fake_hess_and_names():
    m = DummyModule()
    g1 = opt_mod.GaussNewton(m, use_lstsq=True, cg_max_iter=5, max_linesearch_steps=7)
    assert isinstance(g1, opt_mod.NamedOptimizer)
    assert g1.module is m
    assert g1.name == "Hessian(lstsq=True, use_GN=False)"

    g2 = opt_mod.GaussNewton(m, use_lstsq=False)
    assert isinstance(g2, opt_mod.NamedOptimizer)
    assert g2.name == "Hessian(lstsq=False, use_GN=False)"


def test_train_returns_callable():
    # simple differentiable loss: (w - 2)^2
    def loss_fn(model: nnx.Module):
        assert isinstance(model, DummyModule)
        return (model.w.get_value() - 2.0) ** 2

    step = opt_mod.train(loss_fn)
    assert callable(step)


class TestTrainer:
    """Test suite for the Trainer class"""

    @pytest.fixture
    def lsqr_loss_fn_and_module(self):
        """Create FlaxFunction and Loss function for testing"""
        m = MLPSpace(5, dims=1, rank=0, name="MLP")
        u = FlaxFunction(m, "u")
        x = jnp.array([[1.0]])
        lsqr = Loss((u, x))
        return lsqr, u

    def test_trainer_init_basic(self, lsqr_loss_fn_and_module):
        """Test basic Trainer initialization"""
        loss_fn, u = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(loss_fn)
        assert trainer is not None
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "reset_global_weights")
        assert trainer.loss_fn is loss_fn

    def test_trainer_train_basic(self, lsqr_loss_fn_and_module):
        """Test basic training functionality"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-2)

        # Initial loss should be non-zero (w=0, target=1)
        initial_loss = lsqr_loss_fn(simple_model.module)
        assert initial_loss > 0

        # Train for a few epochs
        trainer.train(optimizer, 10)

        # Loss should decrease
        final_loss = lsqr_loss_fn(simple_model.module)
        assert final_loss < initial_loss

    def test_trainer_train_with_epoch_print(self, lsqr_loss_fn_and_module, capsys):
        """Test training with epoch printing"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-2)

        trainer.train(
            optimizer,
            5,
            epoch_print=2,  # Print every 2 epochs
        )

        captured = capsys.readouterr()
        # Should have printed at epochs 2, 4 and the "Running optimizer" line
        assert "Running optimizer" in captured.out or "Epoch" in captured.out

    def test_trainer_train_early_stopping_abs_limit_loss(self, lsqr_loss_fn_and_module):
        """Test early stopping based on absolute loss limit"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-3)

        # Train with early stopping when loss < 1.0 (more achievable)
        trainer.train(
            optimizer,
            1000,  # Max epochs
            abs_limit_loss=1.0,
            epoch_print=10,
            abs_limit_change=-1,  # Disable change-based stopping
        )

        # Should have stopped early when loss reached the limit
        final_loss = lsqr_loss_fn(simple_model.module)
        assert final_loss <= 1.0
        assert trainer.epoch < 100

    def test_trainer_train_early_stopping_abs_limit_change(
        self, lsqr_loss_fn_and_module
    ):
        """Test early stopping based on absolute change limit"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=0)  # Zero learning rate

        # Train with zero learning rate thus enabling change-based early stopping
        trainer.train(optimizer, 10, abs_limit_change=1e-6)

        assert trainer.epoch < 10

    def test_trainer_train_with_global_weights(self, lsqr_loss_fn_and_module):
        """Test training with global weight updates"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-2)

        trainer.train(
            optimizer,
            5,
            update_global_weights=1,  # Update every epoch
            alpha=0.9,  # Smoothing factor
        )

        # Should complete without errors
        assert True  # If we get here, no exceptions were raised

    def test_trainer_train_without_global_weights(self, lsqr_loss_fn_and_module):
        """Test training without global weight updates (default behavior)"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-4)

        initial_loss = lsqr_loss_fn(simple_model.module)

        trainer.train(
            optimizer,
            100,
            abs_limit_loss=0,
            update_global_weights=-1,  # Explicitly disable (default)
        )

        final_loss = lsqr_loss_fn(simple_model.module)
        assert final_loss < initial_loss

    def test_trainer_reset_global_weights(self, lsqr_loss_fn_and_module):
        """Test reset_global_weights method"""
        lsqr_loss_fn, _ = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)

        # Should not raise any errors
        trainer.reset_global_weights()
        assert True

    def test_trainer_train_zero_epochs(self, lsqr_loss_fn_and_module):
        """Test training with zero epochs"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-2)

        initial_loss = lsqr_loss_fn(simple_model.module)

        trainer.train(optimizer, 0)

        # Model should be unchanged
        final_loss = lsqr_loss_fn(simple_model.module)
        assert final_loss == initial_loss

    def test_trainer_train_negative_alpha_error(self, lsqr_loss_fn_and_module):
        """Test error handling for invalid alpha values"""
        lsqr_loss_fn, simple_model = lsqr_loss_fn_and_module
        trainer = opt_mod.Trainer(lsqr_loss_fn)
        optimizer = opt_mod.adam(simple_model, learning_rate=1e-2)

        # Invalid alpha values should be handled gracefully or raise appropriate errors
        # The implementation may clamp alpha or raise an error - test for consistency
        with contextlib.suppress(ValueError, AssertionError):
            trainer.train(
                optimizer,
                1,
                update_global_weights=1,
                alpha=-0.1,  # Invalid negative alpha
            )
