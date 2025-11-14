import re
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from jax.experimental import multihost_utils as mh
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import PyTree
from optax._src import base as optax_base

from jaxfun.pinns import FlaxFunction
from jaxfun.pinns.distributed import process_allmean

from .loss import LSQR


class NamedOptimizer[M: nnx.Module](nnx.Optimizer[M]):
    """Wrapper optimizer that stores a human-readable name and module reference.

    Extends nnx.Optimizer solely to attach:
      * name: A descriptive string (e.g. 'Adam(lr=1e-3->1e-4 in 100 steps)')
      * module: The optimized module (mirrors internal reference)

    Args:
        module: The module whose parameters are being optimized.
        tx: The optax transformation (gradient transformation pipeline).
        wrt: Parameter filter (defaults to nnx.Param).
        name: Readable optimizer name.

    Attributes:
        name: Name string.
        module: Reference to the optimized module (same as internal module).
    """

    def __init__(
        self,
        module: M,
        tx: optax.GradientTransformation,
        wrt: nnx.filterlib.Filter = nnx.Param,
        *,
        name: str = "Optimizer",
    ) -> None:
        super().__init__(module, tx, wrt=wrt)
        self.name = name
        self.module = module


def _gradient_descent_based_optimizer(
    module: nnx.Module,
    learning_rate: float,
    end_learning_rate: float | None,
    decay_steps: int | None,
    opt_constructor: Callable[..., optax.GradientTransformation],
    **opt_kwargs,
) -> NamedOptimizer:
    """Factory for simple (optionally decayed) gradient-descent style optimizers.

    Builds a linear learning-rate schedule if decay_steps is provided; else
    uses a constant step size. Wraps the created transformation in NamedOptimizer.

    Args:
        module: Target module to optimize.
        learning_rate: Initial learning rate.
        end_learning_rate: Final learning rate (defaults to learning_rate/10 if None and
            decay enabled).
        decay_steps: Number of steps for linear decay; if None no decay is applied.
        opt_constructor: Callable returning an optax GradientTransformation given an LR
            (or schedule).
        opt_kwargs: Additional keyword arguments to pass to opt_constructor.

    Returns:
        A NamedOptimizer instance with a formatted name describing the LR behavior.
    """
    lr_str = f"lr={learning_rate}"
    if decay_steps is None:
        lr = learning_rate
    else:
        end_learning_rate = end_learning_rate or learning_rate / 10
        lr = optax.linear_schedule(learning_rate, end_learning_rate, decay_steps)
        lr_str += f"->{end_learning_rate} in {decay_steps} steps"
    opt = opt_constructor(lr, **opt_kwargs)
    base_name = opt_constructor.__name__.capitalize()
    name = f"{base_name}({lr_str})"
    opt = NamedOptimizer(module, opt, nnx.Param, name=name)
    return opt


def sgd(
    u: FlaxFunction | nnx.Module,
    learning_rate: float = 1e-3,
    end_learning_rate: float = None,
    decay_steps: int = None,
    *,
    momentum: float | None = None,
    nesterov: bool = False,
    accumulator_dtype: Any | None = None,
) -> NamedOptimizer:
    """Create an Adam optimizer (optionally with linear LR decay).

    Args:
        u: nnx.Module or FlaxFunction wrapping an nnx.Module.
        learning_rate: Initial learning rate.
        end_learning_rate: Final learning rate if decaying (defaults to lr / 10).
        decay_steps: Steps over which to linearly decay learning rate. If None, constant
            LR.

        momentum: Momentum factor (if None, vanilla SGD).
        nesterov: Whether to use Nesterov momentum.
        accumulator_dtype: Optional `dtype` for the momentum accumulator.

    Returns:
        NamedOptimizer wrapping an sgd transformation.
    """
    return _gradient_descent_based_optimizer(
        u.module if isinstance(u, FlaxFunction) else u,
        learning_rate,
        end_learning_rate,
        decay_steps,
        optax.sgd,
        momentum=momentum,
        nesterov=nesterov,
        accumulator_dtype=accumulator_dtype,
    )


def adam(
    u: FlaxFunction | nnx.Module,
    learning_rate: float = 1e-3,
    end_learning_rate: float = None,
    decay_steps: int = None,
    *,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
    nesterov: bool = False,
) -> NamedOptimizer:
    """Create an Adam optimizer (optionally with linear LR decay).

    Args:
        u: nnx.Module or FlaxFunction wrapping an nnx.Module.
        learning_rate: Initial learning rate.
        end_learning_rate: Final learning rate if decaying (defaults to lr / 10).
        decay_steps: Steps over which to linearly decay learning rate. If None, constant
            LR.

        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: A small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
        eps_root: A small constant applied to denominator inside the square root (as
            in RMSProp), to avoid dividing by zero when rescaling. This is needed for
            example when computing (meta-)gradients through Adam.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        NamedOptimizer wrapping an Adam transformation.
    """
    return _gradient_descent_based_optimizer(
        u.module if isinstance(u, FlaxFunction) else u,
        learning_rate,
        end_learning_rate,
        decay_steps,
        optax.adam,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        mu_dtype=mu_dtype,
        nesterov=nesterov,
    )


def soap(
    u: FlaxFunction | nnx.Module,
    learning_rate: float = 1e-3,
    end_learning_rate: float = None,
    decay_steps: int = None,
    *,
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> NamedOptimizer:
    """Create a SOAP optimizer (from soap_jax) with optional LR decay.

    Args:
        u: nnx.Module or FlaxFunction wrapping an nnx.Module.
        learning_rate: Initial learning rate.
        end_learning_rate: Final learning rate if decaying (defaults to lr / 10).
        decay_steps: Steps over which to linearly decay learning rate. If None, constant
            LR.

        b1: Adam's beta1 parameter.
        b2: Adam's beta2 parameter.
        shampoo_beta: If >= 0, use this beta for the preconditioner (`L` and `R` in
            paper, `GG` below) moving average instead of b2.
        eps: Adam's epsilon for numerical stability.
        weight_decay: Weight decay coefficient.
        precondition_frequency: How often to update the preconditioner.
        max_precond_dim: Maximum dimension of the preconditioner.
            Set to 10000 to exclude most common vocab sizes while including layers.
        precision: Precision to use.

    Returns:
        NamedOptimizer wrapping a SOAP transformation.

    Raises:
        ImportError: If soap_jax is not installed.
    """
    from soap_jax import soap

    return _gradient_descent_based_optimizer(
        u.module if isinstance(u, FlaxFunction) else u,
        learning_rate,
        end_learning_rate,
        decay_steps,
        soap,
        b1=b1,
        b2=b2,
        shampoo_beta=shampoo_beta,
        eps=eps,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        precision=precision,
    )


def lbfgs(
    u: FlaxFunction | nnx.Module,
    learning_rate: optax_base.ScalarOrSchedule | None = None,
    scale_init_precond: bool = True,
    memory_size: int = 10,
    linesearch: optax_base.GradientTransformationExtraArgs
    | optax_base.GradientTransformation
    | None = None,
    max_linesearch_steps: int = 20,
    initial_guess_strategy: str = "one",
) -> nnx.Optimizer:
    """Create an L-BFGS optimizer using optax's limited-memory implementation.

    Args:
        u: nnx.Module or FlaxFunction wrapping an nnx.Module.
        learning_rate: optional global scaling factor, either fixed or evolving
            along iterations with a scheduler. By default the learning rate is
            handled by a linesearch.
        memory_size: Number of past updates to retain (history length).
        scale_init_precond: whether to use a scaled identity as the initial
            preconditioner.
        linesearch: Optional line search transformation to use. If None, a zoom
            linesearch is used.

        max_linesearch_steps: Maximum zoom line search iterations. Only used if
            linesearch is None
        initial_guess_strategy: Strategy for initial step size guess in line search.
            Only used if linesearch is None. Either 'one' or  'keep'.


    Returns:
        NamedOptimizer configured for L-BFGS.
    """
    linesearch = linesearch or optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
    )
    opt = optax.lbfgs(
        learning_rate=learning_rate,
        memory_size=memory_size,
        scale_init_precond=scale_init_precond,
        linesearch=linesearch,
    )
    opt = NamedOptimizer(
        u.module if isinstance(u, FlaxFunction) else u,
        opt,
        name=f"LBFGS(memory_size={memory_size})",
    )
    return opt


def GaussNewton(
    u: FlaxFunction | nnx.Module,
    use_lstsq: bool = True,
    cg_max_iter: int = 1000,
    linesearch: optax_base.GradientTransformationExtraArgs
    | optax_base.GradientTransformation
    | None = None,
    max_linesearch_steps: int = 25,
    max_learning_rate: float = 1.0,
) -> nnx.Optimizer:
    """Create a Gauss-Newton / Hessian-based optimizer.

    Delegates to a custom hess() builder (pseudoinverse or CG-based) from
    jaxfun.pinns.hessoptimizer.

    Args:
        u: nnx.Module or FlaxFunction wrapping an nnx.Module.
        use_lstsq: If True solve normal equations via least squares; else conjugate
            gradient.
        cg_max_iter: Maximum CG iterations (ignored if use_lstsq is True).
        linesearch: Optional line search transformation to use. If None, a zoom
            linesearch is used.
        max_linesearch_steps: Maximum steps for the zoom line search. Only used if
            linesearch is None.
        max_learning_rate: Maximum learning rate for line search. Only used if
            linesearch is None.

    Returns:
        NamedOptimizer with Hessian / Gauss-Newton updates.
    """
    from jaxfun.pinns.hessoptimizer import hess

    linesearch = linesearch or optax.scale_by_zoom_linesearch(
        max_linesearch_steps, max_learning_rate=max_learning_rate
    )

    opt = hess(
        use_lstsq=use_lstsq,
        cg_max_iter=cg_max_iter,
        linesearch=linesearch,
    )

    opt = NamedOptimizer(
        u.module if isinstance(u, FlaxFunction) else u,
        opt,
        name=f"Hessian(lstsq={use_lstsq})",
    )
    return opt


def train(
    loss_fn: LSQR, allreduce_gradients_and_loss: bool = False
) -> Callable[[nnx.Module, nnx.Optimizer], float]:
    """Build a JIT-compiled training step for a loss function.

    Args:
        loss_fn: Callable loss object/function (LSQR instance) accepting a module.
        allreduce_gradients_and_loss: If True, allreduce gradients and loss across
            processes each epoch.

    Returns:
        A function (module, optimizer) -> loss suitable for epoch loops.
    """

    @nnx.jit
    def train_step(
        module: nnx.Module,
        optimizer: nnx.Optimizer,
        gw: Array,
        xs: tuple[Array],
        targets: tuple[Array],
    ) -> float:
        def value_fn(m: nnx.Module) -> Array:
            return loss_fn.loss_with_gw(m, gw, xs, targets)

        loss, gradients = nnx.value_and_grad(value_fn)(module)
        gd, state = nnx.split(module, nnx.Param)
        unravel = jax.flatten_util.ravel_pytree(state)[1]
        value_fn_state = lambda state: value_fn(nnx.merge(gd, state))
        H_loss_fn = lambda flat_weights: value_fn(nnx.merge(gd, unravel(flat_weights)))

        optimizer.update(
            module,
            gradients,
            grad=gradients,
            value_fn=value_fn_state,
            value=loss,
            H_loss_fn=H_loss_fn,
        )

        return loss

    @nnx.jit
    def value_and_grad(
        module: nnx.Module, gw: Array, xs: tuple[Array], targets: tuple[Array]
    ) -> tuple[Array, PyTree]:
        def value_fn(m: nnx.Module) -> Array:
            return loss_fn.loss_with_gw(m, gw, xs, targets)

        return nnx.value_and_grad(value_fn)(module)

    @nnx.jit
    def update(
        loss: Array,
        gradients: PyTree,
        module: nnx.Module,
        optimizer: nnx.Optimizer,
        gw: Array,
        xs: tuple[Array],
        targets: tuple[Array],
    ) -> None:
        def value_fn(m: nnx.Module) -> Array:
            return loss_fn.loss_with_gw(m, gw, xs, targets)

        gd, state = nnx.split(module, nnx.Param)
        unravel = jax.flatten_util.ravel_pytree(state)[1]
        value_fn_state = lambda state: value_fn(nnx.merge(gd, state))
        H_loss_fn = lambda flat_weights: value_fn(nnx.merge(gd, unravel(flat_weights)))

        optimizer.update(
            module,
            gradients,
            grad=gradients,
            value_fn=value_fn_state,
            value=loss,
            H_loss_fn=H_loss_fn,
        )

    def allreduce(loss: Array, gradients: PyTree) -> None:
        reduced_gradients = process_allmean(gradients)
        reduced_loss = mh.process_allgather(loss).mean(axis=0)
        return reduced_loss, reduced_gradients

    def train_step_distributed(
        module: nnx.Module,
        optimizer: nnx.Optimizer,
        gw: Array,
        xs: tuple[Array],
        targets: tuple[Array],
    ) -> float:
        loss, gradients = value_and_grad(module, gw, xs, targets)
        loss, gradients = allreduce(loss, gradients)
        update(loss, gradients, module, optimizer, gw, xs, targets)
        return loss

    if allreduce_gradients_and_loss:
        return train_step_distributed

    return train_step


class Trainer:
    def __init__(self, loss_fn: LSQR) -> None:
        """Trainer for optimization loops.

        Args:
            loss_fn: Loss function / object (LSQR instance) to optimize.
        """
        assert isinstance(loss_fn, LSQR), "Trainer requires an LSQR loss function"
        self.loss_fn = loss_fn
        self.global_weights = jnp.ones(len(self.loss_fn.residuals), dtype=float)
        if jax.local_device_count() > 1:
            self.global_weights = jax.device_put(
                self.global_weights,
                NamedSharding(loss_fn.local_mesh, P()),
            )
        self.epoch = 0

    def reset_global_weights(self) -> None:
        self.global_weights = jnp.ones(len(self.loss_fn.residuals), dtype=float)
        if jax.local_device_count() > 1:
            self.global_weights = jax.device_put(
                self.global_weights,
                NamedSharding(self.loss_fn.local_mesh, P()),
            )

    def allreduce(self, module: nnx.Module) -> None:
        """Allreduce (average) all parameters across processes"""
        state = nnx.state(module)
        averaged_state = process_allmean(state)
        nnx.update(module, averaged_state)

    def train(
        self,
        opt: nnx.Optimizer,
        num: int,
        *,
        epoch_print: int = 100,
        name: str = None,
        module: nnx.Module = None,
        abs_limit_loss: float = 0,
        abs_limit_change: float = 0,
        print_final_loss: bool = False,
        update_global_weights: int = -1,
        print_global_weights: bool = False,
        allreduce_grads_and_loss: bool = False,
        allreduce_module_freq: int = 1,
        alpha: float = 0.9,
    ) -> None:
        """Execute an optimization loop with early stopping & optional callbacks.

        Early stopping criteria:
          * abs(loss) < abs_limit_loss
          * abs(loss - previous_loss) < abs_limit_change

        Optional periodic callback:
          * update_global_weights: if > 0, every N epochs calls

        Args:
            opt: Optimizer instance (NamedOptimizer or custom nnx.Optimizer). If not a
                NamedOptimizer and 'module' is None, a ValueError is raised.
            num: Maximum number of epochs.

            name: Optional name override for logging (defaults to opt.name or
                'Optimizer').
            module: Optional explicit module reference (required if opt is not a
                NamedOptimizer).
            abs_limit_loss: Absolute loss threshold for early stop.
            abs_limit_change: Absolute change threshold between epochs.
            print_final_loss: If True, prints final loss on completion.
            update_global_weights: Period (in epochs) for calling
                loss_fn.update_global_weights; -1 disables.
            print_global_weights: If True, prints global weights after each
                update_global_weights trigger.
            allreduce_grads_and_loss: If True, allreduce gradients and loss
                across processes each epoch.
            allreduce_module_freq: Period (in epochs) for allreducing module
                parameters; < 1 disables.
            alpha: Smoothing factor for global weights update (0 < alpha < 1).
                returns new * (1 - alpha) + old * alpha

        Raises:
            ValueError: If module is None and opt is not a NamedOptimizer.
            ValueError: If alpha is not in the range [0, 1).

        """
        rank = jax.process_index()

        if module is None:
            if not isinstance(opt, NamedOptimizer):
                raise ValueError(
                    "Module must be provided if opt is not a NamedOptimizer"
                )
            module = opt.module

        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be in the range (0, 1)")

        longname = name if name is not None else getattr(opt, "name", "Optimizer")
        if rank == 0:
            print(f"Running optimizer {longname}")

        train_step = train(
            self.loss_fn, allreduce_grads_and_loss and jax.process_count() > 1
        )

        name = re.sub(r"\([^)]*\)", "", longname)  # Remove parentheses content
        allow_early_break = abs_limit_loss > 0 or abs_limit_change > 0
        loss_old = 1.0
        xs, targets = self.loss_fn.args  # Extract points and targets for tracing
        self.losses = []
        for epoch in range(1, num + 1):
            loss = train_step(module, opt, self.global_weights, xs, targets)

            if epoch % epoch_print == 0 and rank == 0:
                print(f"Epoch {epoch} {name}, loss: {loss}")

            if allow_early_break:  # noqa: SIM102
                if loss < abs_limit_loss or abs(loss - loss_old) < abs_limit_change:
                    break

            loss_old = loss
            if update_global_weights > 0 and epoch % update_global_weights == 0:
                self.global_weights = self.loss_fn.update_global_weights(
                    module, self.global_weights, alpha, xs, targets
                )
                if print_global_weights and rank == 0:
                    print("Global weights", self.global_weights)

            if allreduce_module_freq > 0 and epoch % allreduce_module_freq == 0:
                self.allreduce(module)

            self.epoch = epoch
            self.losses.append(loss)

        if print_final_loss and rank == 0:
            print(f"Final loss for {longname}: {loss}")
