import re
from collections.abc import Callable

import jax
import optax
from flax import nnx

from jaxfun.utils.common import ulp

from .loss import LSQR


class NamedOptimizer[M: nnx.Module](nnx.Optimizer[M]):
    """Wrapper optimizer that stores a human-readable name and module reference.

    Extends nnx.Optimizer solely to attach:
      * name: A descriptive string (e.g. 'Adam(lr=1e-3->1e-4 in 100 steps)')
      * module: The optimized module (mirrors internal reference)

    Args:
        model: The module whose parameters are being optimized.
        tx: The optax transformation (gradient transformation pipeline).
        wrt: Parameter filter (defaults to nnx.Param).
        name: Readable optimizer name.

    Attributes:
        name: Name string.
        module: Reference to the optimized module (same as internal model).
    """

    def __init__(
        self,
        model: M,
        tx: optax.GradientTransformation,
        wrt: nnx.filterlib.Filter = nnx.Param,
        *,
        name: str = "Optimizer",
    ) -> None:
        super().__init__(model, tx, wrt)
        self.name = name
        self.module = model


def _gradient_descent_based_optimizer(
    module: nnx.Module,
    learning_rate: float,
    end_learning_rate: float | None,
    decay_steps: int | None,
    opt_constructor: Callable[..., optax.GradientTransformation],
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
    opt = opt_constructor(lr)

    base_name = opt_constructor.__name__.capitalize()
    name = f"{base_name}({lr_str})"
    opt = NamedOptimizer(module, opt, name=name)
    return opt


def adam(
    module: nnx.Module,
    learning_rate: float = 1e-3,
    end_learning_rate: float = None,
    decay_steps: int = None,
) -> NamedOptimizer:
    """Create an Adam optimizer (optionally with linear LR decay).

    Args:
        module: Module to optimize.
        learning_rate: Initial learning rate.
        end_learning_rate: Final learning rate if decaying (defaults to lr / 10).
        decay_steps: Steps over which to linearly decay learning rate. If None, constant
            LR.

    Returns:
        NamedOptimizer wrapping an Adam transformation.
    """
    return _gradient_descent_based_optimizer(
        module,
        learning_rate,
        end_learning_rate,
        decay_steps,
        optax.adam,
    )


def soap(
    module: nnx.Module,
    learning_rate: float = 1e-3,
    end_learning_rate: float = None,
    decay_steps: int = None,
) -> NamedOptimizer:
    """Create a SOAP optimizer (from soap_jax) with optional LR decay.

    Args:
        module: Module to optimize.
        learning_rate: Initial learning rate.
        end_learning_rate: Final learning rate if decaying (defaults to lr / 10).
        decay_steps: Steps over which to linearly decay learning rate. If None, constant
            LR.

    Returns:
        NamedOptimizer wrapping a SOAP transformation.

    Raises:
        ImportError: If soap_jax is not installed.
    """
    from soap_jax import soap

    return _gradient_descent_based_optimizer(
        module,
        learning_rate,
        end_learning_rate,
        decay_steps,
        soap,
    )


def lbfgs(
    module: nnx.Module, memory_size: int = 20, max_linesearch_steps: int = 25
) -> nnx.Optimizer:
    """Create an L-BFGS optimizer using optax's limited-memory implementation.

    Args:
        module: Module to optimize.
        memory_size: Number of past updates to retain (history length).
        max_linesearch_steps: Maximum zoom line search iterations.

    Returns:
        NamedOptimizer configured for L-BFGS.
    """
    opt = optax.lbfgs(
        memory_size=memory_size,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps, max_learning_rate=1.0
        ),
    )
    opt = NamedOptimizer(module, opt, name=f"LBFGS(memory_size={memory_size})")
    return opt


def GaussNewton(
    module: nnx.Module,
    use_lstsq: bool = True,
    cg_max_iter: int = 1000,
    max_linesearch_steps: int = 25,
) -> nnx.Optimizer:
    """Create a Gauss-Newton / Hessian-based optimizer.

    Delegates to a custom hess() builder (pseudoinverse or CG-based) from
    jaxfun.pinns.hessoptimizer.

    Args:
        module: Module to optimize.
        use_lstsq: If True solve normal equations via least squares; else conjugate
            gradient.
        cg_max_iter: Maximum CG iterations (ignored if use_lstsq is True).
        max_linesearch_steps: Maximum steps for the zoom line search.

    Returns:
        NamedOptimizer with Hessian / Gauss-Newton updates.
    """
    from jaxfun.pinns.hessoptimizer import hess

    opt = hess(
        use_lstsq=use_lstsq,
        cg_max_iter=cg_max_iter,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps, max_learning_rate=1.0
        ),
    )
    opt = NamedOptimizer(module, opt, name=f"Hessian(lstsq={use_lstsq})")
    return opt


def train(loss_fn: LSQR) -> Callable[[nnx.Module, nnx.Optimizer], float]:
    """Build a single JIT-compiled training step for a loss function.

    The produced function performs:
      1. Split model into differentiable parameters and auxiliary state.
      2. Compute loss and gradients (value_and_grad).
      3. Provide closures for value_fn and Hessian approximation callbacks.
      4. Update optimizer state.
      5. Return the scalar loss.

    Args:
        loss_fn: Callable loss object/function (LSQR instance) accepting a module.

    Returns:
        A function (model, optimizer) -> loss suitable for epoch loops.
    """

    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer) -> float:
        gd, state = nnx.split(model, nnx.Param)
        unravel = jax.flatten_util.ravel_pytree(state)[1]
        loss, gradients = nnx.value_and_grad(loss_fn)(model)
        loss_fn_split = lambda state: loss_fn(nnx.merge(gd, state))
        H_loss_fn = lambda flat_weights: loss_fn(nnx.merge(gd, unravel(flat_weights)))
        optimizer.update(
            gradients,
            grad=gradients,
            value_fn=loss_fn_split,
            value=loss,
            H_loss_fn=H_loss_fn,
        )
        return loss

    return train_step


def run_optimizer(
    loss_fn: LSQR,
    opt: nnx.Optimizer,
    num: int,
    epoch_print: int = 100,
    module: nnx.Module = None,
    name: str = None,
    abs_limit_loss: float = ulp(1000),
    abs_limit_change: float = ulp(100),
    print_final_loss: bool = False,
    update_global_weights: int = -1,
    print_global_weights: bool = False,
) -> None:
    """Execute an optimization loop with early stopping & optional callbacks.

    Early stopping criteria:
      * abs(loss) < abs_limit_loss
      * abs(loss - previous_loss) < abs_limit_change

    Optional periodic callback:
      * update_global_weights: if > 0, every N epochs calls
        loss_fn.update_global_weights(module).

    Args:
        loss_fn: Loss function / object supporting __call__(module) and optionally
            update_global_weights(module).
        opt: Optimizer instance (NamedOptimizer or custom nnx.Optimizer). If not a
            NamedOptimizer and 'module' is None, a ValueError is raised.
        num: Maximum number of epochs.
        epoch_print: Print frequency (epochs). Set high to reduce logging.
        module: Explicit module reference (required if opt is not a NamedOptimizer).
        name: Override printed optimizer name (defaults to opt.name if present).
        abs_limit_loss: Absolute loss threshold for early stop.
        abs_limit_change: Absolute change threshold between epochs.
        print_final_loss: If True, prints final loss on completion.
        update_global_weights: Period (in epochs) for calling
            loss_fn.update_global_weights; -1 disables.
        print_global_weights: If True, prints global weights after each
            update_global_weights trigger.

    Raises:
        ValueError: If module is None and opt is not a NamedOptimizer.

    Returns:
        None. (Per-epoch loss printing side effects only.)
    """
    longname = name if name is not None else getattr(opt, "name", "Optimizer")
    if module is None:
        if not isinstance(opt, NamedOptimizer):
            raise ValueError("Module must be provided if opt is not a NamedOptimizer")
        module = opt.module

    print(f"Running optimizer {longname}")
    name = re.sub(r"\([^)]*\)", "", longname)  # Remove parentheses content
    train_step = train(loss_fn)
    loss_old = 1.0
    for epoch in range(1, num + 1):
        loss = train_step(module, opt)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < abs_limit_loss or abs(loss - loss_old) < abs_limit_change:
            break
        loss_old = loss
        if update_global_weights > 0 and epoch % update_global_weights == 0:
            loss_fn.update_global_weights(module)
            if print_global_weights:
                print("Global weights", loss_fn.global_weights)
    if print_final_loss:
        print(f"Final loss for {longname}: {loss}")
