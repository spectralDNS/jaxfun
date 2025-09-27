import re
from collections.abc import Callable

import jax
import optax
from flax import nnx

from jaxfun.utils.common import ulp

from .loss import LSQR


class NamedOptimizer[M: nnx.Module](nnx.Optimizer[M]):
    """A temporary subclass of nnx.Optimizer to allow adding attributes."""

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
