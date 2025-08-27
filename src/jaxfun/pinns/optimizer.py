from collections.abc import Callable

import jax
from flax import nnx

from jaxfun.utils.common import ulp

from .loss import LSQR


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
    model: nnx.Module,
    opt: nnx.Optimizer,
    num: int,
    name: str,
    epoch_print: int = 100,
    abs_limit_loss: float = ulp(1000),
    abs_limit_change: float = ulp(100),
    print_final_loss: bool = False,
    update_global_weights: int = -1,
    print_global_weights: bool = False,
) -> None:
    train_step = train(loss_fn)
    loss_old = 1.0
    for epoch in range(1, num + 1):
        loss = train_step(model, opt)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < abs_limit_loss or abs(loss - loss_old) < abs_limit_change:
            break
        loss_old = loss
        if update_global_weights > 0 and epoch % update_global_weights == 0:
            loss_fn.update_global_weights(model)
            if print_global_weights:
                print("Global weights", loss_fn.global_weights)
    if print_final_loss:
        print(f"Final loss for {name}: {loss}")
