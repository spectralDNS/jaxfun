import jax
from jax.flatten_util import ravel_pytree
from optax import OptState, Params, Updates
from optax._src import base, combine, linesearch as _linesearch, transform
from optax.tree_utils import tree_vdot

default_linesearch = _linesearch.scale_by_zoom_linesearch(max_linesearch_steps=15)


def hess(
    learning_rate: base.ScalarOrSchedule | None = None,
    use_lstsq: bool = False,
    use_GN: bool = False,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-5,
    linesearch: base.GradientTransformationExtraArgs
    | base.GradientTransformation
    | None = default_linesearch,
) -> base.GradientTransformationExtraArgs:
    r"""Hessian optimizer.

    Args:
        learning_rate: optional global scaling factor, either fixed or evolving
            along iterations with a scheduler, see
            :func:`optax.scale_by_learning_rate`. By default the learning rate
            is handled by a linesearch.
        use_lstsq: If True, use least-squares to solve the linear system. If
            False, use Conjugate Gradient (CG) method.
        use_GN: If True, use Gauss-Newton approximation of the Hessian. If False,
            use full Hessian.
        cg_max_iter: Maximum number of CG iterations
        cg_tol: Tolerance for CG solver
        linesearch: an instance of :class:`optax.GradientTransformationExtraArgs`
            such as :func:`optax.scale_by_zoom_linesearch` that computes a
            learning rate, a.k.a. stepsize, to satisfy some criterion such as a
            sufficient decrease of the objective by additional calls to the
            objective.

    Returns:
      A :class:`optax.GradientTransformationExtraArgs` object.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = transform.scale_by_learning_rate(learning_rate)
    if linesearch is None:
        linesearch: base.GradientTransformation = base.identity()
    return combine.chain(
        scale_by_hessian(
            use_lstsq=use_lstsq, use_GN=use_GN, cg_max_iter=cg_max_iter, cg_tol=cg_tol
        ),
        base_scaling,
        linesearch,
    )


def scale_by_hessian(
    use_lstsq: bool = False,
    use_GN: bool = False,
    cg_max_iter: int = 100,
    cg_tol: float = 1e-5,
) -> base.GradientTransformationExtraArgs:
    r"""Scales updates by inverse Hessian or approximated Hessian.

    Args:
        cg_max_iter: Maximum number of CG iterations
        cg_tol: Tolerance for CG solver

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    def update_fn(
        updates: Updates, state: OptState, params: Params | None = None, **extra_args
    ) -> tuple[Updates, OptState]:
        hvp = lambda v: jax.jvp(jax.grad(extra_args["value_fn"]), (params,), (v,))[1]  # noqa: E731, F821

        if use_lstsq:
            flat_weights, unravel = ravel_pytree(params)
            flat_updates = ravel_pytree(updates)[0]
            if use_GN:
                H = extra_args["GN_loss_fn"](params)
            else:
                ht = jax.hessian(extra_args["value_fn"])(params)
                hf = ravel_pytree(ht)[0]
                H = hf.reshape((flat_weights.shape[0], flat_weights.shape[0]))

            hess_grads = jax.numpy.linalg.lstsq(H, flat_updates)[0]
            hess_grads = unravel(hess_grads)

        else:
            hess_grads = jax.scipy.sparse.linalg.cg(
                hvp, updates, tol=cg_tol, maxiter=cg_max_iter
            )[0]

        # Check if Hessian leads to a descent direction. If not, then use updates
        descent = tree_vdot(hess_grads, updates)
        # g2 = tree_vdot(updates, updates)

        updates = jax.lax.cond(
            (descent > 0),
            lambda *_: hess_grads,
            # lambda *_: tree_map(lambda p, q: p-1.1*descent/g2*q, hess_grads, updates),
            lambda *_: updates,
        )

        del params
        return updates, state

    return base.GradientTransformationExtraArgs(base.init_empty_state, update_fn)
