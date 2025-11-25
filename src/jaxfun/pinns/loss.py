from collections.abc import Callable
from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxfun.coordinates import BaseScalar, get_system
from jaxfun.typing import Array, LSQR_Tuple
from jaxfun.utils import lambdify

from .module import Comp, Function


# Differs from jaxfun.utils.common.jacn in the last if else
def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    """Return vectorized k-th order Jacobian of a function.

    Repeatedly applies jacfwd/jacrev k times (producing nested Jacobians) and
    then vmaps over the leading batch axis if k > 0.

    Args:
        fun: Function mapping a scalar/array to an Array.
        k: Number of derivatives (k >= 0).

    Returns:
        Callable producing the k-th order Jacobian for batched inputs.
    """
    for i in range(k):
        fun = jax.jacfwd(fun) if (i % 2 == 0) else jax.jacrev(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun


def get_flaxfunction_args(a: sp.Expr) -> tuple[sp.Symbol | BaseScalar, ...]:
    for p in sp.core.traversal.iterargs(a):
        if getattr(p, "argument", -1) == 2:
            return p.args
    return None


def get_flaxfunctions(
    a: sp.Expr,
) -> set[Function]:
    flax_found = set()
    for p in sp.core.traversal.iterargs(a):
        if getattr(p, "argument", -1) == 2:
            flax_found.add(p)
    return flax_found


def evaluate(expr: sp.Expr, x: Array) -> Array:
    """Evaluate a sympy expression containing FlaxFunctions

    Args:
        expr: Sympy expression containing FlaxFunctions
        x: Points where to evaluate the expression (N, D)

    Returns:
        Array: The evaluated expression at points x
    """
    if len(get_flaxfunctions(expr)) == 0:
        raise ValueError("Expression does not contain any FlaxFunctions")
    if len(x.shape) == 1:
        x = x[:, None]

    r = Residual(expr, x)
    return r.evaluate(get_flaxfunctions(expr).pop().module)


def expand(forms: sp.Expr) -> list[sp.Expr]:
    """Expand, and collect all terms without basis functions

    Args:
        forms: Sympy expression

    Returns:
        A tuple, where first item is a sympy expression that does not contain
        flax functions and second item is a list of sp.Exprs containing flax
        functions, used as arguments to Add
    """
    f = sp.Add.make_args(forms.doit().expand())
    consts = []
    flaxs = []
    for fi in f:
        v = get_flaxfunctions(fi)
        if len(v) == 0:
            consts.append(fi)
        else:
            flaxs.append(fi)

    return sp.Add(*consts), flaxs


class Residual:
    """Residual of a single equation"""

    def __init__(
        self,
        f: sp.Expr,
        x: Array,
        target: Number | Array = 0,
        weights: Number | Array = 1,
    ) -> None:
        """Residual of a single equation evaluated at collocation points

        Args:
            f: Sympy expression representing the equation residual. The
                expression may contain one or several FlaxFunctions.
            x: Collocation points where the residual is evaluated.
            target: Target value of the residual.
            weights: Weights for the residual.
        """
        t, expr = expand(f)
        s = get_flaxfunction_args(f.doit())

        # Build list of equations and all required evaluations of flaxfunctions
        eqs = []
        keys = set()
        for h in expr:
            eqs.append(get_fn(h, s))
            v = get_flaxfunctions(h)
            if len(v) == 0:
                continue
            mod_id = hash(v.pop().module)
            for p in sp.core.traversal.iterargs(h):
                if (
                    isinstance(p, sp.Derivative)
                    and getattr(p.args[0], "argument", -1) == 2
                ):
                    keys.add((id(x), mod_id, p.derivative_count))
                if getattr(p, "argument", -1) == 2:
                    keys.add((id(x), mod_id, 0))

        self.eqs = tuple(eqs)
        self.keys = keys

        # Place all terms without flaxfunctions in the target,
        # because these will not need to be computed more than once
        assert isinstance(target, Number | Array)
        t0 = target
        if len(t.free_symbols) > 0:
            assert s is not None, "Could not find base scalars in expression"
            tx = lambdify(s, t)(*x.T)
            if tx.ndim == 1:
                tx = tx[:, None]
            t0 = target - tx
        else:
            assert isinstance(t, Number)
            t = float(t) if t.is_real else complex(t)
            t0 = target - t
        t0 = jnp.squeeze(t0)
        assert isinstance(weights, Number | Array)
        weights = jnp.array(weights)
        if weights.sharding != x.sharding and jax.local_device_count() > 1:
            if len(weights.shape) > 0 and weights.shape[0] == x.shape[0]:
                weights = jax.device_put(weights, x.sharding)
            else:
                weights = jax.device_put(weights, NamedSharding(x.sharding.mesh, P()))
        if t0.sharding != x.sharding and jax.local_device_count() > 1:
            if len(t0.shape) > 0 and t0.shape[0] == x.shape[0]:
                t0 = jax.device_put(t0, x.sharding)
            else:
                t0 = jax.device_put(t0, NamedSharding(x.sharding.mesh, P()))
        self.x, self.target, self.weights = x, t0, weights

    def __call__(
        self,
        x: Array,
        target: Array,
        module: nnx.Module,
        Js: dict[tuple[int, int, int], Array] = None,
        x_id: int = None,
    ) -> Array:
        return sum([eq(x, module, Js=Js, x_id=x_id) for eq in self.eqs]) - target

    def _compute_gradients(
        self, module: nnx.Module, x: Array
    ) -> dict[tuple[int, int, int], Array]:
        """Return (as a dictionary) all the required evaluations of module
        for all the residuals' equations"""
        Js = {}
        for key in self.keys:
            x_id, mod_id, k = key
            mod = (
                module.data[module.mod_index[str(mod_id)]]
                if isinstance(module, Comp)
                else module
            )
            Js[key] = jacn(mod, key[2])(x)
        return Js

    def eval_compute_grad(
        self, x: Array, target: Array, module: nnx.Module, x_id: int
    ) -> Array:
        Js = self._compute_gradients(module, x)
        return self(x, target, module, Js=Js, x_id=x_id)

    def evaluate(self, module: nnx.Module) -> Array:
        """Evaluate the residual at internal points

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The residuals at points self.x, self.target
        """
        return self(self.x, self.target, module)


class LSQR:
    """Least squares loss function"""

    def __init__(self, *fs: LSQR_Tuple):
        """The least squares method computes the loss over all input equations at
        all collocation points. The different equations are all defined with their
        own points.

        Args:
            fs:
                One or several tuples. The tuples contain the subproblems that
                are to be solved. The subproblems are defined by the equation
                residuals (first item) and the collocation points (second item)
                used to evaluate the residuals. The third item is the target,
                which is zero by default, whereas the last item is an optional
                weight. The weight needs to be a number or an array of the same
                shape as the collocation points.

        Note:
            The collocation points need to be arrays of shape (N, D), where N
            is the number of points and D is the number of dimensions. The
            collocation points need to be fully addressable.

        Examples:

            >>> import jax.numpy as jnp
            >>> from jaxfun.operators import Div, Grad
            >>> from jaxfun.pinns.loss import LSQR
            >>> from jaxfun.pinns.module import MLPSpace, FlaxFunction
            >>> V = MLPSpace([8, 8], dims=1, rank=0, name="V")
            >>> u = FlaxFunction(V, name="u")
            >>> eq = Div(Grad(u)) + 2
            >>> xj = jnp.linspace(-1, 1, 10)[:, None]
            >>> xb = jnp.array([[-1.0], [1.0]])
            >>> loss_fn = LSQR((eq, xj, 0, 1), (u, xb, 0, 10))
        """
        from jaxfun.operators import Dot

        self.residuals = []

        for f in fs:
            f0 = f[0].doit()
            f1 = f[1]
            if f0.is_Vector:  # Vector equation
                sys = get_system(f0)
                for i in range(sys.dims):
                    bt = sys.get_contravariant_basis_vector(i)
                    g = (Dot(f0, bt),) + (f1,)
                    if len(f) > 2:
                        if isinstance(f[2], Number):
                            g += (f[2],)
                        else:
                            g += (f[2][..., i],)
                    if len(f) > 3:
                        g += (f[3],)

                    self.residuals.append(Residual(*g))

            else:
                self.residuals.append(Residual(*((f[0], f1) + f[2:])))

        # Store the unique collocation points and their order for later use
        self.xs = {id(eq.x): eq.x for eq in self.residuals}
        x_keys = list(self.xs.keys())
        self.x_ids = tuple(x_keys.index(id(eq.x)) for eq in self.residuals)
        # use indices into xs (0, 1, ...) as keys instead of id(x)
        for i, eq in enumerate(self.residuals):
            eq.keys = set([(self.x_ids[i], mod_id, k) for (_, mod_id, k) in eq.keys])
            eq.x_id = self.x_ids[i]
        # Store all keys needed for gradient computations
        self.keys = set(key for i, eq in enumerate(self.residuals) for key in eq.keys)

    @property
    def args(self) -> tuple[tuple[Array], tuple[Array]]:
        targets = tuple(eq.target for eq in self.residuals)
        return tuple(self.xs.values()), targets

    @property
    def local_mesh(self) -> jax.sharding.Mesh | None:
        if jax.local_device_count() == 1:
            return None

        assert self.residuals[0].x.sharding.num_devices == jax.local_device_count()
        return self.residuals[0].x.sharding.mesh

    def compute_residual_arrays(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array]
    ) -> list[Array]:
        """Return the weighted residuals with given global weights

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The weighted residuals
        """
        Js = self._compute_gradients(module, xs)
        return [
            eq(xs[x_id], target, module, Js=Js, x_id=x_id)
            for i, (eq, target, x_id) in enumerate(
                zip(self.residuals, targets, self.x_ids, strict=True)
            )
        ]

    def JTJ(
        self, module: nnx.Module, gw: Array, xs: tuple[Array], targets: tuple[Array]
    ) -> Array:
        """Return the Gauss-Newton approximation to the Hessian

        For the loss L = sum_i gw[i] * mean( res_i^2 ), the Gauss-Newton
        approximation to the Hessian is given by

            H = 2 * sum_i gw[i] * J_i^T @ J_i / N_i

        where J_i is the Jacobian (J_i)_kj = ∂res_i(x^i_k) / ∂w_j (for weights w_j) of
        the residuals for equation i evaluated at all collocation points
        (x^i = (x^i_k)_{k=0}^{N_i-1}) for equation i, and N_i = |x^i| is the number
        of collocation points for equation i.

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations
            targets: Targets for all residuals"""
        res = jax.jacfwd(self.compute_residual_arrays, argnums=0)(module, xs, targets)
        JTJ = []
        for i, r in enumerate(res):
            jf = jax.flatten_util.ravel_pytree(r)[0]
            N = xs[self.x_ids[i]].shape[0]
            J = jf.reshape((N, -1))
            w = gw[i] * self.residuals[i].weights
            if len(w.shape) == 0:
                w = jnp.array([w])
            JTJ.append(((J * w[:, None]).T @ J) / N)
        return 2 * sum(JTJ)

    @partial(nnx.jit, static_argnums=0)
    def value_and_grad_and_JTJ(
        self, module: nnx.Module, gw: Array, xs: tuple[Array], targets: tuple[Array]
    ) -> tuple[Array, Array, Array]:
        """Return the loss value, gradient and Jacobian

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations
            targets: Targets for all residuals

        Returns:
            A tuple where the first item is the loss value and the second item
            is the gradient and the third is the Gauss-Newton approximation to the
            Hessian
        """
        res = self.compute_residual_arrays(module, xs, targets)
        dres = jax.jacfwd(self.compute_residual_arrays, argnums=0)(module, xs, targets)
        unravel = jax.flatten_util.ravel_pytree(nnx.state(module))[1]
        JTJ = []
        grads = []
        loss = 0
        for i, (r, dr) in enumerate(zip(res, dres, strict=True)):
            jf = jax.flatten_util.ravel_pytree(dr)[0]
            N = xs[self.x_ids[i]].shape[0]
            J = jf.reshape((N, -1))
            w = gw[i] * self.residuals[i].weights
            if len(w.shape) == 0:
                w = jnp.array([w])
            JTJ.append(((J * w[:, None]).T @ J) / N)
            grads.append(
                jax.flatten_util.ravel_pytree(
                    jax.tree.map(
                        lambda g, r=r, w=w: jax.lax.dot_general(
                            r * w, g, (((0,), (0,)), ((), ()))
                        ),
                        dr,
                    )
                )[0]
                / N
            )
            loss = loss + (r * w) @ r / N

        return loss, unravel(2 * sum(grads)), 2 * sum(JTJ)

    def compute_residual_i(self, module: nnx.Module, i: int) -> Array:
        """Return the residuals for equation i

        Args:
            module: The module (nnx.Module)
            i: The equation number

        Returns:
            Array: The residuals for equation i
        """
        xs, targets = self.args
        return self.residuals[i].eval_compute_grad(
            xs[self.x_ids[i]], targets[i], module, x_id=self.x_ids[i]
        )

    def compute_residuals(self, module: nnx.Module) -> Array:
        """Return the residuals for all equations

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The residuals for all equations (not including global weights)
        """
        xs, targets = self.args
        L2 = []
        Js = self._compute_gradients(module, xs)
        for res, target, x_id in zip(self.residuals, targets, self.x_ids, strict=True):
            L2.append(
                (
                    res.weights * res(xs[x_id], target, module, Js=Js, x_id=x_id) ** 2
                ).mean()
            )
        return jnp.array(L2)

    @partial(nnx.jit, static_argnums=(0, 4))
    def compute_Li(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array], i: int
    ) -> Array:
        """Return the loss for equation i

        Does not include the global weight.

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)
            i: The equation number

        Returns:
            Array: The loss for equation i
        """
        x = self.residuals[i].eval_compute_grad(
            xs[self.x_ids[i]], targets[i], module, x_id=self.x_ids[i]
        )
        return (self.residuals[i].weights * x**2).mean()

    @partial(nnx.jit, static_argnums=(0, 4))
    def norm_grad_loss_i(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array], i: int
    ) -> Array:
        """Return the norm of the gradient of the loss for equation i

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)
            i: The equation number

        Returns:
            Array: The norm of the gradient of the loss for equation i
        """
        return jnp.linalg.norm(
            jax.flatten_util.ravel_pytree(
                nnx.grad(self.compute_Li)(module, xs, targets, i)
            )[0]
        )

    @partial(nnx.jit, static_argnums=0)
    def norm_grad_loss(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array]
    ) -> Array:
        """Return the norms of the gradients of the losses for all equations

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The norms of the gradients of the losses for all equations
        """
        norms = []
        for i in range(len(self.residuals)):
            norms.append(self.norm_grad_loss_i(module, xs, targets, i))
        return jnp.array(norms)

    @partial(nnx.jit, static_argnums=0)
    def compute_global_weights(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array]
    ) -> Array:
        """Return global weights based on the norms of the gradients

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The global weights
        """
        norms = self.norm_grad_loss(module, xs, targets)
        return jnp.sum(norms) / jnp.where(norms < 1e-16, 1e-16, norms)

    @partial(nnx.jit, static_argnums=0)
    def update_global_weights(
        self,
        module: nnx.Module,
        gw: Array,
        alpha: float,
        xs: tuple[Array],
        targets: tuple[Array],
    ) -> Array:
        """Return updated global weights

        Args:
            module: The module (nnx.Module)
            gw: Current global weights
            alpha: Smoothing parameter (0 < alpha < 1)

        Returns:
            Array: The updated global weights
        """
        from jax.experimental import multihost_utils as mh

        new = self.compute_global_weights(module, xs, targets)
        if jax.process_count() > 1:
            new = mh.process_allgather(new).mean(0)
        return new * (1 - alpha) + gw * alpha

    def _compute_gradients(
        self, module: nnx.Module, xs: tuple[Array]
    ) -> dict[tuple[int, int, int], Array]:
        """Return (as a dictionary) all the required evaluations of module
        for the loss function"""
        Js = {}
        for key in self.keys:
            x_id, mod_id, k = key
            mod = (
                module.data[module.mod_index[str(mod_id)]]
                if isinstance(module, Comp)
                else module
            )
            Js[key] = jacn(mod, key[2])(xs[key[0]])
        return Js

    def loss_with_gw(
        self, module: nnx.Module, gw: Array, xs: tuple[Array], targets: tuple[Array]
    ) -> Array:
        """Return the weighted loss with given global weights

        Args:
            module: The module (nnx.Module)
            gw: Global weights
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The weighted loss
        """
        Js = self._compute_gradients(module, xs)
        return (
            jnp.array(
                [
                    (
                        eq.weights * eq(xs[x_id], target, module, Js=Js, x_id=x_id) ** 2
                    ).mean()
                    for i, (eq, target, x_id) in enumerate(
                        zip(self.residuals, targets, self.x_ids, strict=True)
                    )
                ]
            )
            @ gw
        )

    def __call__(self, module: nnx.Module) -> Array:
        """Return the total least squares loss

        Does not include global weights.

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The total (unweighted) least squares loss
        """
        xs, targets = self.args
        Js = self._compute_gradients(module, xs)
        return sum(
            [
                (
                    eq.weights * eq(xs[x_id], target, module, Js=Js, x_id=x_id) ** 2
                ).mean()
                for i, (eq, target, x_id) in enumerate(
                    zip(self.residuals, targets, self.x_ids, strict=True)
                )
            ]
        )


def get_fn(
    f: sp.Expr, s: tuple[BaseScalar, ...]
) -> Callable[[Array, nnx.Module, dict[tuple[int, int, int], Array], int], Array]:
    """Return Sympy Expr as function evaluated by points and gradients

    Args:
        f: Sympy expression that may contain FlaxFunction
        s: Tuple of base scalars used as variables in f

    Returns:
        Callable[[Array, nnx.Module, dict[tuple[int, int, int], Array], int], Array]
    """

    v = get_flaxfunctions(f)

    if len(v) == 0:
        # Coefficient independent of basis function
        if len(f.free_symbols) > 0:
            fun = lambdify(s, f)
            return lambda x, mod, Js=None, x_id=None, fun=fun: fun(*x.T)
        else:
            arr = jnp.array(float(f) if f.is_real else complex(f))
            return lambda x, mod, Js=None, x_id=None, arr=arr: arr

    if isinstance(f, sp.Mul):
        # Multiplication of terms that either contain the basis function or not
        f_flax = []
        f_const = []
        for fx in f.args:
            v0 = get_flaxfunctions(fx)
            # Collect terms that do not (f_const) and contain (f_flax) the flaxfunction
            if len(v0) == 0:
                f_const.append(fx)
                continue
            f_flax.append(fx)

        f_const = sp.Mul(*f_const)
        fun = get_fn(f_const, s)
        return lambda x, mod, Js=None, x_id=None, s=s, f=f_flax: fun(
            x, mod, Js=Js, x_id=x_id
        ) * jnp.prod(
            jnp.array([get_fn(fi, s)(x, mod, Js=Js, x_id=x_id) for fi in f]), axis=0
        )

    elif isinstance(f, sp.Pow):
        return (
            lambda x,
            mod,
            Js=None,
            x_id=None,
            s=s,
            f=f.args[0],
            p=int(f.args[1]): get_fn(f, s)(x, mod, Js=Js, x_id=x_id) ** p
        )

    assert len(v) == 1
    v = v.pop()
    return partial(
        _lookup_or_eval,
        mod_id=hash(v.module),
        global_index=v.global_index,
        k=int(getattr(f, "derivative_count", "0")),
        variables=getattr(f, "variables", ()),
    )


def _lookup_or_eval(
    x: Array,
    mod: nnx.Module,
    Js: dict[tuple[int, int, int], Array] = None,
    x_id: int = 0,
    mod_id: int = 0,
    global_index: int = 0,
    k: int = 1,
    variables: tuple[BaseScalar, ...] = (),
) -> Array:
    """Compute or look up k-th order Jacobian of module at points x"""
    if Js is None:
        Js = {}
    module = mod.data[mod.mod_index[str(mod_id)]] if isinstance(mod, Comp) else mod
    assert mod_id == hash(module)
    var: tuple[int] = tuple((slice(None), global_index)) + tuple(
        int(s._id[0]) for s in variables
    )
    key: tuple[int, int, int] = (x_id, mod_id, k)
    if key not in Js:
        # Compute gradient
        z = jacn(module, k)(x)
        return z[var]
    # look up gradient
    return Js[key][var]
