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
from jaxfun.utils import jacn, lambdify

from .module import Comp, Function


def vjacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    """Return vectorized k-th order Jacobian (forward) of a function.

    Repeatedly applies jax.jacfwd k times (producing nested Jacobians) and
    then vmaps over the leading batch axis if k > 0.

    Args:
        fun: Function mapping a scalar/array to an Array.
        k: Order of repeated jacfwd application (k >= 0).

    Returns:
        Callable producing the k-th order Jacobian for batched inputs.
    """
    for _ in range(k):
        fun = jax.jacfwd(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun


def get_flaxfunction_args(a: sp.Expr) -> tuple[sp.Symbol | BaseScalar, ...]:
    for p in sp.core.traversal.iterargs(a):
        if getattr(p, "argument", -1) == 2:
            return p.args


def get_flaxfunctions(
    a: sp.Expr,
) -> set[Function]:
    flax_found = set()
    for p in sp.core.traversal.iterargs(a):
        if getattr(p, "argument", -1) == 2:
            flax_found.add(p)
    return flax_found


def eval_flaxfunction(expr, x: Array) -> Array:
    f = get_flaxfunctions(expr)
    assert len(f) == 1
    f = f.pop()
    du = jacn(f.module, expr.derivative_count)(x)
    var: tuple[int] = tuple((slice(None), slice(None))) + tuple(
        int(s._id[0]) for s in expr.variables
    )
    return du[var]


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
                expression may contain one or several FlaxFunctions
            x: Collocation points where the residual is evaluated
            target: Target value of the residual. Defaults to 0.
            weights: Weights for the residual. Defaults to 1.
        """
        t, expr = expand(f)
        s = get_flaxfunction_args(f.doit())
        self.eqs = [get_fn(h, s) for h in expr]
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
    ) -> Array:
        return sum([eq(x, module, Js=Js) for eq in self.eqs]) - target


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

    @property
    def args(self) -> tuple[tuple[Array], tuple[Array]]:
        targets = tuple(eq.target for eq in self.residuals)
        return tuple(self.xs.values()), targets

    @property
    def local_mesh(self):
        if jax.local_device_count() == 1:
            return None
        assert self.residuals[0].x.sharding.num_devices == jax.local_device_count()
        return self.residuals[0].x.sharding.mesh

    def compute_residual_i(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array], i: int
    ) -> Array:
        """Return the residuals for equation i

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)
            i: The equation number

        Returns:
            Array: The residuals for equation i
        """
        return self.residuals[i](xs[self.x_ids[i]], targets[i], module, Js={})

    def compute_residuals(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array]
    ) -> Array:
        """Return the residuals for all equations

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)

        Returns:
            Array: The residuals for all equations
        """
        self.Js = {}
        L2 = []
        for res, target, x_id in zip(self.residuals, targets, self.x_ids, strict=True):
            L2.append(
                (res.weights * res(xs[x_id], target, module, Js=self.Js) ** 2).mean()
            )
        return jnp.array(L2)

    def compute_Li(
        self, module: nnx.Module, xs: tuple[Array], targets: tuple[Array], i: int
    ) -> Array:
        """Return the loss for equation i

        Args:
            module: The module (nnx.Module)
            xs: All the collocation arrays used for all equations (not necessarily same
                length as self.residuals)
            targets: Targets for all residuals (same length as self.residuals)
            i: The equation number

        Returns:
            Array: The loss for equation i
        """
        x = self.compute_residual_i(module, xs, targets, i)
        return (self.residuals[i].weights * x**2).mean()

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

    def update_global_weights(
        self,
        module: nnx.Module,
        gw: Array,
        alpha: float,
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

        new = self.compute_global_weights(module, *self.args)
        if jax.process_count() > 1:
            new = mh.process_allgather(new).mean(0)
        return new * (1 - alpha) + gw * alpha

    @partial(nnx.jit, static_argnums=0)
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
        self.Js = {}
        return (
            jnp.array(
                [
                    (eq.weights * eq(xs[x_id], target, module, Js=self.Js) ** 2).mean()
                    for i, (eq, target, x_id) in enumerate(
                        zip(self.residuals, targets, self.x_ids, strict=True)
                    )
                ]
            )
            @ gw
        )

    def __call__(self, module: nnx.Module) -> Array:
        """Return the total least squares loss

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The total least squares loss
        """
        xs, targets = self.args
        self.Js = {}
        return sum(
            [
                (eq.weights * eq(xs[x_id], target, module, Js=self.Js) ** 2).mean()
                for i, (eq, target, x_id) in enumerate(
                    zip(self.residuals, targets, self.x_ids, strict=True)
                )
            ]
        )


def get_fn(f: sp.Expr, s: tuple[BaseScalar, ...]) -> Callable[[Array, dict], Array]:
    """Return Sympy Expr as function evaluated by points and gradients

    Args:
        f: Sympy expression that may contain FlaxFunction
        s: Tuple of base scalars used as variables in f

    Returns:
        Callable[[Array, dict], Array]
    """

    v = get_flaxfunctions(f)

    if len(v) == 0:
        # Coefficient independent of basis function
        if len(f.free_symbols) > 0:
            z = lambdify(s, f)
            return lambda x, mod, Js=None, z=z: z(*x.T)
        else:
            f0 = jnp.array(float(f) if f.is_real else complex(f))
            return lambda x, mod, Js=None, f=f0: f

    if isinstance(f, sp.Mul):
        # Multiplication of terms that either contain the basis function or not
        gi = []
        gc = []
        for bii in f.args:
            v0 = get_flaxfunctions(bii)
            # Collect terms that do not contain the basis function
            if len(v0) == 0:
                gc.append(bii)
                continue
            gi.append(bii)
        gc = sp.Mul(*gc)

        return lambda x, mod, Js=None, s=s, f=gc, f0=gi: get_fn(f, s)(
            x, mod, Js=Js
        ) * jnp.prod(jnp.array([get_fn(fi, s)(x, mod, Js=Js) for fi in f0]), axis=0)

    elif isinstance(f, sp.Pow):
        fi = f.args[0]
        p = int(f.args[1])
        return lambda x, mod, Js=None, s=s, f=fi, p=p: get_fn(f, s)(x, mod, Js=Js) ** p

    assert len(v) == 1
    v = v.pop()
    return partial(
        _gradient,
        mod_id=id(v.module),
        i=v.global_index,
        k=int(getattr(f, "derivative_count", "0")),
        variables=getattr(f, "variables", ()),
    )


def _gradient(
    x: Array,
    mod: nnx.Module,
    Js: dict[tuple[int, int, int], Array] = None,
    mod_id: int = 0,
    i: int = 0,
    k: int = 1,
    variables: tuple[int] = [0],
) -> Array:
    """Compute or look up k-th order Jacobian of module at points x"""
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    key: tuple[int, int, int] = (id(x), mod_id, k)
    if key not in Js:
        # jax.debug.print(f"Computing array for key {key}")
        module = mod.__getattribute__(str(mod_id)) if isinstance(mod, Comp) else mod
        z = vjacn(module, k)(x)
        Js[key] = z
        return z[var]
    # jax.debug.print(f"Using cached array for key {key}")
    return Js[key][var]
