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
        self, Js: dict[tuple[int, int, int], Array], x: Array, target: Array
    ) -> Array:
        return sum([eq(x, Js) for eq in self.eqs]) - target


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
        res = []

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
                    res.append(g[0])

            else:
                self.residuals.append(Residual(*((f[0], f1) + f[2:])))
                res.append(f[0])

        # Collection of modules and derivative counts for each residual:
        self.Jres = Jres = [set() for _ in range(len(res))]
        for i, f in enumerate(res):
            f0 = f.doit()
            for s in sp.core.traversal.preorder_traversal(f0):
                # look for either derivatives of flax functions or flax functions
                mod = s.args[0] if isinstance(s, sp.Derivative) else s
                if hasattr(mod, "module"):
                    mod = mod.module
                    ki = int(getattr(s, "derivative_count", "0"))
                    key = (id(mod), ki)
                    Jres[i].add(key)

    @property
    def args(self) -> tuple[tuple[Array, Array], ...]:
        return tuple((eq.x, eq.target) for eq in self.residuals)

    @property
    def local_mesh(self):
        if jax.local_device_count() == 1:
            return None
        assert self.residuals[0].x.sharding.num_devices == jax.local_device_count()
        return self.residuals[0].x.sharding.mesh

    @partial(nnx.jit, static_argnums=0)
    def update_arrays(
        self, module: nnx.Module, Jsi: dict[tuple[int, int, int], Array]
    ) -> dict[tuple[int, int, int], Array]:
        """Return all required derivatives of the module evaluated at the
        collocation points.

        Args:
            module: The module (nnx.Module)
            Jsi: dictionary mapping (coll points id, module id, derivative order) to
            collocation points

        Returns:
            dict[tuple[int, int, int], Array]: dictionary mapping
                (coll points id, module id, derivative order) to evaluated derivatives
        """
        Js = {}
        for k, xs in Jsi.items():
            mod = (
                module.__getattribute__(str(k[1]))
                if isinstance(module, Comp)
                else module
            )
            Js[k] = jacn(mod, k[2])(xs)

        return Js

    def compute_residual_i(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...], i: int
    ) -> Array:
        """Return the residuals for equation i

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations
            i: The equation number

        Returns:
            Array: The residuals for equation i
        """
        Jsi = {(id(args[i][0]),) + k: args[i][0] for k in self.Jres[i]}
        Js = self.update_arrays(module, Jsi)
        return self.residuals[i](Js, args[i][0], args[i][1])

    def compute_residuals(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        """Return the residuals for all equations

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations

        Returns:
            Array: The residuals for all equations
        """
        Jsi = {
            (id(arg[0]),) + z: arg[0]
            for k, arg in zip(self.Jres, args, strict=True)
            for z in k
        }
        Js = self.update_arrays(module, Jsi)
        L2 = []
        for res, arg in zip(self.residuals, args, strict=True):
            L2.append((res.weights * res(Js, arg[0], arg[1]) ** 2).mean())
        return jnp.array(L2)

    def compute_Li(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...], i: int
    ) -> Array:
        """Return the loss for equation i

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations
            i: The equation number

        Returns:
            Array: The loss for equation i
        """
        x = self.compute_residual_i(module, args, i)
        return (self.residuals[i].weights * x**2).mean()

    def norm_grad_loss_i(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...], i: int
    ) -> Array:
        """Return the norm of the gradient of the loss for equation i

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations
            i: The equation number

        Returns:
            Array: The norm of the gradient of the loss for equation i
        """
        return jnp.linalg.norm(
            jax.flatten_util.ravel_pytree(nnx.grad(self.compute_Li)(module, args, i))[0]
        )

    def norm_grad_loss(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        """Return the norms of the gradients of the losses for all equations

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations

        Returns:
            Array: The norms of the gradients of the losses for all equations
        """
        norms = []
        for i in range(len(self.residuals)):
            norms.append(self.norm_grad_loss_i(module, args, i))
        return jnp.array(norms)

    @partial(nnx.jit, static_argnums=0)
    def compute_global_weights(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        """Return global weights based on the norms of the gradients

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations

        Returns:
            Array: The global weights
        """
        norms = self.norm_grad_loss(module, args)
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

        new = self.compute_global_weights(module, self.args)
        if jax.process_count() > 1:
            new = mh.process_allgather(new).mean(0)
        return new * (1 - alpha) + gw * alpha

    @partial(nnx.jit, static_argnums=0)
    def loss_with_gw(
        self, module: nnx.Module, args: tuple[tuple[Array, Array], ...], gw: Array,
    ) -> Array:
        """Return the weighted loss with given global weights

        Args:
            module: The module (nnx.Module)
            args: The collocation points and targets for all equations
            gw: Global weights

        Returns:
            Array: The weighted loss
        """
        Jsi = {
            (id(arg[0]),) + z: arg[0]
            for k, arg in zip(self.Jres, args, strict=True)
            for z in k
        }
        Js = self.update_arrays(module, Jsi)
        return (
            jnp.array(
                [
                    (eq.weights * eq(Js, arg[0], arg[1]) ** 2).mean()
                    for i, (eq, arg) in enumerate(
                        zip(self.residuals, args, strict=True)
                    )
                ]
            )
            @ gw
        )

    @partial(nnx.jit, static_argnums=0)
    def __call__(self, module: nnx.Module) -> Array:
        """Return the total least squares loss

        Args:
            module: The module (nnx.Module)

        Returns:
            Array: The total least squares loss
        """
        args = self.args
        Jsi = {
            (id(arg[0]),) + z: arg[0]
            for k, arg in zip(self.Jres, args, strict=True)
            for z in k
        }
        Js = self.update_arrays(module, Jsi)
        return sum(
            [
                (eq.weights * eq(Js, arg[0], arg[1]) ** 2).mean()
                for i, (eq, arg) in enumerate(zip(self.residuals, args, strict=True))
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
            return lambda x, Js, z=z: z(*x.T)
        else:
            f0 = jnp.array(float(f) if f.is_real else complex(f))
            return lambda x, Js, f=f0: f

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

        return lambda x, Js, s=s, f=gc, f0=gi: get_fn(f, s)(x, Js) * jnp.prod(
            jnp.array([get_fn(fi, s)(x, Js) for fi in f0]), axis=0
        )

    elif isinstance(f, sp.Pow):
        fi = f.args[0]
        p = int(f.args[1])
        return lambda x, Js, s=s, f=fi, p=p: get_fn(f, s)(x, Js) ** p

    assert len(v) == 1
    v = v.pop()
    return partial(
        _lookup_array,
        mod=id(v.module),
        i=v.global_index,
        k=int(getattr(f, "derivative_count", "0")),
        variables=getattr(f, "variables", ()),
    )
    # return partial(
    #    _compute_gradient,
    #    i=v.global_index,
    #    k=int(getattr(f, "derivative_count", "0")),
    #    variables=getattr(f, "variables", ()),
    # )


def _compute_gradient(
    x: Array,
    mod: nnx.Module,
    i: int = 0,
    k: int = 1,
    variables: tuple[int] = [0],
):
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    return vjacn(mod, k)(x)[var]


def _lookup_array(
    x: Array,
    Js: dict[tuple[int, int, int], Array],
    mod: int = 0,
    i: int = 0,
    k: int = 1,
    variables: tuple[int] = [0],
) -> Array:
    """Lookup precomputed derivative from Js dictionary"""
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    return Js[(id(x), mod, k)][var]
