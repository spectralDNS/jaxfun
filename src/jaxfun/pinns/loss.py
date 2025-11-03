from collections.abc import Callable
from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx

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
        self.weights = weights
        self.x, self.target = x, t0

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
        Jres = [set() for _ in range(len(res))]
        for i, f in enumerate(res):
            f0 = f.doit()
            for s in sp.core.traversal.preorder_traversal(f0):
                if isinstance(s, sp.Derivative):
                    func = s.args[0]
                    ki: int = int(s.derivative_count)
                    assert hasattr(func, "module")
                    key = (id(func.module), ki)
                    Jres[i].add(key)

                if hasattr(s, "module"):
                    key = (id(s.module), 0)
                    Jres[i].add(key)
        self.Jres = Jres

    @property
    def args(self) -> tuple[tuple[Array, Array], ...]:
        return tuple((eq.x, eq.target) for eq in self.residuals)

    @partial(nnx.jit, static_argnums=0)
    def update_arrays(
        self, model: nnx.Module, Jsi: dict[tuple[int, int, int], Array]
    ) -> None:
        Js = {}
        for k, xs in Jsi.items():
            mod = (
                model.__getattribute__(str(k[1])) if isinstance(model, Comp) else model
            )
            Js[k] = jacn(mod, k[2])(xs)

        return Js

    def compute_residual_i(
        self, model: nnx.Module, args: tuple[tuple[Array, Array], ...], i: int
    ) -> Array:
        Jsi = {(id(args[i][0]),) + k: args[i][0] for k in self.Jres[i]}
        Js = self.update_arrays(model, Jsi)
        return self.residuals[i](Js, args[i][0], args[i][1])

    def compute_residuals(
        self, model: nnx.Module, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        Jsi = {
            (id(arg[0]),) + z: arg[0]
            for k, arg in zip(self.Jres, args, strict=True)
            for z in k
        }
        Js = self.update_arrays(model, Jsi)
        L2 = []
        for res, arg in zip(self.residuals, args, strict=True):
            L2.append((res.weights * res(Js, arg[0], arg[1]) ** 2).mean())
        return jnp.array(L2)

    def compute_Li(
        self, model: nnx.Module, args: tuple[tuple[Array, Array], ...], i: int
    ):
        x = self.compute_residual_i(model, args, i)
        return (self.residuals[i].weights * x**2).mean()

    def norm_grad_loss_i(
        self, model: nnx.Module, args: tuple[tuple[Array, Array], ...], i: int
    ) -> Array:
        return jnp.linalg.norm(
            jax.flatten_util.ravel_pytree(nnx.grad(self.compute_Li)(model, args, i))[0]
        )

    def norm_grad_loss(
        self, model: nnx.Module, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        norms = []
        for i in range(len(self.residuals)):
            norms.append(self.norm_grad_loss_i(model, args, i))
        return jnp.array(norms)

    @partial(nnx.jit, static_argnums=0)
    def compute_global_weights(
        self, model: nnx.Module, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        norms = self.norm_grad_loss(model, args)
        return jnp.sum(norms) / jnp.where(norms < 1e-16, 1e-16, norms)

    def update_global_weights(
        self,
        model: nnx.Module,
        gw: Array,
        args: tuple[tuple[Array, Array], ...],
        alpha: float,
    ) -> Array:
        from jax.experimental import multihost_utils as mh

        new = self.compute_global_weights(model, args)
        if jax.process_count() > 1:
            new = mh.process_allgather(new).mean(0)
        return new * (1 - alpha) + gw * alpha

    @partial(nnx.jit, static_argnums=0)
    def loss_with_gw(
        self, model: nnx.Module, gw: Array, args: tuple[tuple[Array, Array], ...]
    ) -> Array:
        Jsi = {
            (id(arg[0]),) + z: arg[0]
            for k, arg in zip(self.Jres, args, strict=True)
            for z in k
        }
        Js = self.update_arrays(model, Jsi)

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
    def __call__(self, model: nnx.Module) -> Array:
        args = self.args
        Jsi = {
            (id(arg[0]),) + z: arg[0]
            for k, arg in zip(self.Jres, args, strict=True)
            for z in k
        }
        Js = self.update_arrays(model, Jsi)
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
        lookup_array,
        mod=id(v.module),
        i=v.global_index,
        k=int(getattr(f, "derivative_count", "0")),
        variables=getattr(f, "variables", ()),
    )
    # return partial(
    #    compute_gradient,
    #    # mod=v.module,
    #    i=v.global_index,
    #    k=int(getattr(f, "derivative_count", "0")),
    #    variables=getattr(f, "variables", ()),
    # )


def compute_gradient(
    x: Array,
    mod: nnx.Module,
    i: int = 0,
    k: int = 1,
    variables: tuple[int] = [0],
):
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    return vjacn(mod, k)(x)[var]


def lookup_array(
    x: Array,
    Js: dict[tuple[int, int, int], Array],
    mod: int = 0,
    i: int = 0,
    k: int = 1,
    variables: tuple[int] = [0],
) -> Array:
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    return Js[(id(x), mod, k)][var]
