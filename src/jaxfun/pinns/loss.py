import copy
from collections.abc import Callable
from functools import partial
from numbers import Number

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx

from jaxfun.coordinates import BaseScalar
from jaxfun.pinns.module import Comp, Function
from jaxfun.typing import Array, LSQR_Tuple
from jaxfun.utils import jacn, lambdify


def get_args(a: sp.Expr) -> tuple[sp.Symbol | BaseScalar, ...]:
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


def eval_flaxfunction(expr, x: Array):
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
        self, f: sp.Expr, x: Array, target: Array = 0, weights: Array = 1
    ) -> None:
        """Residual of a single equation evaluated at collocation points

        Args:
            f: Sympy expression representing the equation residual. The
                expression may contain one or several FlaxFunctions
            x: Collocation points where the residual is evaluated
            target: Target value of the residual. Defaults to 0.
            weights: Weights for the residual. Defaults to 1.
        """
        s = get_args(f.doit())
        t, expr = expand(f)
        self.eqs = [get_fn(h, s) for h in expr]
        self.x = x
        # Place all terms without flaxfunctions in the target, because these will not need to be computed more than once
        self.target = target
        if len(t.free_symbols) > 0:
            self.target = target - lambdify(s, t, modules="jax")(*x.T)
        elif isinstance(t, Number):
            self.target = target - float(t)
        self.target = jnp.squeeze(self.target)
        self.weights = weights

    def __call__(self, Js) -> Array:
        return sum([eq(self.x, Js) for eq in self.eqs]) - self.target


class LSQR:
    """Least squares loss function"""

    def __init__(self, *fs: LSQR_Tuple, alpha: float = 0.9):
        """The least squares method computes the loss over all input equations at
        all collocation points. The different equations are all defined with their
        own points.

        Args:
            fs:
                One or several tuples. The tuples contain the subproblems that
                are to be solved. The subproblems are defined by the equation
                residuals (first item) and the collocation points (second item)
                used to evaluate the residuals. The third item is the target,
                which is zero by defauls, whereas the last item is an optional
                weight. The weight needs to be a number or an array of the same
                shape as the collocation points.
            alpha:
                Update factor for adaptive weighting of loss functions for each
                subproblem.

        Examples:

            >>> import jax.numpy as jnp
            >>> from jaxfun.operators import Div, Grad
            >>> from jaxfun.pinns.module import LSQR, MLPSpace, FlaxFunction
            >>> V = MLPSpace([8, 8], dims=1, rank=0, name="V")
            >>> u = FlaxFunction(V, name="u")
            >>> eq = Div(Grad(u)) + 2
            >>> xj = jnp.linspace(-1, 1, 10)[:, None]
            >>> xb = jnp.array([[-1.0], [1.0]])
            >>> loss_fn = LSQR((eq, xj, 0, 1), (u, xb, 0, 10))
        """
        from jaxfun.forms import get_system
        from jaxfun.operators import Dot

        self.alpha = alpha
        self.residuals = []
        self.Js = Js = {}  # All modules' evaluation and derivatives eval wrt variables
        self.xs = xs = {}  # All collocation points
        res = []

        for f in fs:
            f0 = f[0].doit()
            if f0.is_Vector:  # Vector equation
                sys = get_system(f0)
                for i in range(sys.dims):
                    bt = sys.get_contravariant_basis_vector(i)
                    g = (Dot(f0, bt),) + (f[1],)
                    if len(f) > 2:
                        if isinstance(f[2], Number):
                            g += (f[2],)
                        else:
                            g += (f[2][..., i],)
                    if len(f) > 3:
                        g += (f[3],)

                    self.residuals.append(Residual(*g))
                    res.append((g[0], g[1]))

            else:
                self.residuals.append(Residual(*f))
                res.append((f[0], f[1]))

        self.global_weights = jnp.ones(len(self.residuals), dtype=int)
        self.Jres = [set() for _ in range(len(res))]  # Collection for each residual
        for i, f in enumerate(res):
            f0 = f[0].doit()
            for s in sp.core.traversal.preorder_traversal(f0):
                if isinstance(s, sp.Derivative):
                    func = s.args[0]
                    if hasattr(func, "module"):
                        key = (id(f[1]), id(func.module), s.derivative_count)
                        if key not in Js:
                            Js[key] = jacn(func.module, s.derivative_count)(f[1])
                        self.Jres[i].add(key)

                if hasattr(s, "module"):
                    key = (id(f[1]), id(s.module), 0)
                    if key not in Js:
                        Js[key] = s.module(f[1])
                    self.Jres[i].add(key)

            if id(f[1]) not in xs:
                xs[id(f[1])] = f[1]

    def update_arrays(self, model: nnx.Module, Js: dict) -> None:
        for k in Js:
            mod = (
                model.__getattribute__(str(k[1])) if isinstance(model, Comp) else model
            )
            Js[k] = jacn(mod, k[2])(self.xs[k[0]])

    def compute_residual_i(self, model: nnx.Module, i: int) -> Array:
        Jsi = {k: None for k in self.Jres[i]}
        self.update_arrays(model, Jsi)
        return self.residuals[i](Jsi)

    def compute_residuals(self, model: nnx.Module):
        self.update_arrays(model, self.Js)
        L2 = []
        for res in self.residuals:
            L2.append((res.weights * res(self.Js) ** 2).mean())
        return jnp.array(L2)

    def compute_Li(self, model: nnx.Module, i: int):
        x = self.compute_residual_i(model, i)
        return (self.residuals[i].weights * x**2).mean()

    def norm_grad_loss_i(self, model: nnx.Module, i: int) -> float:
        return jnp.linalg.norm(
            jax.flatten_util.ravel_pytree(nnx.grad(self.compute_Li)(model, i))[0]
        )

    def norm_grad_loss(self, model: nnx.Module) -> Array:
        norms = []
        for i in range(len(self.residuals)):
            norms.append(self.norm_grad_loss_i(model, i))
        return jnp.array(norms)

    @partial(nnx.jit, static_argnums=0)
    def compute_global_weights(self, model: nnx.Module) -> Array:
        norms = self.norm_grad_loss(model)
        return jnp.sum(norms) / jnp.where(norms < 1e-6, 1e-6, norms)

    def update_global_weights(self, model: nnx.Module) -> None:
        new = self.compute_global_weights(model)
        old = self.global_weights
        self.global_weights = old * (1 - self.alpha) + new * self.alpha

    def __call__(self, model: nnx.Module) -> float:
        self.update_arrays(model, self.Js)
        return sum(
            [
                self.global_weights[i] * (eq.weights * eq(self.Js) ** 2).mean()
                for i, eq in enumerate(self.residuals)
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
            return lambda x, Js, s0=s, bi0=f: lambdify(s0, bi0, modules="jax")(*x.T)
        else:
            s1 = copy.copy(float(f))
            return lambda x, Js, s0=s1: jnp.array(s0)

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

        return lambda x, Js, gc0=gc, gi0=gi: get_fn(gc0, s)(x, Js) * jnp.prod(
            jnp.array([get_fn(gii, s)(x, Js) for gii in gi0]), axis=0
        )

    elif isinstance(f, sp.Pow):
        bii = f.args[0]
        p = int(f.args[1])
        return lambda x, Js, bi0=bii, p0=p: get_fn(bi0, s)(x, Js) ** p0

    assert len(v) == 1
    v = v.pop()
    return partial(
        lookup_array,
        mod=id(v.module),
        i=v.global_index,
        k=int(getattr(f, "derivative_count", "0")),
        variables=getattr(f, "variables", ()),
    )


def lookup_array(
    x: Array,
    Js: dict,
    mod: int = 0,
    i: int = 0,
    k: int = 1,
    variables: tuple[int] = [0],
) -> Array:
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    return Js[(id(x), mod, k)][var]
