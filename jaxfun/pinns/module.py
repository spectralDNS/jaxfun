import copy
from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import Any

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from flax.typing import Initializer
from jax import Array
from sympy import Function
from sympy.printing import latex
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import VectorAdd

from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, CoordSys, latex_symbols
from jaxfun.utils.common import lambdify, ulp

moduledict = {}


# Differs from jaxfun.utils.common.jacn in the last if else
def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    for i in range(k):
        fun = jax.jacrev(fun) if i % 2 else jax.jacfwd(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun


class MLPSpace(BaseSpace):
    """Multilayer perceptron functionspace"""

    def __init__(
        self,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        transient: bool = False,
        offset: int = 0,
        *,
        name: str,
    ) -> None:
        from jaxfun.arguments import CartCoordSys, x, y, z

        self.in_size = dims + int(transient)
        self.hidden_size = hidden_size
        self.out_size = dims**rank
        self.dims = dims
        self.rank = rank
        self.offset = offset
        self.transient = transient
        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[dims])
            if system is None
            else system
        )
        BaseSpace.__init__(self, system, name)


MLPVectorSpace = partial(MLPSpace, rank=1)


class CompositeMLP:
    """Multilayer perceptron composite functionspace

    To be used for multiple outputs, or multiple coupled
    equations.
    """

    def __init__(self, mlpspaces: list[MLPSpace]) -> None:
        offset = 0
        newspaces = []
        self.name = "".join([V.name for V in mlpspaces])
        self.system = mlpspaces[0].system
        for i, mlp in enumerate(mlpspaces):
            newmlp = MLPSpace(
                mlp.hidden_size,
                mlp.dims,
                mlp.rank,
                system=self.system,
                transient=mlp.transient,
                offset=offset,
                name=self.name + "_" + str(i),
            )
            offset += newmlp.out_size
            newspaces.append(newmlp)
            assert newmlp.hidden_size == mlpspaces[0].hidden_size
            assert newmlp.dims == mlpspaces[0].dims

        self.mlp = newspaces
        self.in_size = self.mlp[0].in_size
        self.hidden_size = self.mlp[0].hidden_size
        self.out_size = sum([p.out_size for p in self.mlp])

    def __getitem__(self, i: int) -> MLPSpace:
        return self.mlp[i]

    def __len__(self) -> int:
        return len(self.mlp)


class MLP(nnx.Module):
    def __init__(
        self,
        V: BaseSpace | CompositeMLP,
        *,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        hidden_size = (
            V.hidden_size
            if isinstance(V.hidden_size, list | tuple)
            else [V.hidden_size]
        )
        self.linear_in = nnx.Linear(
            V.in_size,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden = [
            nnx.Linear(
                hidden_size[i],
                hidden_size[min(i + 1, len(hidden_size) - 1)],
                rngs=rngs,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=float,
                dtype=float,
            )
            for i in range(len(hidden_size))
        ]
        self.linear_out = nnx.Linear(
            hidden_size[-1],
            V.out_size,
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )

    @property
    def dim(self) -> int:
        gd, st = nnx.split(self)
        pyt, ret = jax.flatten_util.ravel_pytree(st)
        return pyt.shape[0]

    def __call__(self, x: Array) -> Array:
        x = nnx.tanh(self.linear_in(x))
        for z in self.hidden:
            x = nnx.tanh(z(x))
        return self.linear_out(x)


class SpectralModule(nnx.Module):
    def __init__(
        self,
        basespace: BaseSpace,
        *,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        self.kernel = nnx.Param(kernel_init(rngs(), (1, basespace.N)))
        self.space = basespace
        self.space.offset = 0

    @property
    def dim(self) -> int:
        return self.space.dim

    def __call__(self, x: Array) -> Array:
        # return self.space.evaluate2(
        #    self.space.map_reference_domain(x), self.kernel.value[0]
        # )
        return (
            jax.vmap(self.space.eval_basis_functions)(
                self.space.map_reference_domain(x)
            ).squeeze()
            @ self.kernel.value.T
        )


class FlaxFunction(Function):
    def __new__(
        cls,
        V: BaseSpace | CompositeMLP,
        name: str,
        *,
        module: nnx.Module = None,
        fun_str: str = None,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(list(coors._cartesian_xyz) + [sp.Symbol(V.name)]))
        obj.functionspace = V
        obj.module = (
            obj.get_flax_module(
                V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
            )
            if module is None
            else module
        )
        obj.name = name
        obj.fun_str = fun_str if fun_str is not None else name
        obj.argument = 2
        if isinstance(V, CompositeMLP):
            assert len(name) == len(V)
        return obj

    def __getitem__(self, i: int):
        return FlaxFunction(
            self.functionspace[i], name=self.name[i], module=self.module
        )

    @property
    def rank(self) -> int:
        return (
            None
            if isinstance(self.functionspace, CompositeMLP)
            else self.functionspace.rank
        )

    @property
    def is_scalar(self) -> bool:
        return self.rank == 0

    @property
    def is_Vector(self) -> bool:
        return self.rank == 1

    @property
    def is_Dyadic(self) -> bool:
        return self.rank == 2

    @staticmethod
    def get_flax_module(
        V,
        *,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs,
    ) -> nnx.Module:
        if isinstance(V, MLPSpace | CompositeMLP):
            return MLP(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        return SpectralModule(
            V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def doit(self, **hints: dict) -> sp.Expr:
        from jaxfun.arguments import functionspacedict

        V = self.functionspace
        functionspacedict[V.name] = V
        moduledict[V.name] = self.module

        if isinstance(V, CompositeMLP):
            raise RuntimeError

        if V.rank == 0:
            return Function(
                self.fun_str,
                global_index=V.offset,
                functionspace_name=V.name,
                rank_parent=V.rank,
                module=self.module,
                argument=2,
            )(*V.system.base_scalars())

        if V.rank == 1:
            b = V.system.base_vectors()
            s = V.system.base_scalars()
            return VectorAdd.fromiter(
                Function(
                    self.fun_str + "_" + s[i].name,
                    global_index=V.offset + i,
                    functionspace_name=V.name,
                    rank_parent=V.rank,
                    module=self.module,
                    argument=2,
                )(*s)
                * b[i]
                for i in range(V.dims)
            )
        raise NotImplementedError

    def cartesian_mesh(self, xs: Array) -> Array:
        """Return mesh in Cartesian (physical) domain

        Args:
            xs (Array): Coordinates in computational domain

        Returns:
            Array: Coordinates in real space
        """
        system = self.functionspace.system
        rv = system.position_vector(False)
        s = system.base_scalars()
        mesh = []
        for r in rv:
            mesh.append(lambdify(s, r, modules="jax")(*xs.T))
        return jnp.array(mesh).T

    def __str__(self) -> str:
        name = "\033[1m%s\033[0m" % (self.name,) if self.rank == 1 else self.name
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _latex(self, printer: Any = None) -> str:
        name = r"\mathbf{ {%s} }" % (self.name,) if self.rank == 1 else self.name  # noqa: UP031
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.functionspace.system._cartesian_xyz]),
                "; ",
                self.functionspace.name,
                ")",
            )
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def __call__(self, x: Array) -> Array:
        y = self.module(x)
        V = self.functionspace
        if self.rank == 0:
            return y[:, V.offset]
        elif self.rank == 1:
            return y[:, V.offset : V.offset + V.dims]
        return y


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
    V = f.functionspace
    offset = V.offset if f.rank == 0 else slice(V.offset, V.offset + V.dims)
    var: tuple[int] = tuple((slice(None), offset)) + tuple(
        int(s._id[0]) for s in expr.variables
    )
    return du[var]


# Experimental...
class Comp(nnx.Module):
    def __init__(self, flaxfunctions: list[FlaxFunction]) -> None:
        self.flaxfunctions = flaxfunctions
        [setattr(self, str(id(p.module)), p.module) for p in flaxfunctions]

    def __call__(self, x: Array) -> Array:
        return jnp.hstack([f.module(x) for f in self.flaxfunctions])


def expand(forms: sp.Expr) -> list[sp.Expr]:
    """Expand and collect all terms without basis functions

    Args:
        forms: Sympy expression

    Returns:
        A list of sp.Exprs as arguments to Add
    """
    f = sp.Add.make_args(forms.doit().expand())
    # return f
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
    def __init__(
        self,
        f: sp.Expr,
        x: Array,
        target: Array | Number = 0,
        weights: Array | Number = 1,
    ) -> None:
        from jaxfun.forms import get_system

        sys = get_system(f.doit())
        s = sys.base_scalars()
        t, expr = expand(f)
        self.eqs = [get_fn(h, s) for h in expr]
        self.x = x
        # Place all terms without flaxfunctions in the target
        self.target = target
        if len(t.free_symbols) > 0:
            self.target = target - lambdify(s, t, modules="jax")(*x.T)
        elif t != 0:
            self.target = target - t
        self.weights = weights

    def __call__(self, Js) -> Array:
        return sum([eq(self.x, Js) for eq in self.eqs]) - self.target


class LSQR:
    """Least squares loss function"""

    def __init__(
        self,
        fs: tuple[sp.Expr, Array, Array | Number | None, Array | Number | None],
    ):
        """The least squares method is to compute the loss over all input equations at
        all collocation points. The equations are all defined with their own points

        Args:
            fs (tuple[tuple[sp.Expr, Array, Array  |  Number  |  None, Array  |  Number  |  None]]):
                tuple of tuples, where the latter contains the subproblems that
                are to be solved. The subproblems are defined by the equation
                residuals (first item) and the collocation points (second item)
                used to evaluate the residuals. The third item is the target,
                which is zero by defauls, whereas the last item is an optional
                weight. The weight needs to be a number or an array of the same
                shape as the collocation points.

        Examples:

            >>> import jax.numpy as jnp
            >>> from flax import nnx
            >>> import optax
            >>> from jaxfun.operators import Div, Grad
            >>> from jaxfun.pinns.module import LSQR, MLPSpace, FlaxFunction
            >>> V = MLPSpace([8, 8], dims=1, rank=0, name="V")
            >>> u = FlaxFunction(V, rngs=nnx.Rngs(1001), name="u")
            >>> eq = Div(Grad(u)) + 2
            >>> xj = jnp.linspace(-1, 1, 10)[:, None]
            >>> xb = jnp.array([[-1.0], [1.0]])
            >>> loss_fn = LSQR(((eq, xj, 0, 1), (u, xb, 0, 10)))
        """
        from jaxfun.forms import get_system
        from jaxfun.operators import Dot

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

        self.Jres = [set() for _ in range(len(res))]
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
        Jsi = {k: self.Js[k] for k in self.Jres[i]}
        self.update_arrays(model, Jsi)
        self.Js.update(Jsi)
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

    def __call__(self, model: nnx.Module) -> float:
        self.update_arrays(model, self.Js)
        return sum([(eq.weights * eq(self.Js) ** 2).mean() for eq in self.residuals])


def get_fn(f: sp.Expr, s: tuple[BaseScalar]) -> Callable[[Array, dict], Array]:
    """Return Sympy Expr as function evaluated by points and gradients

    Args:
        f (sp.Expr)
        s (x, y, ...)

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

        def mult(gg):
            g0 = gg[0]
            for gj in gg[1:]:
                g0 = g0 * gj
            return g0

        return lambda x, Js, gc0=gc, gi0=gi: get_fn(gc0, s)(x, Js) * mult(
            [get_fn(gii, s)(x, Js) for gii in gi0]
        )
        # return lambda x, Js, gc0=gc, gi0=gi: get_fn(gc0, s)(x, Js) * jnp.prod(
        #    jnp.array([get_fn(gii, s)(x, Js) for gii in gi0]), axis=0
        # )

    elif isinstance(f, sp.Pow):
        bii = f.args[0]
        p: int = int(f.args[1])
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
    train: Callable[[nnx.Module, nnx.Optimizer], float],
    model: nnx.Module,
    opt: nnx.Optimizer,
    num: int,
    name: str,
    epoch_print: int = 100,
    abs_limit_loss: float = ulp(1000),
    abs_limit_change: float = ulp(100),
):
    loss_old = 1.0
    for epoch in range(1, num + 1):
        loss = train(model, opt)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < abs_limit_loss or abs(loss - loss_old) < abs_limit_change:
            break
        loss_old = loss


# def train_CPINN(
#    eqs: list[Residual],
# ) -> Callable[[nnx.Module, nnx.Optimizer, nnx.Module, nnx.Optimizer], float]:
#    @nnx.jit
#    def train_step(
#        model: nnx.Module,
#        optimizer: nnx.Optimizer,
#        discriminator: nnx.Module,
#        optd: nnx.Optimizer,
#    ) -> tuple[float, float]:
#        gd, state = nnx.split(model)
#        unravel = jax.flatten_util.ravel_pytree(state)[1]
#
#        def loss_fn(mod: nnx.Module) -> Array:
#            return sum([((eq(mod) * discriminator(eq.x)) ** 2).mean() for eq in eqs])
#
#        loss, gradients = nnx.value_and_grad(loss_fn)(model)
#        loss_fn_split = lambda state: loss_fn(nnx.merge(gd, state))
#        H_loss_fn = lambda flat_weights: loss_fn(nnx.merge(gd, unravel(flat_weights)))
#        optimizer.update(
#            gradients,
#            grad=gradients,
#            value_fn=loss_fn_split,
#            value=loss,
#            H_loss_fn=H_loss_fn,
#        )
#
#        def loss_fn_d(disc: nnx.Module) -> Array:
#            return sum([((eq(disc) * model(eq.x)) ** 2).mean() for eq in eqs])
#
#        gdd, stated = nnx.split(discriminator)
#        unraveld = jax.flatten_util.ravel_pytree(stated)[1]
#        lossd, gradd = nnx.value_and_grad(loss_fn_d)(discriminator)
#        loss_fn_d_split = lambda state: loss_fn_d(nnx.merge(gdd, state))
#        H_loss_fn_d = lambda flat_weights: loss_fn_d(
#            nnx.merge(gdd, unraveld(flat_weights))
#        )
#        gradd = jax.tree_util.tree_map(lambda p: -p, gradd)
#        optd.update(
#            gradd,
#            grad=gradd,
#            value_fn=loss_fn_d_split,
#            value=lossd,
#            H_loss_fn=H_loss_fn_d,
#        )
#
#        return loss, lossd
#
#    return train_step
#
#
# def run_optimizer_d(t, model, opt, disc, optd, num, name, epoch_print=100):
#    loss_old = 1.0
#    for epoch in range(1, num + 1):
#        loss, lossd = t(model, opt, disc, optd)
#        if epoch % epoch_print == 0:
#            print(f"Epoch {epoch} {name}, loss: {loss}, lossd: {lossd}")
#        if abs(loss) < ulp(1000) or abs(loss - loss_old) < ulp(1):
#            break
#        loss_old = loss
