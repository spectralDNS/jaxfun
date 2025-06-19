import copy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from flax.typing import Initializer
from jax import Array
from sympy import Function
from sympy.printing.pretty.stringpict import prettyForm

from jaxfun.arguments import test
from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import Domain, jacn, lambdify, ulp


class NNSpace(BaseSpace):
    def __init__(
        self,
        in_size: int,
        hidden_size: list[int] | int,
        out_size: int,
        system: CoordSys = None,
        name: str = "NN",
        fun_str: str = "theta",
        rank: int = 0,
    ) -> None:
        from jaxfun.arguments import CartCoordSys, x, y, z

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.rank = rank
        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[in_size])
            if system is None
            else system
        )
        BaseSpace.__init__(self, system, name, fun_str)


class NNFunction(Function):
    def __init__(
        self,
        V: BaseSpace,
        module: nnx.Module,
        *,
        name: str = None,
        fun_str: str = "phi",
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        self.functionspace = V
        self.module = module(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        self.name = name
        self.fun_str = fun_str

    def __new__(
        cls,
        V: BaseSpace,
        module: nnx.Module,
        *,
        name: str = None,
        fun_str: str = "phi",
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Symbol(V.name)]))
        return obj

    def doit(self, **hints: dict) -> sp.Expr:
        from jaxfun.arguments import BasisFunctionNN

        return BasisFunctionNN(
            self.functionspace.system.base_scalars(),
            "-".join((self.functionspace.name, self.fun_str)),
        )

    def __str__(self) -> str:
        name = self.name if self.name is not None else "NNFunction"
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
        name = self.name if self.name is not None else "NNFunction"
        name = name if self.functionspace.rank == 0 else r"\mathbf{ {%s} }" % (name,)  # noqa: UP031
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

    def __call__(self, x):
        return self.module(x)


class SpectralModule(nnx.Module):
    def __init__(
        self,
        basespace: BaseSpace,
        *,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        domain: Domain = (-1, 1),
        rngs: nnx.Rngs,
    ) -> None:
        self.kernel = nnx.Param(kernel_init(rngs(), (1, basespace.N)))
        self.space = basespace

    @partial(jax.jit, static_argnums=0)
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


class MLP(nnx.Module):
    def __init__(
        self,
        V: BaseSpace,
        *,
        rngs: nnx.Rngs,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
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

    def __call__(self, x: Array) -> Array:
        x = nnx.tanh(self.linear_in(x))
        for z in self.hidden:
            x = nnx.tanh(z(x))
        return self.linear_out(x)


class LSQRes:
    def __init__(self, f: sp.Expr, x: Array) -> None:
        from jaxfun.forms import split

        self.eqs = [get_fn(hi) for hi in (split(f)["linear"])]
        self.x = x

    def __call__(self, model):
        return sum([eq(model, self.x) for eq in self.eqs])


def add_constant(res):
    from jaxfun.forms import get_basisfunctions

    vals = res[0].copy()
    x = list(vals.keys())[0]
    vals[x] = 0
    vals["coeff"] = 1
    rs = []
    for h in res:
        v = get_basisfunctions(h[x])[0]
        if v is None:
            vals[x] = vals[x] + h["coeff"] * h[x]
        else:
            rs.append(h)
    if vals[x] != 0:
        vals[x] = vals[x].factor()
        rs.append(vals)
    return rs


def get_fn(f: sp.Expr):
    from jaxfun.forms import get_basisfunctions

    sc = sp.sympify(f["coeff"])
    sc = float(sc) if sc.is_real else complex(sc)
    for key, bi in f.items():
        if key in ("coeff", "multivar"):
            continue

        if isinstance(bi, test):
            g = lambda mod, x: sc * mod(x)
            continue

        v, _ = get_basisfunctions(bi)

        if v is None:
            if len(bi.free_symbols) > 0:
                s = bi.free_symbols.pop()
                g = lambda mod, x, s0=s, bi0=bi: lambdify(s0, sc * bi0, modules="jax")(
                    x
                )
            else:
                s = copy.copy(float(sc * bi))
                g = lambda mod, x, s0=s: jnp.array(s0)
        else:
            if isinstance(bi, sp.Mul):
                gi = [lambda mod, x: jnp.ones(1) * copy.copy(sc)]
                for bii in bi.args:
                    k: int = 0
                    if hasattr(bii, "derivative_count"):
                        k = int(bii.derivative_count)
                    gi.append(lambda mod, x, k0=k: jacn(mod, k0)(x).reshape((-1, 1)))

                def mult(gg):
                    g0 = gg[0]
                    for gj in gg[1:]:
                        g0 = g0 * gj
                    return g0

                g = lambda mod, x, gi0=gi: mult([gii(mod, x) for gii in gi0])

            elif isinstance(bi, sp.Pow):
                bii = bi.args[0]
                p: int = copy.copy(int(bi.args[1]))
                k: int = 0
                if hasattr(bii, "derivative_count"):
                    k = int(bii.derivative_count)
                g = lambda mod, x, k0=k: sc * (jacn(mod, k0)(x).reshape((-1, 1))) ** p

            else:
                k: int = 0
                if hasattr(bi, "derivative_count"):
                    k = int(bi.derivative_count)
                g = lambda mod, x, k0=k: sc * jacn(mod, k0)(x).reshape((-1, 1))

    return g


def train(eqs: list[LSQRes]):
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer) -> Array:
        gd, state = nnx.split(model)
        unravel = jax.flatten_util.ravel_pytree(state)[1]

        def loss_fn(model: nnx.Module) -> Array:
            return sum([(eq(model) ** 2).mean() for eq in eqs])

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


def run_optimizer(t, model, opt, num, name, epoch_print=100):
    loss_old = 1.0
    for epoch in range(num):
        loss = t(model, opt)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < ulp(1000) or abs(loss - loss_old) < ulp(100):
            break
        loss_old = loss
