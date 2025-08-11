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
from sympy.printing.pretty.stringpict import prettyForm

from jaxfun.arguments import FlaxBasisFunction, test
from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, CoordSys
from jaxfun.utils.common import jacn, lambdify, ulp


class MLPSpace(BaseSpace):
    """Multilayer perceptron functionspace"""

    def __init__(
        self,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        name: str = "MLP",
        transient: bool = False,
        offset: int = 0,
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


class PirateSpace(MLPSpace):
    """MLP alternative with PirateNet architecture."""

    def __init__(
        self,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        name: str = "PirateNet",
        transient: bool = False,
        offset: int = 0,
        # PirateNet specific parameters
        nonlinearity: float = 0.0,
        periodicity: dict | None = None,
        fourier_emb: dict | None = None,
        pi_init: jnp.ndarray | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            dims=dims,
            rank=rank,
            system=system,
            name=name,
            transient=transient,
            offset=offset,
        )

        self.nonlinearity = nonlinearity
        self.periodicity = periodicity
        self.fourier_emb = fourier_emb
        self.pi_init = pi_init


class CompositeMLP:
    def __init__(self, mlpspaces, name: str = "CMLP"):
        offset = 0
        newspaces = []
        self.name = name
        self.system = mlpspaces[0].system
        for i, mlp in enumerate(mlpspaces):
            newmlp = MLPSpace(
                mlp.hidden_size,
                mlp.dims,
                mlp.rank,
                system=self.system,
                name=name + "_" + str(i),
                transient=mlp.transient,
                offset=offset,
            )
            offset += newmlp.out_size
            newspaces.append(newmlp)
        self.mlp = newspaces
        self.in_size = self.mlp[0].in_size
        self.hidden_size = self.mlp[0].hidden_size
        self.out_size = sum([p.out_size for p in self.mlp])

    def __getitem__(self, i: int):
        return self.mlp[i]

    def __len__(self):
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

    def __call__(self, x: Array) -> Array:
        x = nnx.tanh(self.linear_in(x))
        for z in self.hidden:
            x = nnx.tanh(z(x))
        return self.linear_out(x)


class PeriodEmbs(nnx.Module):
    """Per-axis cosine/sine embeddings with optionally trainable periods."""

    def __init__(
        self,
        *,
        period: tuple[float, ...],
        axis: tuple[int, ...],
        trainable: tuple[bool, ...],
    ) -> None:
        self.axis = tuple(axis)
        # Store trainable periods as nnx.Param, constants as plain arrays.
        store = {}
        for p, idx, is_trainable in zip(period, axis, trainable, strict=True):
            val = jnp.asarray(p)
            store[f"period_{idx}"] = nnx.Param(val) if is_trainable else val
        self._periods = store

    def __call__(self, x: Array) -> Array:
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                p = self._periods[f"period_{idx}"]
                vals = [jnp.cos(p * xi), jnp.sin(p * xi)]
            else:
                vals = [xi]

            y += vals

        return jnp.hstack(y)


class FourierEmbs(nnx.Module):
    """Gaussian RFFs with fixed projection matrix."""

    def __init__(
        self, *, embed_scale: float, embed_dim: int, in_dim: int, rngs: nnx.Rngs
    ) -> None:
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even (cos & sin halves).")

        init = nnx.initializers.normal(embed_scale)
        k = init(rngs.params(), (in_dim, embed_dim // 2), jnp.float32)
        self.embed_dim = embed_dim
        self.kernel = nnx.Param(k)

    def __call__(self, x: Array) -> Array:
        proj = x @ self.kernel
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)


class Embedding(nnx.Module):
    """Optionally apply PeriodEmbs then FourierEmbs."""

    def __init__(
        self,
        *,
        periodicity: dict | None = None,
        fourier_emb: dict | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        self.periodic = PeriodEmbs(**periodicity) if periodicity else None
        self.fourier = FourierEmbs(**fourier_emb, rngs=rngs) if fourier_emb else None

    def __call__(self, x: Array) -> Array:
        if self.periodic is not None:
            x = self.periodic(x)
        if self.fourier is not None:
            x = self.fourier(x)

        return x


class PIModifiedBottleneck(nnx.Module):
    def __init__(self, *args) -> None:
        raise NotImplementedError


class PirateNet(nnx.Module):
    def __init__(
        self,
        V: PirateSpace,
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
        # TODO: Need a smarter way to handle the input size at each step
        self.embedder = Embedding(
            periodicity=V.periodicity, fourier_emb=V.fourier_emb, rngs=rngs
        )
        in_dim = V.in_size
        if V.periodicity is not None:
            in_dim += len(V.periodicity.axis)
        if V.fourier_emb is not None:
            in_dim = V.fourier_emb.embed_dim

        self.u_net = nnx.Linear(
            in_dim,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.v_net = nnx.Linear(
            in_dim,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden_layers = []
        for i in range(len(hidden_size)):
            layer = PIModifiedBottleneck(
                in_channels=hidden_size[i - 1] if i > 0 else in_dim,
                out_channels=hidden_size[i],
                kernel_init=kernel_init,
                bias_init=bias_init,
                rngs=rngs,
            )
            self.hidden_layers.append(layer)

        if V.pi_init is not None:
            raise NotImplementedError("Least squares initialization not implemented")
        else:
            self.output_layer = nnx.Linear(
                hidden_size[-1],
                V.out_size,
                rngs=rngs,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=float,
                dtype=float,
            )

    def __call__(self, x: Array) -> Array:
        x = self.embedder(x)
        u = nnx.tanh(self.u_net(x))
        v = nnx.tanh(self.v_net(x))

        for layer in self.hidden_layers:
            x = layer(x, u, v)

        y = self.output_layer(x)

        return x, y


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
    def __init__(
        self,
        V: BaseSpace | CompositeMLP,
        name: str,
        *,
        module: nnx.Module = None,
        fun_str: str = None,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs = None,
    ) -> None:
        self.functionspace = V

        self.module = (
            self.get_flax_module(
                V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
            )
            if module is None
            else module
        )
        self.name = name
        if isinstance(V, CompositeMLP):
            assert len(name) == len(V)

    def __new__(
        cls,
        V: BaseSpace | CompositeMLP,
        name: str,
        *,
        module: nnx.Module = None,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs = None,
    ) -> Function:
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Dummy(V.name)]))
        return obj

    def __getitem__(self, i: int):
        return FlaxFunction(
            self.functionspace[i], name=self.name[i], module=self.module
        )

    @property
    def rank(self):
        return (
            None
            if isinstance(self.functionspace, CompositeMLP)
            else self.functionspace.rank
        )

    @staticmethod
    def get_flax_module(
        V,
        *,
        kernel_init: Initializer = nnx.nn.linear.default_kernel_init,
        bias_init: Initializer = nnx.nn.linear.default_bias_init,
        rngs: nnx.Rngs,
    ):
        if isinstance(V, MLPSpace | CompositeMLP):
            return MLP(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        return SpectralModule(
            V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def doit(self, **hints: dict) -> sp.Expr:
        from jaxfun.arguments import FlaxBasisFunction

        V = self.functionspace
        if isinstance(V, CompositeMLP):
            raise RuntimeError

        if V.rank == 0:
            return FlaxBasisFunction(
                *(
                    V.system.base_scalars()
                    + (sp.Dummy("+".join((str(V.offset), V.name, self.name))),)
                )
            )

        if V.rank == 1:
            b = V.system.base_vectors()
            s = V.system.base_scalars()
            return sp.vector.VectorAdd(
                *[
                    FlaxBasisFunction(
                        *(
                            s
                            + (
                                sp.Dummy(
                                    "+".join(
                                        (
                                            str(V.offset + i),
                                            V.name,
                                            self.name + "_" + s[i].name,
                                        )
                                    )
                                ),
                            )
                        )
                    )
                    * b[i]
                    for i in range(V.dims)
                ]
            )
        raise NotImplementedError

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

    def __call__(self, x):
        y = self.module(x)
        V = self.functionspace
        if self.rank == 0:
            return y[:, V.offset]
        elif self.rank == 1:
            return y[:, V.offset : V.offset + V.dims]
        return y


def expand(forms: sp.Expr) -> dict:
    return sp.Add.make_args(forms.doit().expand())


def derivative_count(f: sp.Expr, k: set[int]):
    for p in sp.core.traversal.preorder_traversal(f):
        if isinstance(p, sp.Derivative):
            k.add(int(p.derivative_count))


class Residual:
    def __init__(
        self, f: sp.Expr, x: Array, target: Array = 0, weights: Array = 1
    ) -> None:
        from jaxfun.forms import get_system

        sys = get_system(f)
        s = sys.base_scalars()
        self.eqs = [get_fn(h, s) for h in expand(f)]
        self.x = x
        self.target = target
        self.weights = weights
        self.derivatives = set()
        [derivative_count(h, self.derivatives) for h in expand(f)]

    def __call__(self, model, Js):
        return sum([eq(model, self.x, Js) - self.target for eq in self.eqs])


class LSQR:
    def __init__(self, fs: tuple[tuple[sp.Expr, Array, Array | None]]):
        from jaxfun.forms import get_basisfunctions
        from jaxfun.operators import Dot

        w = get_basisfunctions(fs[0][0])[0]
        assert w is not None
        if isinstance(w, set):
            w = w.pop()
        sys = w.functionspace.system
        self.module = w.module
        self.residuals = []
        for f in fs:
            if f[0].doit().is_Vector:  # Vector equation
                for i, vi in enumerate(sys.base_vectors()):
                    g = (Dot(f[0], vi),) + (f[1],)
                    if len(f) > 2:
                        if isinstance(f[2], Number):
                            g += (f[2],)
                        else:
                            g += (f[2][..., i],)
                    if len(f) > 3:
                        g += (f[3],)
                    self.residuals.append(Residual(*g))
            else:
                self.residuals.append(Residual(*f))

        Js = {}
        xs = {}
        for eq in self.residuals:
            for ki in eq.derivatives:
                key = (id(eq.x), ki)
                if key not in Js:
                    Js[key] = jacn(self.module, ki)(eq.x)
            xs[id(eq.x)] = eq.x
        self.Js = Js
        self.xs = xs

    def update_gradients(self, model):
        for k in self.Js:
            self.Js[k] = jacn(model, k[1])(self.xs[k[0]])

    def compute_residual_i(self, model, i: int):
        self.update_gradients(model)
        return self.residuals[i](model, self.Js)

    def __call__(self, model):
        self.update_gradients(model)
        return sum(
            [(eq.weights * eq(model, self.Js) ** 2).mean() for eq in self.residuals]
        )


def get_fn(f: sp.Expr, s: tuple[BaseScalar]) -> Callable[[nnx.Module, Array], Array]:
    """Return Sympy Expr as function evaluated by Module and spatial coordinates

    Args:
        f (sp.Expr)
        w (FlaxFunction)

    Returns:
        Callable[[nnx.Module, Array], Array]
    """
    from jaxfun.forms import get_basisfunctions

    v, _ = get_basisfunctions(f)

    if v is None:
        # Coefficient independent of basis function
        if len(f.free_symbols) > 0:
            return lambda mod, x, Js, s0=s, bi0=f: lambdify(s0, bi0, modules="jax")(
                *x.T
            )
        else:
            s1 = copy.copy(float(f))
            return lambda mod, x, Js, s0=s1: jnp.array(s0)

    if isinstance(f, sp.Mul):
        # Multiplication of terms that either contain the basis function or not
        gi = []
        gc = []
        for bii in f.args:
            v0, _ = get_basisfunctions(bii)
            # Collect terms that do not contain the basis function
            if v0 is None:
                gc.append(bii)
                continue
            gi.append(bii)
        gc = sp.Mul(*gc)

        def mult(gg):
            g0 = gg[0]
            for gj in gg[1:]:
                g0 = g0 * gj
            return g0

        return lambda mod, x, Js, gc0=gc, gi0=gi: get_fn(gc0, s)(mod, x, Js) * mult(
            [get_fn(gii, s)(mod, x, Js) for gii in gi0]
        )

    elif isinstance(f, sp.Pow):
        bii = f.args[0]
        p: int = int(f.args[1])
        return lambda mod, x, Js, bi0=bii, p0=p: get_fn(bi0, s)(mod, x, Js) ** p0

    elif isinstance(f, FlaxBasisFunction):
        return lambda mod, x, Js: mod(x)[:, f.global_index]

    return partial(
        get_derivative,
        i=v.global_index,
        k=int(f.derivative_count),
        variables=f.variables,
    )

    # return partial(
    #   eval_derivative,
    #   i=v.global_index,
    #   k=f.derivative_count,
    #   variables=f.variables,
    # )


def get_derivative(mod, x, Js, i: int = 0, k: int = 1, variables: tuple[int] = [0]):
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    return Js[(id(x), k)][var]


def eval_derivative(mod, x, Js, i: int = 0, k: int = 1, variables: tuple[int] = [0]):
    var: tuple[int] = tuple((slice(None), i)) + tuple(int(s._id[0]) for s in variables)
    val = jacn(mod, k)(x)
    return val[var]


def train(eqs: LSQR) -> Callable[[nnx.Module, nnx.Optimizer], float]:
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer) -> float:
        gd, state = nnx.split(model, nnx.Param)
        unravel = jax.flatten_util.ravel_pytree(state)[1]

        def loss_fn(model: nnx.Module) -> Array:
            return eqs(model)

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
    for epoch in range(1, num + 1):
        loss = t(model, opt)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < ulp(1000) or abs(loss - loss_old) < ulp(100):
            break
        loss_old = loss


def train_CPINN(
    eqs: list[Residual],
) -> Callable[[nnx.Module, nnx.Optimizer, nnx.Module, nnx.Optimizer], float]:
    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        discriminator: nnx.Module,
        optd: nnx.Optimizer,
    ) -> tuple[float, float]:
        gd, state = nnx.split(model)
        unravel = jax.flatten_util.ravel_pytree(state)[1]

        def loss_fn(mod: nnx.Module) -> Array:
            return sum([((eq(mod) * discriminator(eq.x)) ** 2).mean() for eq in eqs])

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

        def loss_fn_d(disc: nnx.Module) -> Array:
            return sum([((eq(disc) * model(eq.x)) ** 2).mean() for eq in eqs])

        gdd, stated = nnx.split(discriminator)
        unraveld = jax.flatten_util.ravel_pytree(stated)[1]
        lossd, gradd = nnx.value_and_grad(loss_fn_d)(discriminator)
        loss_fn_d_split = lambda state: loss_fn_d(nnx.merge(gdd, state))
        H_loss_fn_d = lambda flat_weights: loss_fn_d(
            nnx.merge(gdd, unraveld(flat_weights))
        )
        gradd = jax.tree_util.tree_map(lambda p: -p, gradd)
        optd.update(
            gradd,
            grad=gradd,
            value_fn=loss_fn_d_split,
            value=lossd,
            H_loss_fn=H_loss_fn_d,
        )

        return loss, lossd

    return train_step


def run_optimizer_d(t, model, opt, disc, optd, num, name, epoch_print=100):
    loss_old = 1.0
    for epoch in range(1, num + 1):
        loss, lossd = t(model, opt, disc, optd)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}, lossd: {lossd}")
        if abs(loss) < ulp(1000) or abs(loss - loss_old) < ulp(1):
            break
        loss_old = loss
