import copy
from collections.abc import Callable
from functools import partial
from numbers import Number
from typing import Any

import jax
import jax.numpy as jnp
import sympy as sp
from flax import nnx
from flax.nnx.nn import dtypes
from flax.typing import (
    DotGeneralT,
    Dtype,
    Initializer,
    PrecisionLike,
    PromoteDtypeFn,
)
from jax import Array, lax
from sympy import Function
from sympy.printing.pretty.stringpict import prettyForm

from jaxfun.arguments import FlaxBasisFunction
from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, CoordSys
from jaxfun.pinns.embeddings import Embedding
from jaxfun.utils.common import jacn, lambdify, ulp

default_kernel_init = nnx.initializers.glorot_normal()
default_bias_init = nnx.initializers.zeros_init()


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


class PirateSpace(BaseSpace):
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


class CompositeNetwork:
    def __init__(self, spaces: tuple[BaseSpace, ...], name: str = "C") -> None:
        offset = 0
        # TODO: should refactor name from mlp to something more general
        self.mlp = []
        self.name = name  # Prefix, not name
        self.system = spaces[0].system

        has_pirate = any(isinstance(s, PirateSpace) for s in spaces)
        if has_pirate:
            p_params = {"period": (), "axis": (), "trainable": ()}
            f_params = {"embed_dim": 0, "embed_scale": 1.0}
            self.nonlinearity = 0.0

        for i, space in enumerate(spaces):
            # Is there any point in initializing a brand new space,
            # instead of just correcting the existing one?
            # With this, it's easier to be agnostic about the space type.
            space.system = self.system
            space.name = f"{name}{space.name}_{i}"
            space.offset = offset

            if isinstance(space, PirateSpace):
                if space.periodicity is not None:
                    prev_p = space.periodicity
                    p_params["period"] += prev_p["period"]
                    p_params["axis"] += (a + offset for a in prev_p["axis"])
                    p_params["trainable"] += prev_p["trainable"]
                if space.fourier_emb is not None:
                    f_params["embed_dim"] += space.fourier_emb["embed_dim"]
                    f_params["embed_scale"] = space.fourier_emb["embed_scale"]
                self.nonlinearity = max(self.nonlinearity, space.nonlinearity)

            offset += space.out_size
            self.mlp.append(space)
        self.in_size = self.mlp[0].in_size

        if has_pirate:
            self.periodicity = p_params if p_params["axis"] else None
            if f_params["embed_dim"] > 0:
                f_params["in_dim"] = self.in_size
                self.fourier_emb = f_params
            else:
                self.fourier_emb = None
            self.pi_init = None

        self.hidden_size = self.mlp[0].hidden_size
        self.out_size = sum([p.out_size for p in self.mlp])

    def __getitem__(self, i: int) -> BaseSpace:
        return self.mlp[i]

    def __len__(self) -> int:
        return len(self.mlp)


class RWFLinear(nnx.Module):
    """A linear transformation applied over the last dimension of the input.

    Args:
      in_features: the number of input features.
      out_features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      dot_general: dot product function.
      promote_dtype: function to promote the dtype of the arrays to the desired
        dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
        and a ``dtype`` keyword argument, and return a tuple of arrays with the
        promoted dtype.
      rngs: rng key.
    """

    __data__ = ("kernel", "scaling", "bias")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        dot_general: DotGeneralT = lax.dot_general,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: nnx.Rngs,
    ):
        kernel_key = rngs.params()
        w = kernel_init(kernel_key, (in_features, out_features), param_dtype)
        scaling_key = rngs.params()
        # Use RWF params from https://arxiv.org/pdf/2507.08972
        scaling_init = nnx.initializers.normal(0.1)
        g = 1.0 + scaling_init(scaling_key, (out_features,), param_dtype)
        self.g = nnx.Param(jnp.exp(g))
        self.kernel = nnx.Param(w / g)

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            bias_key = rngs.params()
            self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general
        self.promote_dtype = promote_dtype

    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.kernel.value
        bias = self.bias.value if self.bias is not None else None
        g = self.g.value

        inputs, kernel, bias, g = self.promote_dtype(
            (inputs, kernel, bias, g), dtype=self.dtype
        )
        weights = g * kernel
        y = self.dot_general(
            inputs,
            weights,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        assert self.use_bias == (bias is not None)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MLP(nnx.Module):
    def __init__(
        self,
        V: BaseSpace | CompositeMLP,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        hidden_size = (
            V.hidden_size
            if isinstance(V.hidden_size, list | tuple)
            else [V.hidden_size]
        )
        self.linear_in = RWFLinear(
            V.in_size,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden = [
            RWFLinear(
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
        self.linear_out = RWFLinear(
            hidden_size[-1],
            V.out_size,
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )

    def __call__(self, x: Array) -> Array:
        x = nnx.swish(self.linear_in(x))
        for z in self.hidden:
            x = nnx.swish(z(x))
        return self.linear_out(x)


class PIModifiedBottleneck(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.alpha = nnx.Param(jnp.array(nonlinearity).reshape((1,)))

        self.layer1 = nnx.Linear(
            in_dim, hidden_dim, rngs=rngs, dtype=float, param_dtype=float
        )
        self.layer2 = nnx.Linear(
            hidden_dim, hidden_dim, rngs=rngs, dtype=float, param_dtype=float
        )
        self.layer3 = nnx.Linear(
            hidden_dim, output_dim, rngs=rngs, dtype=float, param_dtype=float
        )

        self.act_fun = nnx.tanh

    def __call__(self, x: Array, u: Array, v: Array) -> Array:
        identity = x

        x = self.act_fun(self.layer1(x))
        x = x * u + (1 - x) * v

        x = self.act_fun(self.layer2(x))
        x = x * u + (1 - x) * v

        x = self.act_fun(self.layer3(x))
        x = self.alpha * x + (1 - self.alpha) * identity

        return x


class PirateNet(nnx.Module):
    def __init__(
        self,
        V: PirateSpace | CompositeNetwork,
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
            in_dim += len(V.periodicity["axis"])
        if V.fourier_emb is not None:
            in_dim = V.fourier_emb["embed_dim"]

        # print(rngs, V)

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
            # in_dim = hidden_size[i - 1] if i > 0 else in_dim
            layer = PIModifiedBottleneck(
                in_dim=in_dim,
                hidden_dim=hidden_size[i],
                output_dim=in_dim,
                nonlinearity=V.nonlinearity,
                rngs=rngs,
            )
            self.hidden_layers.append(layer)

        if V.pi_init is not None:
            raise NotImplementedError("Least squares initialization not implemented")
        else:
            self.output_layer = nnx.Linear(
                in_dim,
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

        return y


class SpectralModule(nnx.Module):
    def __init__(
        self,
        basespace: BaseSpace,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
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
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs = None,
    ) -> None:
        self.functionspace = V

        self.module = (
            self.get_flax_module(
                V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
            )
            if module is None  # and not isinstance(V, CompositeNetwork)
            else module
        )
        # print(self.module, V, name, rngs)
        self.rngs = rngs

        self.name = name
        if isinstance(V, CompositeMLP | CompositeNetwork):
            assert len(name) == len(V)

    def __new__(
        cls,
        V: BaseSpace | CompositeMLP,
        name: str,
        *,
        module: nnx.Module = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs = None,
    ) -> Function:
        # print("Creating FlaxFunction", V, name, module, rngs)
        coors = V.system
        obj = Function.__new__(cls, *(coors._cartesian_xyz + [sp.Dummy(V.name)]))
        return obj

    def __getitem__(self, i: int):
        # print("FlaxFunction __getitem__", i, self.functionspace[i])
        return FlaxFunction(
            self.functionspace[i], name=self.name[i], module=self.module, rngs=self.rngs
        )

    @property
    def rank(self):
        return (
            None
            if isinstance(self.functionspace, CompositeMLP | CompositeNetwork)
            else self.functionspace.rank
        )

    @staticmethod
    def get_flax_module(
        V,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> MLP | PirateNet | SpectralModule:
        if isinstance(V, MLPSpace | CompositeMLP):
            return MLP(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        elif isinstance(V, PirateSpace | CompositeNetwork):
            return PirateNet(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        return SpectralModule(
            V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def doit(self, **hints: dict) -> sp.Expr:
        from jaxfun.arguments import FlaxBasisFunction

        V = self.functionspace
        if isinstance(V, CompositeMLP | CompositeNetwork):
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
        name = "\033[1m%s\033[0m" % (self.name,) if self.rank == 1 else self.name  # noqa: UP031
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
    def __init__(self, fs: tuple[tuple[sp.Expr, Array, Array | None], ...]) -> None:
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

    def update_gradients(self, model) -> None:
        # print(model)
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


class AdaptiveLSQR(LSQR):
    def __init__(
        self,
        fs: tuple[tuple[sp.Expr, Array, Array | None], ...],
        momentum: float = 0.9,
    ) -> None:
        super().__init__(fs)
        self.weights = jnp.ones(len(self.residuals))
        self.momentum = momentum

    def update_weights(self): ...

    def __call__(self, model):
        self.update_gradients(model)
        residuals = [
            (eq.weights * eq(model, self.Js) ** 2).mean() for eq in self.residuals
        ]
        return sum(residuals), jax.tree_util.tree_map(jnp.sqrt, residuals)


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
    # print(Js)
    # print(type(Js[(id(x), k)]))
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

        # print("Model in TrainStep", model)
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
        # if abs(loss) < abs_limit_loss or abs(loss - loss_old) < abs_limit_change:
        #     break
        loss_old = loss
    print(f"Final loss for {name}: {loss}")


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
