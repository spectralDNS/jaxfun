from collections.abc import Callable
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
    # PromoteDtypeFn,
)
from jax import Array, lax
from sympy import Function
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import VectorAdd

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import BaseTime
from jaxfun.pinns.embeddings import Embedding
from jaxfun.pinns.nnspaces import MLPSpace, PirateSpace
from jaxfun.utils.common import lambdify

default_kernel_init = nnx.initializers.glorot_normal()
default_bias_init = nnx.initializers.zeros_init()
default_rngs = nnx.Rngs(101)


# Differs from jaxfun.utils.common.jacn in the last if else
def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    for i in range(k):
        fun = jax.jacrev(fun) if i % 2 else jax.jacfwd(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun


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
        promote_dtype=dtypes.promote_dtype,
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
        V: BaseSpace,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        """Multilayer perceptron

        Args:
            V: Functionspace with detailed layer structure for the MLP
            rngs: Seed
            kernel_init (optional): Initializer for kernel. Defaults
                to default_kernel_init.
            bias_init (optional): Initializer for bias. Defaults to default_bias_init.
        """
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
        self.hidden = (
            [
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
            if isinstance(V.hidden_size, list | tuple)
            else []
        )
        self.linear_out = RWFLinear(
            hidden_size[-1],
            V.out_size,
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.act_fun = V.act_fun

    @property
    def dim(self) -> int:
        st = nnx.split(self, nnx.Param)[1]
        return jax.flatten_util.ravel_pytree(st)[0].shape[0]

    def __call__(self, x: Array) -> Array:
        x = self.act_fun(self.linear_in(x))
        for z in self.hidden:
            x = self.act_fun(z(x))
        return self.linear_out(x)


class PIModifiedBottleneck(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        nonlinearity: float,
        act_fun: Callable[[Array], Array] = nnx.tanh,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.alpha = nnx.Param(jnp.array(nonlinearity).reshape((1,)))

        self.layer1 = RWFLinear(
            in_dim, hidden_dim, rngs=rngs, dtype=float, param_dtype=float
        )
        self.layer2 = RWFLinear(
            hidden_dim, hidden_dim, rngs=rngs, dtype=float, param_dtype=float
        )
        self.layer3 = RWFLinear(
            hidden_dim, output_dim, rngs=rngs, dtype=float, param_dtype=float
        )

        self.act_fun = act_fun

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
            in_dim += len(V.periodicity["axis"])
        if V.fourier_emb is not None:
            in_dim = V.fourier_emb["embed_dim"]

        self.act_fun = V.act_fun

        self.u_net = RWFLinear(
            in_dim,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.v_net = RWFLinear(
            in_dim,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden = [
            PIModifiedBottleneck(
                in_dim=in_dim,
                hidden_dim=hidden_size[i],
                output_dim=in_dim,
                nonlinearity=V.nonlinearity,
                rngs=rngs,
                act_fun=V.act_fun_hidden,
            )
            for i in range(len(hidden_size))
        ]

        if V.pi_init is not None:
            raise NotImplementedError("Least squares initialization not implemented")
        else:
            self.output_layer = RWFLinear(
                in_dim,
                V.out_size,
                rngs=rngs,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=float,
                dtype=float,
            )

    @property
    def dim(self) -> int:
        st = nnx.split(self, nnx.Param)[1]
        return jax.flatten_util.ravel_pytree(st)[0].shape[0]

    def __call__(self, x: Array) -> Array:
        x = self.embedder(x)
        u = self.act_fun(self.u_net(x))
        v = self.act_fun(self.v_net(x))

        for layer in self.hidden:
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
        V: BaseSpace,
        name: str,
        *,
        module: nnx.Module = None,
        fun_str: str = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs = default_rngs,
    ) -> Function:
        coors = V.system
        args = list(coors._cartesian_xyz)
        t = BaseTime(V.system)
        args = args + [t] if V.is_transient else args
        args = args + [sp.Symbol(V.name)]
        obj = Function.__new__(cls, *args)
        obj.functionspace = V
        obj.t = t
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
        obj.rngs = rngs
        return obj

    def __getitem__(self, i: int):
        return FlaxFunction(
            self.functionspace[i], name=self.name[i], module=self.module, rngs=self.rngs
        )

    @property
    def rank(self):
        return self.functionspace.rank

    @property
    def dim(self):
        return self.module.dim

    @staticmethod
    def get_flax_module(
        V,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> MLP | PirateNet | SpectralModule:
        if isinstance(V, MLPSpace):
            return MLP(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        elif isinstance(V, PirateSpace):
            return PirateNet(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        return SpectralModule(
            V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def get_args(self, Cartesian=True):
        if Cartesian:
            return self.args[:-1]
        V = self.functionspace
        s = V.system.base_scalars()
        return s + (self.t,) if V.is_transient else s

    def doit(self, **hints: dict) -> sp.Expr:
        V = self.functionspace
        s = V.system.base_scalars()
        args = self.get_args(Cartesian=False)

        if V.rank == 0:
            return Function(
                self.fun_str,
                global_index=0,
                functionspace_name=V.name,
                rank_parent=V.rank,
                module=self.module,
                argument=2,
            )(*args)

        if V.rank == 1:
            b = V.system.base_vectors()
            return VectorAdd.fromiter(
                Function(
                    self.fun_str + "_" + s[i].name,
                    global_index=i,
                    functionspace_name=V.name,
                    rank_parent=V.rank,
                    module=self.module,
                    argument=2,
                )(*args)
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
        name = "\033[1m%s\033[0m" % (self.name,) if self.rank == 1 else self.name  # noqa: UP031
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.args[:-1]]),
                "; ",
                self.args[-1].name,
                ")",
            )
        )

    def _latex(self, printer: Any = None) -> str:
        name = r"\mathbf{ {%s} }" % (self.name,) if self.rank == 1 else self.name  # noqa: UP031
        return "".join(
            (
                name,
                "(",
                ", ".join([i.name for i in self.args[:-1]]),
                "; ",
                self.args[-1].name,
                ")",
            )
        )

    def _pretty(self, printer: Any = None) -> str:
        return prettyForm(self.__str__())

    def _sympystr(self, printer: Any) -> str:
        return self.__str__()

    def __call__(self, x):
        y = self.module(x)
        if self.rank == 0:
            return y[:, 0]
        return y


class Comp(nnx.Module):
    def __init__(self, *flaxfunctions: FlaxFunction) -> None:
        """Collection of FlaxFunctions to be evaluated and stacked

        Args:
            flaxfunctions: FlaxFunctions to be evaluated and stacked
        """
        self.flaxfunctions = list(flaxfunctions)
        [setattr(self, str(id(p.module)), p.module) for p in flaxfunctions]

    @property
    def dim(self) -> int:
        return sum([f.module.dim for f in self.flaxfunctions])

    def __call__(self, x: Array) -> Array:
        return jnp.hstack([f.module(x) for f in self.flaxfunctions])
