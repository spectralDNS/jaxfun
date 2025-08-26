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
    # PromoteDtypeFn,
)
from jax import Array, lax
from sympy import Function
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import VectorAdd

from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import BaseScalar, BaseTime, CoordSys
from jaxfun.pinns.embeddings import Embedding
from jaxfun.typing import LSQR_Tuple
from jaxfun.utils.common import lambdify, ulp

default_kernel_init = nnx.initializers.glorot_normal()
default_bias_init = nnx.initializers.zeros_init()


# Differs from jaxfun.utils.common.jacn in the last if else
def jacn(fun: Callable[[float], Array], k: int = 1) -> Callable[[Array], Array]:
    for i in range(k):
        fun = jax.jacrev(fun) if i % 2 else jax.jacfwd(fun)
    return jax.vmap(fun, in_axes=0, out_axes=0) if k > 0 else fun


class NNSpace(BaseSpace):
    """Neural network functionspace"""

    def __init__(
        self,
        dims: int = 1,
        rank: int = 0,
        transient: bool = False,
        system: CoordSys = None,
        name: str = "NN",
    ) -> None:
        """Class for the structure of a neural network functionspace

        Args:
            dims: Spatial dimensions. Defaults to 1.
            rank:
                Scalars, vectors and dyadics have rank if 0, 1 and 2, respectively.
                Defaults to 0.
            transient:  Whether to include the variable time or not. Defaults to False.
            system:
                Coordinate system. Defaults to None, in which case the coordinate
                system will be Cartesian
            name: Name of NN space

        """
        from jaxfun.arguments import CartCoordSys, x, y, z

        self.in_size = dims + int(transient)
        self.out_size = dims**rank
        self.dims = dims
        self.rank = rank
        self.transient = transient
        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[dims])
            if system is None
            else system
        )
        BaseSpace.__init__(self, system, name)

    @property
    def is_transient(self):
        return self.transient

    def base_variables(self) -> list[BaseScalar | BaseTime]:
        """Return the base variables, including time if transient."""
        if self.transient:
            return self.system.base_scalars() + (self.system.base_time(),)
        else:
            return self.system.base_scalars()

class MLPSpace(NNSpace):
    """Multilayer perceptron functionspace"""

    def __init__(
        self,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        transient: bool = False,
        offset: int = 0,
        act_fun: Callable[[Array], Array] = nnx.tanh,
        *,
        name: str,
    ) -> None:
        """Class for the structure of an MLP

        Args:
            hidden_size:
                If list of integers, like hidden_size = [X, Y, Z], then there will be
                len(hidden_size) hidden layer of size X, Y and Z, respectively.
                If integer, like hidden_size = X, then there will be no hidden layers,
                but the size of the weights in the input layer will be dims * X and the
                output will be of shape X * self.out_size
            dims: Spatial dimensions. Defaults to 1.
            rank:
                Scalars, vectors and dyadics have rank if 0, 1 and 2, respectively.
                Defaults to 0.
            system:
                Coordinate system. Defaults to None, in which case the coordinate
                system will be Cartesian
            transient:
                Whether to include the variable time or not. Defaults to False.
            offset: If part of a CompositeMLP, then the offset tells how many
                outputs there are before this space. The accumulated sum of all
                out_sizes of all prior spaces. Defaults to 0.
            act_fun:
                Activation function for all except the output layer
            name: Name of MLPSpace

        """        
        NNSpace.__init__(self, dims, rank, transient, system, name)
        self.hidden_size = hidden_size
        self.offset = offset
        self.act_fun = act_fun



MLPVectorSpace = partial(MLPSpace, rank=1)


class PirateSpace(NNSpace):
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
        act_fun: Callable[[Array], Array] = nnx.tanh,
        act_fun_hidden: Callable[[Array], Array] = nnx.tanh,
        # PirateNet specific parameters
        nonlinearity: float = 0.0,
        periodicity: dict | None = None,
        fourier_emb: dict | None = None,
        pi_init: jnp.ndarray | None = None,
    ) -> None:
        
        NNSpace.__init__(self, dims, rank, transient, system, name)

        # PirateSpace requires at least one hidden layer, so change integer hidden_size to [hidden_size]
        self.hidden_size = (
            hidden_size if isinstance(hidden_size, list | tuple) else [hidden_size]
        )
        self.offset = offset
        self.act_fun = act_fun
        self.act_fun_hidden = act_fun_hidden
        
        self.nonlinearity = nonlinearity
        self.periodicity = periodicity
        self.fourier_emb = fourier_emb
        self.pi_init = pi_init


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
        self.mlp = newspaces
        self.in_size = self.mlp[0].in_size
        self.hidden_size = self.mlp[0].hidden_size
        self.out_size = sum([p.out_size for p in self.mlp])

    def __getitem__(self, i: int):
        return self.mlp[i]

    def __len__(self):
        return len(self.mlp)


# Note: We should probably get rid of this?
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


class Count(nnx.Variable):
    pass


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
        V: BaseSpace | CompositeMLP,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        """Multilayer perceptron

        Args:
            V: Functionspace with detailed layer structure for the MLP
            rngs: Seed
            kernel_init (optional): Initializer for kernel. Defaults to default_kernel_init.
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
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs = nnx.Rngs(101),
    ) -> Function:
        from jaxfun.coordinates import BaseTime

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
            if module is None  # and not isinstance(V, CompositeNetwork)
            else module
        )
        obj.name = name
        obj.fun_str = fun_str if fun_str is not None else name
        obj.argument = 2
        obj.rngs = rngs
        if isinstance(V, CompositeMLP):
            assert len(name) == len(V)
        return obj

    def __getitem__(self, i: int):
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
        if isinstance(V, MLPSpace | CompositeMLP):
            return MLP(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        elif isinstance(V, PirateSpace | CompositeNetwork):
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
        from jaxfun.arguments import functionspacedict

        V = self.functionspace
        functionspacedict[V.name] = V
        s = V.system.base_scalars()
        args = self.get_args(Cartesian=False)

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
            )(*args)

        if V.rank == 1:
            b = V.system.base_vectors()
            return VectorAdd.fromiter(
                Function(
                    self.fun_str + "_" + s[i].name,
                    global_index=V.offset + i,
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
        V = self.functionspace
        if self.rank == 0:
            return y[:, V.offset]
        elif self.rank == 1:
            return y[:, V.offset : V.offset + V.out_size]
        return y


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
    V = f.functionspace
    offset = V.offset if f.rank == 0 else slice(V.offset, V.offset + V.out_size)
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
        self, f: sp.Expr, x: Array, target: Array = 0, weights: Array = 1
    ) -> None:
        s = get_args(f.doit())
        t, expr = expand(f)
        self.eqs = [get_fn(h, s) for h in expr]
        self.x = x
        # Place all terms without flaxfunctions in the target, because these will not need to be computed more than once
        self.target = target
        if len(t.free_symbols) > 0:
            self.target = target - lambdify(s, t, modules="jax")(*x.T)
        elif t != 0:
            self.target = target - t
        self.target = jnp.squeeze(self.target)
        self.weights = weights

    def __call__(self, Js) -> Array:
        return sum([eq(self.x, Js) for eq in self.eqs]) - self.target


class LSQR:
    """Least squares loss function"""

    def __init__(self, *fs: LSQR_Tuple, alpha: float = 0.9):
        """The least squares method is to compute the loss over all input equations at
        all collocation points. The equations are all defined with their own points

        Args:
            fs:
                tuples, where the latter contains the subproblems that
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
            >>> from flax import nnx
            >>> import optax
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


def get_fn(f: sp.Expr, s: tuple[BaseScalar]) -> Callable[[Array, dict], Array]:
    """Return Sympy Expr as function evaluated by points and gradients

    Args:
        f (sp.Expr)
        w (FlaxFunction)

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
    loss_fn: LSQR,
    model: nnx.Module,
    opt: nnx.Optimizer,
    num: int,
    name: str,
    epoch_print: int = 100,
    abs_limit_loss: float = ulp(1000),
    abs_limit_change: float = ulp(100),
    print_final_loss: bool = False,
    update_global_weights: int = -1,
    print_global_weights: bool = False,
):
    train_step = train(loss_fn)
    loss_old = 1.0
    for epoch in range(1, num + 1):
        loss = train_step(model, opt)
        if epoch % epoch_print == 0:
            print(f"Epoch {epoch} {name}, loss: {loss}")
        if abs(loss) < abs_limit_loss or abs(loss - loss_old) < abs_limit_change:
            break
        loss_old = loss
        if update_global_weights > 0 and epoch % update_global_weights == 0:
            loss_fn.update_global_weights(model)
            if print_global_weights:
                print("Global weights", loss_fn.global_weights)
    if print_final_loss:
        print(f"Final loss for {name}: {loss}")
