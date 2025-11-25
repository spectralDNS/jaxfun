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
)
from jax import Array, lax
from sympy import Function
from sympy.printing.pretty.stringpict import prettyForm
from sympy.vector import VectorAdd

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import BaseTime, CoordSys
from jaxfun.galerkin import Chebyshev
from jaxfun.galerkin.tensorproductspace import (
    TensorProductSpace,
    VectorTensorProductSpace,
)
from jaxfun.utils.common import Domain, lambdify

from .embeddings import Embedding
from .nnspaces import KANMLPSpace, MLPSpace, PirateSpace, sPIKANSpace

default_kernel_init = nnx.initializers.glorot_normal()
default_bias_init = nnx.initializers.zeros_init()
default_rngs = nnx.Rngs(11)


class BaseModule(nnx.Module):
    """Base class for PINN modules."""

    def __hash__(self) -> int:
        # Use static hash for module (does not reflect a change of state)
        if not hasattr(self, "_hash"):
            self._hash = hash(nnx.graphdef(self))
        return self._hash


class RWFLinear(nnx.Module):
    """Linear layer with RWF (Random Weight Factorization) style scaling.

    Implements a linear transform y = x W + b with an additional learnable
    exponential scaling vector g applied to columns of W (per output
    feature) following RWF initialization ideas.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        use_bias: Whether to add bias.
        dtype: Computation dtype (promotes params+inputs if set).
        param_dtype: Dtype for parameter initialization.
        precision: JAX dot precision.
        kernel_init: Weight initializer.
        bias_init: Bias initializer.
        dot_general: Dot routine (default lax.dot_general).
        promote_dtype: Callable to promote dtypes of (inputs, kernel, bias, g).
        rngs: RNG container.
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
        """Apply linear transform (with column scaling) to inputs.

        Args:
            inputs: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
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


class KANLayer(nnx.Module):
    """Single Kolmogorov-Arnold (KAN) spectral expansion layer.

    Expands each input coordinate (or spectral feature in hidden layers)
    into spectral basis coefficients and applies a linear combination to
    produce output features.

    For the input layer, one spectral basis per coordinate is used. Hidden
    layers reuse a single basis (shared across channels).

    Args:
        in_features: Number of input channels/features.
        out_features: Number of output channels.
        spectral_size: Number of spectral modes per input.
        hidden: True if this is a hidden (spectral→spectral) layer.
        basespace: Spectral base class (e.g. Chebyshev.Chebyshev).
        domains: List of (a, b) domains for each input (only for input layer).
        system: Coordinate system (only needed for input layer mapping).
        dtype: Computation dtype.
        param_dtype: Parameter init dtype.
        precision: JAX dot precision.
        kernel_init: Weight initializer.
        dot_general: Dot routine (defaults to lax.dot_general).
        promote_dtype: Dtype promotion callable.
        rngs: RNG container.
    """

    __data__ = ("kernel",)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        spectral_size: int,
        *,
        hidden: bool = False,
        basespace: BaseSpace = Chebyshev.Chebyshev,
        domains: list[Domain] | None = None,
        system: CoordSys = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        dot_general: DotGeneralT = lax.dot_general,
        promote_dtype=dtypes.promote_dtype,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.spectral_size = spectral_size
        self.hidden = hidden
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.dot_general = dot_general
        self.promote_dtype = promote_dtype

        kernel_key = rngs.params()
        w = kernel_init(
            kernel_key, (in_features, spectral_size, out_features), param_dtype
        )
        y = jnp.logspace(0, -min(6, spectral_size), spectral_size)[None, :, None]
        self.kernel = nnx.Param(w * y)

        # Select subsystem(s) for per-dimension mapping (input layer only).
        subsystems = (
            [system.sub_system(i) if system else None for i in range(system.dims)]
            if (not hidden and system and system.dims > 1)
            else [system]
        )
        if system and in_features > system.dims and not hidden:
            # Transient extra coordinate has no subsystem mapping.
            subsystems += [None]

        domains = (
            domains if domains is not None else [(-1, 1) for _ in range(in_features)]
        )

        self.basespaces = (
            [
                basespace(spectral_size, domain=domains[i], system=subsystems[i])
                for i in range(in_features)
            ]
            if not hidden
            else [
                basespace(spectral_size, domain=Domain(-1, 1))
            ]  # tanh co-domain is [-1, 1]
        )

    def __call__(self, inputs: Array) -> Array:
        """Apply spectral expansion + linear projection.

        Args:
            inputs: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        kernel = self.kernel.value
        inputs, kernel = self.promote_dtype((inputs, kernel), dtype=self.dtype)

        if not self.hidden:
            # Expand each input dimension independently.
            T = [
                self.basespaces[j].eval_basis_functions(
                    self.basespaces[j].map_reference_domain(inputs[..., j])
                )
                for j in range(self.in_features)
            ]
        else:
            # Hidden layer, all with the same domain [-1, 1]: reuse one basis.
            T = [
                self.basespaces[0].eval_basis_functions(inputs[..., j])
                for j in range(self.in_features)
            ]

        return sum(
            [
                self.dot_general(
                    T[i],
                    kernel[i],
                    (((T[i].ndim - 1,), (0,)), ((), ())),
                    precision=self.precision,
                )
                for i in range(len(T))
            ]
        )


class MLP(BaseModule):
    """Standard multilayer perceptron ((weighted) Linear layers + activation)."""

    def __init__(
        self,
        V: BaseSpace,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        """Build an MLP from a function space description.

        Args:
            V: Function space (must supply in_size, out_size, hidden_size, act_fun).
            kernel_init: Kernel initializer (defaults glorot normal).
            bias_init: Bias initializer (defaults zeros).
            rngs: RNG container.
        """
        hidden_size = (
            V.hidden_size
            if isinstance(V.hidden_size, list | tuple)
            else [V.hidden_size]
        )
        linlayer = RWFLinear if V.weight_factorization else nnx.Linear
        self.linear_in = linlayer(
            V.in_size,
            hidden_size[0],
            rngs=rngs,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden = (
            nnx.List(
                linlayer(
                    hidden_size[i],
                    hidden_size[min(i + 1, len(hidden_size) - 1)],
                    rngs=rngs,
                    bias_init=bias_init,
                    kernel_init=kernel_init,
                    param_dtype=float,
                    dtype=float,
                )
                for i in range(len(hidden_size))
            )
            if isinstance(V.hidden_size, list | tuple)
            else []
        )
        self.linear_out = linlayer(
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
        """Return flattened parameter count."""
        st = nnx.state(self)
        return jax.flatten_util.ravel_pytree(st)[0].shape[0]

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input batch (N, in_size).

        Returns:
            Output batch (N, out_size).
        """
        x = self.act_fun(self.linear_in(x))
        for z in self.hidden:
            x = self.act_fun(z(x))
        return self.linear_out(x)


class PIModifiedBottleneck(nnx.Module):
    """Physics-inspired modified bottleneck residual mixing block.

    Applies three linear layers separated by nonlinear mixing with two
    auxiliary feature tensors (u, v) and a learnable global scaling alpha.

    Form (schematically):
        h1 = act(W1 x)
        h1 = h1 * u + (1 - h1) * v
        h2 = act(W2 h1)
        h2 = h2 * u + (1 - h2) * v
        h3 = act(W3 h2)
        out = alpha * h3 + (1 - alpha) * x

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer width.
        output_dim: Output dimension.
        nonlinearity: Initial alpha (0 => identity, 1 => full residual).
        act_fun: Activation function.
        rngs: RNG container.
    """

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
        """Forward pass with residual mixing.

        Args:
            x: Input tensor (N, in_dim).
            u: Auxiliary tensor (broadcast compatible with hidden dims).
            v: Auxiliary tensor (same shape as u).

        Returns:
            Mixed output tensor (N, output_dim).
        """
        identity = x

        x = self.act_fun(self.layer1(x))
        x = x * u + (1 - x) * v

        x = self.act_fun(self.layer2(x))
        x = x * u + (1 - x) * v

        x = self.act_fun(self.layer3(x))
        x = self.alpha * x + (1 - self.alpha) * identity

        return x


class PirateNet(BaseModule):
    """PirateNet: MLP with dual branch + bottleneck residual mixing.

    Features:
      * Fourier / periodic embeddings (optional) via Embedding
      * Two initial branches (u_net, v_net) combined by PIModifiedBottleneck
      * Optional learnable nonlinearity scaling

    Args:
        V: PirateSpace specification.
        kernel_init: Kernel initializer.
        bias_init: Bias initializer.
        rngs: RNG container.
    """

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
        self.hidden = nnx.List(
            PIModifiedBottleneck(
                in_dim=in_dim,
                hidden_dim=hidden_size[i],
                output_dim=in_dim,
                nonlinearity=V.nonlinearity,
                rngs=rngs,
                act_fun=V.act_fun_hidden,
            )
            for i in range(len(hidden_size))
        )

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
        """Return flattened parameter count."""
        st = nnx.split(self, nnx.Param)[1]
        return jax.flatten_util.ravel_pytree(st)[0].shape[0]

    def __call__(self, x: Array) -> Array:
        """Forward pass.

        Args:
            x: Input batch (N, in_dim_pre_embedding).

        Returns:
            Output batch (N, out_size).
        """
        x = self.embedder(x)
        u = self.act_fun(self.u_net(x))
        v = self.act_fun(self.v_net(x))

        for layer in self.hidden:
            x = layer(x, u, v)

        y = self.output_layer(x)
        return y


class SpectralModule(BaseModule):
    """Wrapper for a spectral function space (1D or tensor product)."""

    def __init__(
        self,
        basespace: BaseSpace,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,  # kept for uniform API
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize spectral kernel parameters.

        Args:
            basespace: Spectral basis object with .evaluate method.
            kernel_init: Kernel initializer.
            bias_init: Ignored (present for consistency).
            rngs: RNG container.
        """
        if basespace.dims == 1:
            w = kernel_init(rngs(), (1, basespace.num_dofs))
            # Spectral modes should decay - apply logscaled weighting
            x = jnp.logspace(0, -6, basespace.num_dofs)
            self.kernel = nnx.Param(w * x[None, :])

        elif basespace.dims == 2:
            w = kernel_init(rngs(), basespace.num_dofs)
            x = jnp.logspace(0, -6, basespace.num_dofs[-2])
            y = jnp.logspace(0, -6, basespace.num_dofs[-1])
            if isinstance(basespace, TensorProductSpace):
                self.kernel = nnx.Param(x[:, None] * y[None, :] * w)
            elif isinstance(basespace, VectorTensorProductSpace):
                self.kernel = nnx.Param((x[None, :, None] * y[None, None, :]) * w)

        self.space = basespace

    @property
    def dim(self) -> int:
        """Return number of spectral coefficients."""
        return self.space.dim

    @property
    def dims(self) -> int:
        """Return spatial dimensionality of the basespace."""
        return self.space.dims

    def __call__(self, x: Array) -> Array:
        """Evaluate spectral expansion at coordinates.

        Args:
            x: Coordinates (N, d).

        Returns:
            Values (N,) if d=1 else (N, rank+1).
        """
        if self.dims == 1:
            X = self.space.map_reference_domain(x)
            return self.space.evaluate(X, self.kernel.value[0])

        z = self.space.evaluate(x, self.kernel.value, True)
        if self.space.rank == 0:
            return jnp.expand_dims(z, -1)
        return z


class KANMLPModule(BaseModule):
    """Hybrid KAN input layer + MLP hidden/output layers."""

    def __init__(
        self,
        V: KANMLPSpace,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> None:
        self.kanspace = V
        hidden_size = (
            V.hidden_size
            if isinstance(V.hidden_size, list | tuple)
            else [V.hidden_size]
        )
        self.act_fun = V.act_fun
        self.layer_in = KANLayer(
            V.in_size,
            hidden_size[0],
            V.spectral_size,
            hidden=False,
            basespace=V.basespace,
            domains=V.domains,
            system=V.system,
            rngs=rngs,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden = (
            nnx.List(
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
            )
            if isinstance(V.hidden_size, list | tuple)
            else []
        )
        if hidden_size[-1] > 1:
            self.layer_out = RWFLinear(
                hidden_size[-1],
                V.out_size,
                rngs=rngs,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=float,
                dtype=float,
            )
        else:
            self.layer_out = lambda x: x

    @property
    def dim(self) -> int:
        """Return flattened parameter count."""
        st = nnx.split(self, nnx.Param)[1]
        return jax.flatten_util.ravel_pytree(st)[0].shape[0]

    def __call__(self, x: Array) -> Array:
        """Forward pass through KAN + MLP stack.

        Args:
            x: Input batch (N, in_size).

        Returns:
            Output batch (N, out_size).
        """
        x = self.act_fun(self.layer_in(x))
        for z in self.hidden:
            x = self.act_fun(z(x))
        return self.layer_out(x)


class sPIKANModule(BaseModule):
    """Pure spectral KAN in every layer (input, hidden, output)."""

    def __init__(
        self,
        V: sPIKANSpace,
        *,
        kernel_init: Initializer = default_kernel_init,
        rngs: nnx.Rngs,
    ) -> None:
        self.kanspace = V
        hidden_size = (
            V.hidden_size
            if isinstance(V.hidden_size, list | tuple)
            else [V.hidden_size]
        )
        spectral_size = (
            V.spectral_size
            if isinstance(V.spectral_size, list | tuple)
            else [V.spectral_size for _ in range(len(hidden_size) + 1)]
        )
        self.act_fun = V.act_fun
        self.layer_in = KANLayer(
            V.in_size,
            hidden_size[0],
            spectral_size[0],
            hidden=False,
            basespace=V.basespace,
            domains=V.domains,
            system=V.system,
            rngs=rngs,
            kernel_init=kernel_init,
            param_dtype=float,
            dtype=float,
        )
        self.hidden = (
            nnx.List(
                KANLayer(
                    hidden_size[i],
                    hidden_size[min(i + 1, len(hidden_size) - 1)],
                    spectral_size[i + 1],
                    hidden=True,
                    basespace=V.basespace,
                    system=V.system,
                    rngs=rngs,
                    kernel_init=kernel_init,
                    param_dtype=float,
                    dtype=float,
                )
                for i in range(len(hidden_size))
            )
            if isinstance(V.hidden_size, list | tuple)
            else []
        )

        if hidden_size[-1] > 1:
            self.layer_out = KANLayer(
                hidden_size[-1],
                V.out_size,
                spectral_size[-1],
                hidden=True,
                basespace=V.basespace,
                system=V.system,
                rngs=rngs,
                kernel_init=kernel_init,
                param_dtype=float,
                dtype=float,
            )
        else:
            self.layer_out = lambda x: x

    @property
    def dim(self) -> int:
        """Return flattened parameter count."""
        st = nnx.state(self)
        return jax.flatten_util.ravel_pytree(st)[0].shape[0]

    def __call__(self, x: Array) -> Array:
        """Forward pass through fully spectral KAN stack.

        Args:
            x: Input coordinates/features (N, in_size).

        Returns:
            Output batch (N, out_size).
        """
        x = self.act_fun(self.layer_in(x))
        for z in self.hidden:
            x = self.act_fun(z(x))
        return self.layer_out(x)


class FlaxFunction(Function):
    """Symbolic wrapper for a neural/spectral module bound to coordinates.

    Creates a SymPy Function carrying:
      * functionspace (metadata + dims/rank)
      * module (the nnx.Module used for evaluation)
      * symbolic arguments (coordinates, optional time, space name)

    Supports vector fields (rank=1).
    """

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
        """Return component function for a vector-valued space."""
        return FlaxFunction(
            self.functionspace[i], name=self.name[i], module=self.module, rngs=self.rngs
        )

    @property
    def rank(self):
        """Return tensor rank of the represented field."""
        return self.functionspace.rank

    @property
    def dim(self):
        """Return flattened parameter count of underlying module."""
        return self.module.dim

    @staticmethod
    def get_flax_module(
        V,
        *,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        rngs: nnx.Rngs,
    ) -> MLP | PirateNet | SpectralModule:
        """Instantiate appropriate nnx module given a function space."""
        if isinstance(V, MLPSpace):
            return MLP(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        elif isinstance(V, PirateSpace):
            return PirateNet(V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        elif isinstance(V, KANMLPSpace):
            return KANMLPModule(
                V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
            )
        elif isinstance(V, sPIKANSpace):
            return sPIKANModule(V, kernel_init=kernel_init, rngs=rngs)
        return SpectralModule(
            V, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs
        )

    def get_args(self, Cartesian=True):
        """Return symbolic arguments (Cartesian or base scalars + time)."""
        if Cartesian:
            return self.args[:-1]
        V = self.functionspace
        s = V.system.base_scalars()
        return s + (self.t,) if V.is_transient else s

    def doit(self, **hints: dict) -> sp.Expr:
        """Return an evaluated SymPy expression (vector or scalar).

        For rank 0: returns a scalar Function placeholder.
        For rank 1: returns a VectorAdd assembling components.
        """
        V = self.functionspace
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
            s = V.system.base_scalars()
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
        """Map computational coordinates to Cartesian physical domain.

        Args:
            xs: Computational coordinates (N, dims).

        Returns:
            Physical coordinates (N, dims).
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
        """Evaluate underlying module; flatten scalar output if rank 0."""
        y = self.module(x)
        if self.rank == 0:
            return y[:, 0]
        return y


class Comp(nnx.Module):
    """Module composing multiple FlaxFunctions in parallel (stack outputs)."""

    def __init__(self, *flaxfunctions: FlaxFunction) -> None:
        """Store list of FlaxFunctions and register modules as attributes.

        Args:
            *flaxfunctions: One or more FlaxFunction instances.
        """
        self.flaxfunctions = list(flaxfunctions)
        self.mod_index = {
            str(hash(p.module)): i for i, p in enumerate(self.flaxfunctions)
        }
        self.data = nnx.List([p.module for p in self.flaxfunctions])

    @property
    def dim(self) -> int:
        """Return total flattened parameter count across all functions."""
        return sum([f.module.dim for f in self.flaxfunctions])

    def __call__(self, x: Array) -> Array:
        """Evaluate and horizontally stack all component module outputs.

        Args:
            x: Input batch.

        Returns:
            Concatenated outputs (N, sum(out_sizes)).
        """
        return jnp.hstack([f.module(x) for f in self.flaxfunctions])
