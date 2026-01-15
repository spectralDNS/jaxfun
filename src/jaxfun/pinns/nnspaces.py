from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from flax import nnx

from jaxfun.basespace import BaseSpace
from jaxfun.coordinates import (
    BaseScalar,
    BaseTime,
    CoordSys,
)
from jaxfun.galerkin import Chebyshev
from jaxfun.typing import Array
from jaxfun.utils.common import Domain


class NNSpace(BaseSpace):
    """Neural network function space base class.

    Provides a common interface for neural function spaces (MLP, Pirate,
    KAN / sPIKAN). Handles bookkeeping of input/output sizes, spatial
    dimensionality, tensor rank and optional time coordinate.

    Args:
        dims: Number of spatial dimensions (1-3).
        rank: Tensor rank of the field: 0=scalar, 1=vector, 2=dyadic.
        transient: If True, time is appended as an extra input coordinate.
        system: Optional coordinate system. If None, a Cartesian system is
            created automatically for the given dims.
        name: Name of the space.

    Attributes:
        in_size: Size of the input feature vector (dims + time if transient).
        out_size: Flattened output size (dims**rank for basic tensor shapes).
        dims: Spatial dimensionality.
        rank: Tensor rank.
        is_transient: Whether time is included.
        system: Underlying coordinate system.
    """

    def __init__(
        self,
        dims: int = 1,
        rank: int = 0,
        transient: bool = False,
        system: CoordSys = None,
        name: str = "NN",
    ) -> None:
        """Initialize a neural network function space."""
        from jaxfun.coordinates import CartCoordSys, x, y, z

        self.in_size = dims + int(transient)
        self.out_size = dims**rank
        self.dims = dims
        self.rank = rank
        self.is_transient = transient
        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[dims])
            if system is None
            else system
        )
        BaseSpace.__init__(self, system, name)

    def base_variables(self) -> tuple[BaseScalar | BaseTime, ...]:
        """Return base variables (add time if transient).

        Returns:
            Tuple of spatial BaseScalar objects, plus BaseTime if transient.
        """
        if self.is_transient:
            return self.system.base_scalars() + (self.system.base_time(),)
        else:
            return self.system.base_scalars()


class MLPSpace(NNSpace):
    """Multilayer perceptron function space specification.

    Defines the structural metadata (hidden sizes, activation) needed to
    instantiate an MLP module later via a dispatcher (e.g. FlaxFunction).

    Args:
        hidden_size: Layer sizes. If a list, each entry is a hidden layer size.
            If an int, interpreted as width of a single linear projection
            (no intermediate hidden layers).
        dims: Number of spatial dimensions.
        rank: Tensor rank of output (0, 1, or 2).
        system: Optional coordinate system (defaults to Cartesian).
        transient: If True, time is appended as input coordinate.
        act_fun: Activation function used for all hidden layers.
        weight_factorization: Optional weight factorization method.
        name: Space name.
    """

    def __init__(
        self,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        transient: bool = False,
        act_fun: Callable[[Array], Array] = nnx.tanh,
        weight_factorization: bool = False,
        *,
        name: str,
    ) -> None:
        """Initialize MLPSpace metadata."""
        NNSpace.__init__(self, dims, rank, transient, system, name)
        self.hidden_size = hidden_size
        self.weight_factorization = weight_factorization
        self.act_fun = act_fun


MLPVectorSpace = partial(MLPSpace, rank=1)


class PirateSpace(NNSpace):
    """PirateNet function space (MLP variant with feature embeddings).

    Extends a vanilla MLP space by adding:
      * Separate activation for first hidden layer(s)
      * Optional periodic feature embeddings
      * Optional Fourier embeddings
      * Learnable nonlinearity scaling (pi parameters)

    Args:
        hidden_size: Hidden layer configuration (list or single int).
        dims: Number of spatial dimensions.
        rank: Tensor rank of output.
        system: Optional coordinate system.
        name: Space name.
        transient: If True, include time as input.
        act_fun: Activation for final (and possibly intermediate) layers.
        act_fun_hidden: Activation for internal hidden layers (if different).
        nonlinearity: Scaling factor for special nonlinear transforms.
        periodicity: Dict configuring periodic embeddings (e.g. {'period': (...,)}).
        fourier_emb: Dict configuring Fourier embeddings (e.g. {'embed_dim': 8}).
        pi_init: Optional initial array for pi parameters.
    """

    def __init__(
        self,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        name: str = "PirateNet",
        transient: bool = False,
        act_fun: Callable[[Array], Array] = nnx.tanh,
        act_fun_hidden: Callable[[Array], Array] = nnx.tanh,
        # PirateNet specific parameters
        nonlinearity: float = 0.0,
        periodicity: dict | None = None,
        fourier_emb: dict | None = None,
        pi_init: jnp.ndarray | None = None,
    ) -> None:
        """Initialize PirateSpace metadata."""
        NNSpace.__init__(self, dims, rank, transient, system, name)

        self.hidden_size = (
            hidden_size if isinstance(hidden_size, list | tuple) else [hidden_size]
        )
        self.act_fun = act_fun
        self.act_fun_hidden = act_fun_hidden

        self.nonlinearity = nonlinearity
        self.periodicity = periodicity
        self.fourier_emb = fourier_emb
        self.pi_init = pi_init


class KANMLPSpace(NNSpace):
    """Hybrid Kolmogorov-Arnold (KAN) + MLP function space.

    The input transformation applies a spectral (e.g. Chebyshev) expansion
    per coordinate (KAN-style outer composition). Subsequent hidden layers
    form a standard MLP (or no hidden layers if a scalar/int is supplied).
    Supports a purely spectral 1D mode if hidden_size == 1 and dims == 1.

    Args:
        spectral_size: Number of spectral modes per input dimension.
        hidden_size: Hidden configuration (int or list). If 1 with dims==1,
            disables hidden/output activations (pure spectral).
        dims: Number of spatial dimensions.
        rank: Tensor rank of output.
        system: Optional coordinate system.
        name: Space name.
        transient: If True, include time as input.
        act_fun: Activation for hidden layers (ignored if purely spectral).
        basespace: Spectral base class (e.g. Chebyshev.Chebyshev).
        domains: List of (a, b) tuples for input domain mapping. If None,
            defaults to (-1, 1) per dimension.

    Raises:
        ValueError: If hidden_size == 1 while dims != 1.
    """

    def __init__(
        self,
        spectral_size: int,
        hidden_size: int | list[int],
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        name: str = "KANMLP",
        transient: bool = False,
        act_fun: Callable[[Array], Array] = nnx.tanh,
        basespace: BaseSpace = Chebyshev.Chebyshev,
        domains: list[tuple[float, float]] | None = None,
    ) -> None:
        """Initialize KANMLPSpace metadata."""
        NNSpace.__init__(self, dims, rank, transient, system, name)
        self.spectral_size = spectral_size
        self.hidden_size = hidden_size
        self.act_fun = act_fun
        self.basespace = basespace
        self.domains = domains
        if hidden_size == 1 and self.dims != 1:
            raise ValueError(
                "hidden_size=1 only allowed for dims=1. Consider using a "
                "TensorProductSpace instead."
            )
        if hidden_size == 1:
            self.act_fun = lambda x: x  # Pure spectral in 1D


class sPIKANSpace(NNSpace):
    """Fully spectral Kolmogorov-Arnold style function space.

    All layers (input + hidden + output) employ spectral bases (e.g.
    Chebyshev). Supports a pure spectral 1D case if hidden_size == 1.
    For dims > 1, at least one non-trivial hidden layer required.

    Args:
        spectral_size: Number of spectral modes per layer.
        hidden_size: Hidden configuration (int or list). If 1 with dims==1,
            network degenerates to a single spectral projection.
        dims: Number of spatial dimensions.
        rank: Tensor rank of output.
        system: Optional coordinate system.
        name: Space name.
        transient: If True, include time as input.
        act_fun: Activation for hidden layers (skipped if purely spectral).
        basespace: Spectral base class (e.g. Chebyshev.Chebyshev).
        domains: List of Domain tuples mapping each coordinate. If None,
            defaults to (-1, 1) per dimension.

    Raises:
        ValueError: If hidden_size == 1 while dims != 1.
    """

    def __init__(
        self,
        spectral_size: list[int] | int,
        hidden_size: list[int] | int,
        dims: int = 1,
        rank: int = 0,
        system: CoordSys = None,
        name: str = "sPIKAN",
        transient: bool = False,
        act_fun: Callable[[Array], Array] = nnx.tanh,
        basespace: BaseSpace = Chebyshev.Chebyshev,
        domains: list[Domain] | None = None,
    ) -> None:
        """Initialize sPIKANSpace metadata."""
        NNSpace.__init__(self, dims, rank, transient, system, name)
        self.spectral_size = spectral_size
        self.hidden_size = hidden_size
        self.act_fun = act_fun
        self.basespace = basespace
        self.domains = domains
        if hidden_size == 1 and self.dims != 1:
            raise ValueError(
                "hidden_size=1 only allowed for dims=1. Consider using a "
                "TensorProductSpace instead."
            )
        if hidden_size == 1:
            self.act_fun = lambda x: x  # Pure spectral in 1D


class UnionSpace(NNSpace):
    """Union of multiple neural network function spaces.

    Combines several NNSpace instances into a single space, e.g. for
    domain decomposition or multi-fidelity modeling.

    Args:
        spaces: Iterable of NNSpace instances to combine.
        name: Name of the union space.

    Attributes:
        spaces: Tuple of constituent NNSpace instances.
    """

    def __init__(self, *spaces: NNSpace, name: str = "UnionNN") -> None:
        """Initialize UnionSpace metadata."""
        self.spaces = tuple(*spaces)
        NNSpace.__init__(
            self,
            dims=self.spaces[0].dims,
            rank=self.spaces[0].rank,
            transient=self.spaces[0].is_transient,
            system=self.spaces[0].system,
            name=name,
        )

    def __getitem__(self, i: int) -> NNSpace:
        return self.spaces[i]
