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
from jaxfun.typing import Array


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
        """Return the base variables, including time if transient."""
        if self.is_transient:
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
            act_fun:
                Activation function for all except the output layer
            name: Name of MLPSpace

        """
        NNSpace.__init__(self, dims, rank, transient, system, name)
        self.hidden_size = hidden_size
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
        act_fun: Callable[[Array], Array] = nnx.tanh,
        act_fun_hidden: Callable[[Array], Array] = nnx.tanh,
        # PirateNet specific parameters
        nonlinearity: float = 0.0,
        periodicity: dict | None = None,
        fourier_emb: dict | None = None,
        pi_init: jnp.ndarray | None = None,
    ) -> None:
        NNSpace.__init__(self, dims, rank, transient, system, name)

        # PirateSpace requires at least one hidden layer, so change integer hidden_size
        # to [hidden_size]
        self.hidden_size = (
            hidden_size if isinstance(hidden_size, list | tuple) else [hidden_size]
        )
        self.act_fun = act_fun
        self.act_fun_hidden = act_fun_hidden

        self.nonlinearity = nonlinearity
        self.periodicity = periodicity
        self.fourier_emb = fourier_emb
        self.pi_init = pi_init
