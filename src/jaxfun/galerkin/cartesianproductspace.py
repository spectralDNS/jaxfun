from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self, cast, overload

if TYPE_CHECKING:
    from jaxfun.pinns.nnspaces import CartesianNNSpace, NNSpace

import jax
import jax.numpy as jnp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.galerkin.tensorproductspace import TensorProductSpace, multiplication_sign
from jaxfun.sharding import physical_sharding, spectral_sharding
from jaxfun.typing import MeshKind, RankTag

type MultiDimensionalSpace = TensorProductSpace | CartesianTensorProductSpace
type OneDimensionalSpace = OrthogonalSpace | DirectSum | CartesianProductSpace


@overload
def CartesianProduct(
    *basespaces: NNSpace,
    name: str = "CP",
    rank: int | RankTag = RankTag.NONE,
) -> CartesianNNSpace: ...
@overload
def CartesianProduct(
    *basespaces: MultiDimensionalSpace,
    name: str = "CP",
    rank: int | RankTag = RankTag.VECTOR,
) -> VectorTensorProductSpace: ...
@overload
def CartesianProduct(
    *basespaces: MultiDimensionalSpace,
    name: str = "CP",
    rank: int | RankTag = RankTag.NONE,
) -> CartesianTensorProductSpace: ...
@overload
def CartesianProduct(
    *basespaces: OneDimensionalSpace,
    name: str = "CP",
    rank: int | RankTag = RankTag.NONE,
) -> CartesianProductSpace: ...
def CartesianProduct(
    *basespaces: Any,
    name: str = "CP",
    rank: int | RankTag = RankTag.NONE,
) -> CartesianTensorProductSpace | CartesianProductSpace | CartesianNNSpace:
    """Factory returning the appropriate Cartesian product space.

    Returns CartesianNNSpace for NNSpace components, CartesianProductSpace for
    1D spectral components, VectorTensorProductSpace when rank==1, and
    CartesianTensorProductSpace otherwise.

    Args:
        *basespaces: Component spaces (all must have the same dims).
        name: Label for the Cartesian product space.
        rank: Rank of the Cartesian (tensor) product space.

    Returns:
        CartesianNNSpace, CartesianTensorProductSpace, CartesianProductSpace, or
        VectorTensorProductSpace instance.
    """
    from jaxfun.pinns.nnspaces import NNSpace as _NNSpace  # lazy — avoids circular

    rank_tag = RankTag(rank) if isinstance(rank, int) else rank
    if basespaces and all(isinstance(b, _NNSpace) for b in basespaces):
        from jaxfun.pinns.nnspaces import CartesianNNSpace as _CartesianNNSpace

        return _CartesianNNSpace(*basespaces, name=name, rank=rank_tag)

    basespaces_list = [copy.deepcopy(space) for space in basespaces]
    if all(basespace.dims == 1 for basespace in basespaces_list):
        return CartesianProductSpace(
            *cast(list[OneDimensionalSpace], basespaces_list), name=name, rank=rank_tag
        )
    assert all(basespace.dims > 1 for basespace in basespaces_list)
    assert len({b.dims for b in basespaces_list}) == 1

    if rank_tag == RankTag.VECTOR:
        return VectorTensorProductSpace(
            *cast(list[MultiDimensionalSpace], basespaces_list),
            name=name,
            rank=RankTag.VECTOR,
        )
    return CartesianTensorProductSpace(
        *cast(list[MultiDimensionalSpace], basespaces_list), name=name, rank=rank_tag
    )


class CartesianBaseSpace(ABC):
    """Abstract space for all Cartesian classes."""

    is_transient = False
    basespaces: Sequence[OneDimensionalSpace | MultiDimensionalSpace]
    _rank: RankTag

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """return iterator over basespaces"""

    @abstractmethod
    def __getitem__(self, i: int) -> Any:
        """Return component space i."""

    @abstractmethod
    def flatten(self) -> list[Any]:
        """Return flattened list of all component TensorProductSpace objects."""

    @abstractmethod
    def get_homogeneous(self) -> Self:
        """Return self as homogeneous space"""

    @abstractmethod
    def get_orthogonal(self) -> Self:
        """Return orthogonal version of self"""

    @abstractmethod
    def evaluate_mesh(
        self,
        u: tuple[Array, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: Any = None,
    ) -> Array: ...

    @abstractmethod
    def backward(
        self,
        u: tuple[Array, ...],
        N: Any = None,
    ) -> Array: ...

    @abstractmethod
    def backward_primitive(
        self,
        u: tuple[Array, ...],
        k: Any,
        N: Any = None,
    ) -> Array: ...

    def __len__(self) -> int:
        """Return number of subspaces."""
        return len(self.basespaces)

    @property
    def num_components(self) -> int:
        """Return total number of scalar components."""
        return len(self.flatten())

    @property
    def rank(self) -> RankTag:
        """Return tensor rank (1 for vector fields, -1 for composite)."""
        return self._rank

    @property
    def dims(self) -> int:
        """Return spatial dimension."""
        return self.basespaces[0].dims

    @property
    def dim(self) -> int:
        """Return total number of modes."""
        return sum(self.block_sizes)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Return tuple of component space dimensions."""
        return tuple(space.dim for space in self.flatten())

    @property
    def num_dofs(self) -> tuple[tuple[int, ...], ...]:
        """Return tuple of active degrees of freedom per axis."""
        result = []
        for space in self.flatten():
            nd = space.num_dofs
            result.append((nd,) if isinstance(nd, int) else nd)
        return tuple(result)

    @property
    def is_orthogonal(self) -> bool:
        """Return True if underlying bases are all orthogonal."""
        return all(space.is_orthogonal for space in self.basespaces)

    @property
    def shape(self) -> Any:
        """Return physical-space shape for each subspace."""
        return tuple(space.shape for space in self.basespaces)

    def evaluate(self, x: Array, c: tuple[Array, ...]) -> Array:
        """Evaluate each component at scattered points and stack into one Array."""
        return jnp.array(
            [space.evaluate(x, c[i]) for i, space in enumerate(self.flatten())]
        )

    def forward(self, u: Array) -> tuple[Array, ...]:
        """Forward transform each physical component into spectral space."""
        return tuple(space.forward(u[i]) for i, space in enumerate(self.flatten()))

    def scalar_product(self, u: Array) -> tuple[Array, ...]:
        """Return scalar products along each axis for all components."""
        return tuple(
            space.scalar_product(u[i]) for i, space in enumerate(self.flatten())
        )

    def to_orthogonal(self, c: tuple[Array, ...]) -> tuple[Array, ...]:
        """Convert coefficients to orthogonal basis."""
        return tuple(
            space.to_orthogonal(c[i]) for i, space in enumerate(self.flatten())
        )

    def from_orthogonal(self, c: tuple[Array, ...]) -> tuple[Array, ...]:
        """Convert coefficients from orthogonal basis."""
        return tuple(
            space.from_orthogonal(c[i]) for i, space in enumerate(self.flatten())
        )


class CartesianTensorProductSpace(CartesianBaseSpace):
    """Composite tensor product space for ND (dims > 1) problems.

    Holds a tuple of TensorProductSpace (or DirectSumTPS) components, such as
    used for vector-valued or mixed-variable formulations on multi-dimensional
    domains (e.g. Stokes equations in 2D/3D).

    Attributes:
        basespaces: List of component tensor product spaces.
        system: Shared coordinate system.
        name: Label.
        tensorname: Joined printable representation.
        leaf: CartesianTensorProductSpace this space is part of or self.
    """

    is_transient = False
    basespaces: Sequence[MultiDimensionalSpace]

    def __init__(
        self,
        *basespaces: MultiDimensionalSpace,
        name: str = "CTPS",
        rank: int | RankTag = RankTag.NONE,
    ) -> None:
        from jaxfun.galerkin import DirectSumTPS

        self.basespaces = list(basespaces)
        self.system: CoordSys = self.basespaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.basespaces])
        self._spectral_sharding = spectral_sharding if len(jax.devices()) > 1 else None
        self._physical_sharding = physical_sharding if len(jax.devices()) > 1 else None
        self._rank = RankTag(rank) if isinstance(rank, int) else rank
        self.leaf: CartesianTensorProductSpace = self

        for space in self.basespaces:
            space.leaf = self

        for i, space in enumerate(self.flatten()):
            space.global_index = i
            space.leaf = self
            if isinstance(space, DirectSumTPS):
                for tpspace in space.tpspaces.values():
                    tpspace.global_index = i
                    tpspace.leaf = self

        self.mesh = self.basespaces[0].mesh
        self.num_quad_points = self.basespaces[0].num_quad_points

    def __iter__(self) -> Iterator[MultiDimensionalSpace]:
        """Iterate over component spaces."""
        return iter(self.basespaces)

    def __getitem__(self, i: int) -> MultiDimensionalSpace:
        """Return component space i."""
        return self.basespaces[i]

    def flatten(self) -> list[TensorProductSpace]:
        """Return flattened list of all component TensorProductSpace objects."""
        spaces: list[TensorProductSpace] = []
        for space in self.basespaces:
            if isinstance(space, CartesianTensorProductSpace):
                spaces.extend(space.flatten())
            else:
                spaces.append(space)
        return spaces

    def get_homogeneous(self) -> CartesianTensorProductSpace:
        from .tensorproductspace import DirectSumTPS

        f = []
        for space in self.basespaces:
            f.append(
                space.get_homogeneous()
                if isinstance(space, DirectSumTPS | CartesianTensorProductSpace)
                else space
            )
        return CartesianProduct(*f, name=self.name + "H", rank=self.rank)

    def get_orthogonal(self) -> CartesianTensorProductSpace:
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return CartesianProduct(
            *orthogonal_spaces, name=self.name + "O", rank=self.rank
        )

    def evaluate_mesh(
        self,
        u: tuple[Array, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Evaluate each component on a mesh and stack into one Array.

        Args:
            u: input coefficients.
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
                MeshKind.UNIFORM).
            N: Number of physical points along each axis.

        Note:
            All basespaces must use the same number of points in physical space.

        """
        results = []
        for i, space in enumerate(self.flatten()):
            results.append(space.evaluate_mesh(u[i], kind=kind, N=N))
        return jnp.stack(results)

    def backward(
        self,
        u: tuple[Array, ...],
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Backward transform per-component spectral coefficients to physical
        space.

        Args:
            u: input coefficients
            N: Number of physical points along each axis.

        Note:
            All basespaces must use the same number of points in physical space.

        """
        coeffs = [space.backward(u[i], N=N) for i, space in enumerate(self.flatten())]
        return jnp.stack(coeffs)

    def backward_primitive(
        self,
        u: tuple[Array, ...],
        k: tuple[tuple[int, ...], ...],
        N: tuple[int | None, ...] | None = None,
    ) -> Array:
        """Backward primitive transform for all components.

        Args:
            u: input coefficients
            k: tuple of the number of derivatives for each direction. One
                tuple item for each basespace.
            N: Number of physical points along each axis.

        Note:
            All basespaces must use the same number of points in physical space.
        """
        coeffs = [
            space.backward_primitive(u[i], k=k[i], N=N)
            for i, space in enumerate(self.flatten())
        ]
        return jnp.stack(coeffs)


class CartesianProductSpace(CartesianBaseSpace):
    """Composite 1D function space (Cartesian product of OrthogonalSpace objects).

    Represents a tuple of 1D spaces for mixed-variable formulations on 1D
    domains (e.g. a 1D Stokes-type problem).

    Attributes:
        basespaces: List of component 1D spaces.
        system: Shared coordinate system.
        name: Label.
        tensorname: Joined printable representation.
        leaf: CartesianProductSpace this space is part of or self.
    """

    is_transient = False
    basespaces: Sequence[OneDimensionalSpace]
    _rank: RankTag

    def __init__(
        self,
        *basespaces: OneDimensionalSpace,
        name: str = "CPS",
        rank: int | RankTag = RankTag.NONE,
    ) -> None:
        self.basespaces = list(basespaces)
        self.system: CoordSys = self.basespaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.basespaces])
        self._rank = RankTag(rank) if isinstance(rank, int) else rank
        self.leaf: CartesianProductSpace = self

        for space in self.basespaces:
            space.leaf = self

        for i, space in enumerate(self.flatten()):
            space.global_index = i
            space.leaf = self
            if isinstance(space, DirectSum):
                for subspace in space.basespaces:
                    subspace.global_index = i
                    subspace.leaf = self

        self.mesh = self.basespaces[0].mesh
        self.num_quad_points = self.basespaces[0].num_quad_points

    def __iter__(self) -> Iterator[OneDimensionalSpace]:
        """Iterate over component spaces."""
        return iter(self.basespaces)

    def __getitem__(self, i: int) -> OneDimensionalSpace:
        """Return component space i."""
        return self.basespaces[i]

    def flatten(self) -> list[OrthogonalSpace | DirectSum]:
        """Return flattened list of all component 1D spaces."""
        spaces: list[OrthogonalSpace | DirectSum] = []
        for space in self.basespaces:
            if isinstance(space, CartesianProductSpace):
                spaces.extend(space.flatten())
            else:
                spaces.append(space)
        return spaces

    @property
    def dims(self) -> int:
        """Return spatial dimension (always 1)."""
        return 1

    def get_homogeneous(self) -> CartesianProductSpace:
        f = []
        for space in self.basespaces:
            f.append(
                space.get_homogeneous()
                if isinstance(space, DirectSum | CartesianProductSpace)
                else space
            )
        return CartesianProduct(*f, name=self.name + "H", rank=self.rank)

    def get_orthogonal(self) -> CartesianProductSpace:
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return CartesianProduct(
            *orthogonal_spaces, name=self.name + "O", rank=self.rank
        )

    def evaluate_mesh(
        self,
        u: tuple[Array, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: int | None = None,
    ) -> Array:
        """Evaluate each component on a mesh and stack into one Array.

        Args:
            u: input coefficients.
            kind: Mesh type for backward evaluation (MeshKind.QUADRATURE or
                MeshKind.UNIFORM).
            N: Number of physical points along each axis.

        Note:
            All basespaces must use the same number of points in physical space.

        """
        results = []
        for i, space in enumerate(self.flatten()):
            results.append(space.evaluate_mesh(u[i], kind=kind, N=N))
        return jnp.stack(results)

    def backward(
        self,
        u: tuple[Array, ...],
        N: int | None = None,
    ) -> Array:
        """Backward transform per-component spectral coefficients to physical
        space.

        Args:
            u: input coefficients
            N: Number of physical points.

        Note:
            All basespaces must use the same number of points in physical space.

        """
        coeffs = [space.backward(u[i], N=N) for i, space in enumerate(self.flatten())]
        return jnp.stack(coeffs)

    def backward_primitive(
        self,
        u: tuple[Array, ...],
        k: tuple[int, ...],
        N: int | None = None,
    ) -> Array:
        """Backward primitive transform for all components.

        Args:
            u: input coefficients
            k: tuple of the number of derivatives for each space.
            N: Number of physical points.

        Note:
            All basespaces must use the same number of points in physical space.
        """
        coeffs = [
            space.backward_primitive(u[i], k=k[i], N=N)
            for i, space in enumerate(self.flatten())
        ]
        return jnp.stack(coeffs)


class VectorTensorProductSpace(CartesianTensorProductSpace):
    """Vector-valued tensor product space (rank-1 CartesianTensorProductSpace)."""

    def __iter__(self) -> Iterator[TensorProductSpace]:
        """Iterate over component spaces."""
        return iter(cast(list[TensorProductSpace], self.basespaces))

    def __getitem__(self, i: int) -> TensorProductSpace:
        """Return component space i."""
        return cast(list[TensorProductSpace], self.basespaces)[i]
