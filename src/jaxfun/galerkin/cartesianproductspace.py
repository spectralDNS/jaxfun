from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any, cast, overload

import jax
import jax.numpy as jnp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.composite import DirectSum
from jaxfun.galerkin.orthogonal import OrthogonalSpace
from jaxfun.galerkin.tensorproductspace import TensorProductSpace, multiplication_sign
from jaxfun.sharding import physical_sharding, spectral_sharding
from jaxfun.typing import MeshKind

type MultiDimensionalSpace = TensorProductSpace | CartesianTensorProductSpace
type OneDimensionalSpace = OrthogonalSpace | DirectSum | CartesianProductSpace


@overload
def CartesianProduct(
    *basespaces: MultiDimensionalSpace, name: str = "CP", rank: int = -1
) -> CartesianTensorProductSpace: ...
@overload
def CartesianProduct(
    *basespaces: OneDimensionalSpace, name: str = "CP", rank: int = -1
) -> CartesianProductSpace: ...
@overload
def CartesianProduct(
    *basespaces: MultiDimensionalSpace, name: str = "CP", rank: int = 1
) -> VectorTensorProductSpace: ...
def CartesianProduct(
    *basespaces: MultiDimensionalSpace | OneDimensionalSpace,
    name: str = "CP",
    rank: int = -1,
) -> CartesianTensorProductSpace | CartesianProductSpace:
    """Factory returning the appropriate Cartesian product space.

    Returns CartesianProductSpace for 1D components, VectorTensorProductSpace
    when rank==1, and CartesianTensorProductSpace otherwise.

    Args:
        *basespaces: Component spaces (all must have the same dims).
        name: Label for the Cartesian product space.
        rank: Rank of the Cartesian (tensor) product space.

    Returns:
        CartesianTensorProductSpace, CartesianProductSpace, or
        VectorTensorProductSpace instance.
    """
    basespaces_list = [copy.deepcopy(space) for space in basespaces]
    if basespaces_list[0].dims == 1:
        return CartesianProductSpace(
            *cast(list[OneDimensionalSpace], basespaces_list), name=name, rank=rank
        )
    if rank == 1:
        return VectorTensorProductSpace(
            *cast(list[MultiDimensionalSpace], basespaces_list), name=name, rank=1
        )
    return CartesianTensorProductSpace(
        *cast(list[MultiDimensionalSpace], basespaces_list), name=name, rank=rank
    )


class CartesianTensorProductSpace:
    """Composite tensor product space for ND (dims > 1) problems.

    Holds a tuple of TensorProductSpace (or DirectSumTPS) components, such as
    used for vector-valued or mixed-variable formulations on multi-dimensional
    domains (e.g. Stokes equations in 2D/3D).

    Attributes:
        basespaces: List of component tensor product spaces.
        system: Shared coordinate system.
        name: Label.
        tensorname: Joined printable representation.
    """

    is_transient = False

    def __init__(
        self,
        *basespaces: MultiDimensionalSpace,
        name: str = "CTPS",
        rank: int = -1,
    ) -> None:
        from jaxfun.galerkin import DirectSumTPS

        self.basespaces = list(basespaces)
        self.system: CoordSys = self.basespaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.basespaces])
        self._spectral_sharding = spectral_sharding if len(jax.devices()) > 1 else None
        self._physical_sharding = physical_sharding if len(jax.devices()) > 1 else None
        self._rank = rank
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

    def __len__(self) -> int:
        """Return number of subspaces."""
        return len(self.basespaces)

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
                spaces.append(cast(TensorProductSpace, space))
        return spaces

    @property
    def num_components(self) -> int:
        """Return total number of scalar components."""
        return len(self.flatten())

    @property
    def rank(self) -> int:
        """Return tensor rank (1 for vector fields, -1 for composite)."""
        return self._rank

    @property
    def dims(self) -> int:
        """Return spatial dimension."""
        return self.basespaces[0].dims

    @property
    def dim(self) -> int:
        """Return total number of modes."""
        return sum(space.dim for space in self.basespaces)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Return tuple of component space dimensions."""
        return tuple(space.dim for space in self.flatten())

    @property
    def num_dofs(self) -> tuple[tuple[int, ...], ...]:
        """Return tuple of active degrees of freedom per axis."""
        return tuple(space.num_dofs for space in self.flatten())

    @property
    def is_orthogonal(self) -> bool:
        """Return True if underlying bases are all orthogonal."""
        return all(space.is_orthogonal for space in self.basespaces)

    def shape(self) -> Any:
        """Return modal shape for each subspace."""
        return tuple(space.shape() for space in self.basespaces)

    def evaluate(self, x: Array, c: tuple[Array, ...]) -> Array:
        """Evaluate each component at scattered points and stack into one Array."""
        return jnp.array(
            [space.evaluate(x, c[i]) for i, space in enumerate(self.flatten())]
        )

    def evaluate_mesh(
        self,
        u: tuple[Array, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Evaluate each component on a mesh and stack into one Array."""
        results = []
        for i, space in enumerate(self.flatten()):
            ni = N[i] if N is not None else None
            results.append(space.evaluate_mesh(u[i], kind=kind, N=ni))
        return jnp.stack(results)

    def forward(self, u: Array) -> tuple[Array, ...]:
        """Forward transform each physical component into spectral space."""
        return tuple(space.forward(u[i]) for i, space in enumerate(self.flatten()))

    def scalar_product(self, u: Array) -> tuple[Array, ...]:
        """Return scalar products along each axis for all components."""
        return tuple(
            space.scalar_product(u[i]) for i, space in enumerate(self.flatten())
        )

    def backward(
        self,
        u: tuple[Array, ...],
        N: tuple[tuple[int | None, ...] | None, ...] | None = None,
    ) -> Array:
        """Backward transform per-component spectral coefficients to physical space."""
        coeffs = [
            space.backward(u[i], N=N[i] if N is not None else None)
            for i, space in enumerate(self.flatten())
        ]
        return jnp.stack(coeffs)

    def backward_primitive(
        self,
        u: tuple[Array, ...],
        k: tuple[int, ...],
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Backward primitive transform for all components."""
        coeffs = [
            space.backward_primitive(u[i], k=k, N=N[i] if N is not None else None)
            for i, space in enumerate(self.flatten())
        ]
        return jnp.stack(coeffs)

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


class CartesianProductSpace:
    """Composite 1D function space (Cartesian product of OrthogonalSpace objects).

    Represents a tuple of 1D spaces for mixed-variable formulations on 1D
    domains (e.g. a 1D Stokes-type problem).

    Attributes:
        basespaces: List of component 1D spaces.
        system: Shared coordinate system.
        name: Label.
        tensorname: Joined printable representation.
    """

    is_transient = False

    def __init__(
        self,
        *basespaces: OneDimensionalSpace,
        name: str = "CPS",
        rank: int = -1,
    ) -> None:
        self.basespaces = list(basespaces)
        self.system: CoordSys = self.basespaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.basespaces])
        self._rank = rank
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

    def __len__(self) -> int:
        """Return number of subspaces."""
        return len(self.basespaces)

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
    def num_components(self) -> int:
        """Return total number of scalar components."""
        return len(self.flatten())

    @property
    def rank(self) -> int:
        """Return tensor rank (-1 for composite mixed spaces)."""
        return self._rank

    @property
    def dims(self) -> int:
        """Return spatial dimension (always 1)."""
        return 1

    @property
    def dim(self) -> int:
        """Return total number of modes."""
        return sum(space.dim for space in self.basespaces)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Return tuple of component space dimensions."""
        return tuple(space.dim for space in self.flatten())

    @property
    def num_dofs(self) -> tuple[tuple[int, ...], ...]:
        """Return tuple of active degrees of freedom per axis.

        OrthogonalSpace.num_dofs returns an int; wrap in a 1-tuple so
        BlockArray always receives a uniform tuple[tuple[int, ...], ...].
        """
        result = []
        for space in self.flatten():
            nd = space.num_dofs
            result.append((nd,) if isinstance(nd, int) else nd)
        return tuple(result)

    @property
    def is_orthogonal(self) -> bool:
        """Return True if underlying bases are all orthogonal."""
        return all(space.is_orthogonal for space in self.basespaces)

    def shape(self) -> Any:
        """Return modal shape for each subspace."""
        result = []
        for space in self.basespaces:
            if isinstance(space, DirectSum):
                result.append((space.dim,))
            else:
                result.append(space.shape())
        return tuple(result)

    def evaluate(self, x: Array, c: tuple[Array, ...]) -> Array:
        """Evaluate each component at scattered points and stack into one Array."""
        return jnp.array(
            [space.evaluate(x, c[i]) for i, space in enumerate(self.flatten())]
        )

    def evaluate_mesh(
        self,
        u: tuple[Array, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Evaluate each component on a mesh and stack into one Array."""
        results = []
        for i, space in enumerate(self.flatten()):
            ni = N[i] if N is not None else None
            if isinstance(space, DirectSum):
                results.append(
                    space[0].evaluate_mesh(space.to_orthogonal(u[i]), kind=kind, N=ni)
                )
            else:
                results.append(space.evaluate_mesh(u[i], kind=kind, N=ni))
        return jnp.stack(results)

    def forward(self, u: Array) -> tuple[Array, ...]:
        """Forward transform each physical component into spectral space."""
        return tuple(space.forward(u[i]) for i, space in enumerate(self.flatten()))

    def scalar_product(self, u: Array) -> tuple[Array, ...]:
        """Return scalar products for all components."""
        return tuple(
            space.scalar_product(u[i]) for i, space in enumerate(self.flatten())
        )

    def backward(
        self,
        u: tuple[Array, ...],
        N: tuple[tuple[int | None, ...] | None, ...] | None = None,
    ) -> Array:
        """Backward transform per-component spectral coefficients to physical space."""
        coeffs = [
            space.backward(u[i], N=N[i] if N is not None else None)
            for i, space in enumerate(self.flatten())
        ]
        return jnp.stack(coeffs)

    def backward_primitive(
        self,
        u: tuple[Array, ...],
        k: tuple[int, ...],
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Backward primitive transform for all components."""
        coeffs = [
            space.backward_primitive(u[i], k=k, N=N[i] if N is not None else None)
            for i, space in enumerate(self.flatten())
        ]
        return jnp.stack(coeffs)

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


class VectorTensorProductSpace(CartesianTensorProductSpace):
    """Vector-valued tensor product space (rank-1 CartesianTensorProductSpace)."""

    def __iter__(self) -> Iterator[TensorProductSpace]:
        """Iterate over component spaces."""
        return iter(cast(list[TensorProductSpace], self.basespaces))

    def __getitem__(self, i: int) -> TensorProductSpace:
        """Return component space i."""
        return cast(list[TensorProductSpace], self.basespaces)[i]

    def mesh(
        self,
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[int | None, ...] | None = None,
        broadcast: bool = True,
    ) -> tuple[Array, ...]:
        """Delegate to first component TensorProductSpace mesh."""
        return self[0].mesh(kind=kind, N=N, broadcast=broadcast)
