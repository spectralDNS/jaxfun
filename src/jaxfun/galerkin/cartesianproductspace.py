from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from jaxfun.coordinates import CoordSys
from jaxfun.galerkin.tensorproductspace import TensorProductSpace, multiplication_sign
from jaxfun.sharding import physical_sharding, spectral_sharding
from jaxfun.typing import MeshKind


def CartesianProduct(
    *basespaces: TensorProductSpace | CartesianProductSpace,
    name: str = "CP",
    rank: int = -1,
) -> CartesianProductSpace:
    """Factory returning CartesianProductSpace.

    Args:
        *basespaces: TensorProductSpace or CartesianProductSpace objects.
        name: Label for the Cartesian product space.
        rank: Rank of the Cartesian product space.

    Returns:
        Instance of CartesianProductSpace
    """
    basespaces_list: list[TensorProductSpace | CartesianProductSpace] = [
        copy.deepcopy(space) for space in basespaces
    ]
    if rank == 1:
        return VectorTensorProductSpace(*basespaces_list, name=name, rank=1)
    return CartesianProductSpace(*basespaces_list, name=name, rank=rank)


class CartesianProductSpace:
    """Composite tensor product space.

    Represents a tuple of identical (or differing in boundary conditions)
    TensorProductSpace objects.

    Attributes:
        tensorspaces: Tuple of component tensor spaces.
        system: Shared coordinate system.
        name: Label.
        tensorname: Joined printable representation.
    """

    is_transient = False

    def __init__(
        self,
        *basespaces: TensorProductSpace | CartesianProductSpace,
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
        self.leaf: CartesianProductSpace = self

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

    def __iter__(self) -> Iterator[CartesianProductSpace | TensorProductSpace]:
        """Iterate over component spaces."""
        return iter(self.basespaces)

    def __getitem__(self, i: int) -> CartesianProductSpace | TensorProductSpace:
        """Return component space i."""
        return self.basespaces[i]

    def flatten(self) -> list[TensorProductSpace]:
        """Return flattened list of all component tensor spaces."""
        spaces: list[TensorProductSpace] = []
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
        """Return tensor rank (1 for vector fields, 0 for scalars and -1
        for composite)."""
        return self._rank

    @property
    def dims(self) -> int:
        """Return spatial dimension."""
        return self.basespaces[0].dims

    @property
    def dim(self) -> int:
        """Return total number of modes."""
        return sum([space.dim for space in self.basespaces])

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
        """Evaluate each component at scattered points and stack into one Array.

        All components are evaluated at the same points so outputs have equal
        leading shape and can be stacked.

        Args:
            x: Array of per-axis coordinates stacked (N, d).
            c: Sequence of Coefficient arrays.

        Returns:
            Stacked array of shape (n_components, N).
        """
        return jnp.array(
            [space.evaluate(x, c[i]) for i, space in enumerate(self.flatten())]
        )

    def evaluate_mesh(
        self,
        u: tuple[Array, ...],
        kind: MeshKind | str = MeshKind.QUADRATURE,
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> Array:
        """Evaluate each component on a mesh and stack into one Array.

        Args:
            u: Sequence of input arrays.
            kind: Type of mesh to evaluate on.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Stacked array of shape (n_components, ...).
        """
        return jnp.stack(
            [
                space.evaluate_mesh(u[i], kind=kind, N=N[i] if N is not None else None)
                for i, space in enumerate(self.flatten())
            ]
        )

    def forward(self, u: tuple[Array, ...]) -> tuple[Array, ...]:
        """Forward transform with optional truncation.

        Args:
            u: Input array.

        Returns:
            Array of forward transform values.
        """
        coeffs = []
        for i, space in enumerate(self.flatten()):
            ci = space.forward(u[i])
            coeffs.append(ci)
        return tuple(coeffs)

    def scalar_product(self, u: tuple[Array, ...]) -> tuple[Array, ...]:
        """Return tensor of inner products along each axis (separable).
        Args:
            u: Sequence of input arrays.

        Returns:
            Sequence of array of inner products for all components.
        """
        coeffs = []
        for i, space in enumerate(self.flatten()):
            ci = space.scalar_product(u[i])
            coeffs.append(ci)
        return tuple(coeffs)

    def backward(
        self,
        u: tuple[Array, ...],
        N: tuple[tuple[int | None, ...] | None, ...] | None = None,
    ) -> tuple[Array, ...]:
        """Backward transform with optional padding.

        Args:
            u: Input array.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of backward transform values.
        """
        coeffs = []
        for i, space in enumerate(self.flatten()):
            ci = space.backward(u[i], N=N[i] if N is not None else None)
            coeffs.append(ci)
        return tuple(coeffs)

    def backward_primitive(
        self,
        u: tuple[Array, ...],
        k: tuple[int, ...],
        N: tuple[tuple[int | None, ...], ...] | None = None,
    ) -> tuple[Array, ...]:
        """Backward primitive transform with optional padding.

        Args:
            u: Sequence of input arrays.
            k: Tuple of derivative orders.
            N: Optional per-axis counts (defaults each to space.num_quad_points).

        Returns:
            Array of backward primitive transform values.
        """
        coeffs = []
        for i, space in enumerate(self.flatten()):
            ci = space.backward_primitive(u[i], k=k, N=N[i] if N is not None else None)
            coeffs.append(ci)
        return tuple(coeffs)

    def to_orthogonal(self, c: tuple[Array, ...]) -> tuple[Array, ...]:
        """Convert coefficients to orthogonal basis.

        Args:
            c: Input array of coefficients.

        Returns:
            Array of coefficients in orthogonal basis.
        """
        coeffs = []
        for i, space in enumerate(self.flatten()):
            ci = space.to_orthogonal(c[i])
            coeffs.append(ci)
        return tuple(coeffs)

    def from_orthogonal(self, c: tuple[Array, ...]) -> tuple[Array, ...]:
        """Convert coefficients from orthogonal basis.

        Args:
            c: Input array of coefficients.

        Returns:
            Array of coefficients in the original basis.
        """
        coeffs = []
        for i, space in enumerate(self.flatten()):
            ci = space.from_orthogonal(c[i])
            coeffs.append(ci)
        return tuple(coeffs)

    def get_homogeneous(self) -> CartesianProductSpace:
        from .tensorproductspace import DirectSumTPS

        f = []
        for space in self.basespaces:
            f.append(
                space.get_homogeneous()
                if isinstance(space, DirectSumTPS | CartesianProductSpace)
                else space
            )
        return CartesianProduct(*f, name=self.name + "o", rank=self.rank)

    def get_orthogonal(self) -> CartesianProductSpace:
        orthogonal_spaces = [space.get_orthogonal() for space in self.basespaces]
        return CartesianProduct(
            *orthogonal_spaces, name=self.name + "o", rank=self.rank
        )


class VectorTensorProductSpace(CartesianProductSpace):
    def __iter__(self) -> Iterator[TensorProductSpace]:
        """Iterate over component spaces."""
        return iter(cast(list[TensorProductSpace], self.basespaces))

    def __getitem__(self, i: int) -> TensorProductSpace:
        """Return component space i."""
        return cast(list[TensorProductSpace], self.basespaces)[i]
