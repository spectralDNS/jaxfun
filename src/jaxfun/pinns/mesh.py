"""Sampling meshes for 1D/2D reference domains and mapped geometric regions.

Supported domains:
- CartesianProductMesh (abstract class for product meshes)
- UnitLine / Line
- UnitSquare / Rectangle
- Annulus (polar -> Cartesian conversion)
- ShapelyMesh (arbitrary polygonal domains)

The Cartesian product mesh can be created from any combination of
meshes derived from the BaseMesh class. For example, the UnitSqure/Rectangle
are implemented as the Cartesian product of two Line meshes.

Sampling kinds for lines:
- 'uniform'   : Equidistant points
- 'legendre'  : Gauss-Legendre nodes
- 'chebyshev' : Chebyshev nodes of the first kind
- 'random'    : Pseudorandom uniform samples

Weights:
Return 1 when uniform/random (each point equal) or arrays for quadrature-based kinds.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.typing import ArrayLike
from shapely import LineString, Polygon

from jaxfun.typing import Array, DomainType, SampleMethod, cast
from jaxfun.utils import leggauss

type SampleMethodLike = SampleMethod | str
type KindType = SampleMethodLike | Sequence[SampleMethodLike] | None
type NPointsType = int | tuple[int, ...]
type ParameterMesh = MultiParameterMesh | SingleParameterMesh


def _coerce_sample_method(kind: SampleMethodLike) -> SampleMethod:
    return kind if isinstance(kind, SampleMethod) else SampleMethod(kind)


def _normalize_kind(kind: KindType, n: int) -> list[SampleMethod]:
    if kind is None:
        return [SampleMethod.UNIFORM] * n
    if isinstance(kind, str | SampleMethod):
        return [_coerce_sample_method(kind)] * n
    return [_coerce_sample_method(k) for k in kind]


class BaseMesh:
    """Marker base class for all meshes.

    All concrete meshes should derive from this class so runtime
    `isinstance(x, BaseMesh)` checks remain valid.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return spatial dimension of the mesh."""
        ...


class SingleParameterMesh(BaseMesh, ABC):
    """Base class for meshes with total number of points N and scalar sampling kind."""

    def get_points(
        self,
        N: int,
        domain: DomainType = "all",
        kind: SampleMethodLike = SampleMethod.UNIFORM,
    ) -> Array:
        """Return sampled points.

        Args:
            N: Number of total points in mesh. Identical to length of returned
                array in case domain is 'all'.
            domain: 'inside' | 'boundary' | 'all'
            kind: Sampling kind.
        """
        if domain == "inside":
            return self.get_points_inside_domain(N, kind=kind)
        if domain == "boundary":
            return self.get_points_on_domain(N, kind=kind)
        if domain == "all":
            return self.get_all_points(N, kind=kind)
        raise ValueError("domain must be 'inside', 'boundary' or 'all'")

    def get_weights(
        self,
        N: int,
        domain: DomainType = "inside",
        kind: SampleMethodLike | None = None,
    ) -> Array | Literal[1]:
        """Return sampled weights.

        Args:
            N: Total number of weights in mesh. Identical to length of returned
                array in case domain is 'all'.
            domain: 'inside' | 'boundary' | 'all'
            kind: Sampling kind.
        """
        if domain == "inside":
            return self.get_weights_inside_domain(N, kind=kind)
        if domain == "boundary":
            return self.get_weights_on_domain(N, kind=kind)
        if domain == "all":
            return self.get_all_weights(N, kind=kind)
        raise ValueError("domain must be 'inside', 'boundary' or 'all'")

    @abstractmethod
    def get_points_inside_domain(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array: ...

    @abstractmethod
    def get_points_on_domain(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array: ...

    @abstractmethod
    def get_all_points(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array: ...

    @abstractmethod
    def get_weights_inside_domain(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Array | Literal[1]: ...

    @abstractmethod
    def get_weights_on_domain(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Array | Literal[1]: ...

    @abstractmethod
    def get_all_weights(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Array | Literal[1]: ...

    @abstractmethod
    def boundary_mask(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array: ...


class MultiParameterMesh(BaseMesh, ABC):
    """Base class for meshes with cartesian-product style signatures."""

    def get_points(
        self,
        *N: NPointsType,
        domain: DomainType = "all",
        kind: KindType = SampleMethod.UNIFORM,
    ) -> Array:
        """Return sampled points.

        Args:
            N: The tuple N will represent all arguments needed for all submeshes.
                For example, a 2D Rectangle mesh, which is the Cartesian product
                of two lines, will expect two integer arguments.

                For a Rectangle/UnitSquare mesh, it is also possible to draw
                random points within the domain, and not use a Cartesian product
                of line meshes. This case is enabled when kind='random', and
                the 2-tuple N then represents the total number of points and
                the number of boundary points to sample.
            domain: 'inside' | 'boundary' | 'all'
            kind: Sampling kind(s). For CartesianProductMesh, this should be a
                list of kinds for each submesh. A single kind string can also
                be provided, which is applied to all submeshes.
        """
        if domain == "inside":
            return self.get_points_inside_domain(*N, kind=kind)
        elif domain == "boundary":
            return self.get_points_on_domain(*N, kind=kind)
        elif domain == "all":
            return self.get_all_points(*N, kind=kind)
        raise ValueError("domain must be 'inside', 'boundary' or 'all'")

    def get_weights(
        self,
        *N: NPointsType,
        domain: DomainType = "inside",
        kind: KindType = None,
    ) -> Array | Literal[1]:
        """Return sampled weights.

        Args:
            N: The tuple N represent all arguments needed for all submeshes.
                For example, a 2D Rectangle mesh, which is the Cartesian product
                of two lines, will expect two integer arguments.
            domain: 'inside' | 'boundary' | 'all'
            kind: Sampling kind(s). This should be a list of kinds for each submesh.
                If a single kind string is provided, it is applied to all submeshes.

        Returns:
            Array of weights.
        """
        if domain == "inside":
            return self.get_weights_inside_domain(*N, kind=kind)
        elif domain == "boundary":
            return self.get_weights_on_domain(*N, kind=kind)
        elif domain == "all":
            return self.get_all_weights(*N, kind=kind)
        raise ValueError("domain must be 'inside', 'boundary' or 'all'")

    @abstractmethod
    def get_points_inside_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array: ...

    @abstractmethod
    def get_points_on_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array: ...

    @abstractmethod
    def get_all_points(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array: ...

    @abstractmethod
    def get_weights_inside_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]: ...

    @abstractmethod
    def get_weights_on_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]: ...

    @abstractmethod
    def get_all_weights(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]: ...

    @abstractmethod
    def boundary_mask(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array: ...


class CartesianProductMesh(MultiParameterMesh):
    """Cartesian product mesh."""

    def __init__(self, *m0: BaseMesh) -> None:
        """Cartesian product mesh.

        Args:
            *m0: Submeshes to form the Cartesian product.

        """
        self.submeshes = list(m0)

    @property
    def dim(self) -> int:
        """Return spatial dimension of the mesh."""
        return sum(m.dim for m in self.submeshes)

    def boundary_mask(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        """Return boolean mask for boundary points.

        Args:
            N: Number of points for each submesh.
            kind: List of the kind of sampling used for submeshes. If a single
                string is provided, it is applied to all submeshes.

        Returns:
            Boolean array of shape (N[0]*N[1]*...,) with True for boundary points.
        """
        if (
            kind == SampleMethod.RANDOM
            and len(self.submeshes) == 2
            and isinstance(self.submeshes[0], Line)
        ):
            assert len(N) == 2
            s0, s1 = cast(tuple[Line, Line], self.submeshes)
            N0, N1 = cast(tuple[int, int], N)
            smesh = RectangleShapely(
                left=s0.left, right=s0.right, bottom=s1.left, top=s1.right
            )
            smesh.boundary_factor = N1 / N0
            return smesh.boundary_mask(N0, kind=SampleMethod.RANDOM)

        kind_list = _normalize_kind(kind, len(self.submeshes))
        bnd_marks = []
        for m, Ni, knd in zip(self.submeshes, N, kind_list, strict=True):
            args = (Ni,) if np.isscalar(Ni) else tuple(Ni)
            bnd_marks.append(m.boundary_mask(*args, kind=knd))
        mask = jnp.array(list(itertools.product(*bnd_marks)))

        return jnp.any(mask, axis=1)

    def get_all_points(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        """Return all points including boundaries.

        Args:
            N: Number of points for each submesh.
            kind: List of the kind of sampling used for submeshes. If a single
                string is provided, it is applied to all submeshes.

                If kind='random' and the mesh is 2D with first submesh a Line,
                random points are sampled within the rectangular domain. In this
                case, N should be a 2-tuple with total number of points and
                number of boundary points.

        Returns:
            Array of shape (N[0]*N[1]*..., dims) with coordinates.
        """
        if (
            kind == SampleMethod.RANDOM
            and len(self.submeshes) == 2
            and isinstance(self.submeshes[0], Line)
        ):
            s0, s1 = cast(tuple[Line, Line], self.submeshes)
            N0, N1 = cast(tuple[int, int], N)
            smesh = RectangleShapely(
                left=s0.left, right=s0.right, bottom=s1.left, top=s1.right
            )
            smesh.boundary_factor = N1 / N0
            return smesh.get_all_points(N0, kind=SampleMethod.RANDOM)

        kind_list = _normalize_kind(kind, len(self.submeshes))
        assert len(N) == len(self.submeshes)
        meshes = []
        for mi, Ni, knd in zip(self.submeshes, N, kind_list, strict=True):
            args = (Ni,) if np.isscalar(Ni) else tuple(Ni)
            meshes.append(mi.get_all_points(*args, kind=knd))

        if len(meshes) == 2:
            return jnp.array(
                [jnp.hstack((xi, yi)) for xi in meshes[0] for yi in meshes[1]]
            )
        elif len(meshes) == 3:
            return jnp.array(
                [
                    jnp.hstack((xi, yi, zi))
                    for xi in meshes[0]
                    for yi in meshes[1]
                    for zi in meshes[2]
                ]
            )
        raise NotImplementedError

    def get_points_inside_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        """Return interior points (exclude perimeter).

        Args:
            N: Number of total points for each submesh.
            kind: Lis of the kind of sampling used for submeshes. If a single
                string is provided, it is applied to all submeshes.

        Returns:
            Array of interior coordinates.
        """
        x = self.get_all_points(*N, kind=kind)
        mask = self.boundary_mask(*N, kind=kind)
        return x[mask == False]  # noqa: E712

    def get_points_on_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        """Return boundary points.

        Args:
            N: Number of total points for each submesh.
            kind: List of the kind of sampling used for submeshes.

        Returns:
            Array of boundary points.
        """
        x = self.get_all_points(*N, kind=kind)
        mask = self.boundary_mask(*N, kind=kind)
        return x[mask]

    def get_all_weights(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]:
        """Return weights for all points including boundaries.

        Args:
            N: Total number of weights for each submesh.
            kind: List of the kind of sampling used for submeshes. If a single
                string is provided, it is applied to all submeshes.

        Returns:
            Array of weights for all points.
        """
        kind_list = _normalize_kind(kind, len(self.submeshes))
        assert len(N) == len(self.submeshes)
        weights = []
        for mi, Ni, knd in zip(self.submeshes, N, kind_list, strict=True):
            args = (Ni,) if np.isscalar(Ni) else tuple(Ni)
            weights.append(mi.get_all_weights(*args, kind=knd))

        if len(weights) == 2:
            wx = jnp.outer(weights[0], weights[1])
            if wx.shape == (1, 1):
                return wx.item()
            if 1 in wx.shape:
                return jnp.broadcast_to(wx, N).flatten()
            return wx.flatten()
        elif len(weights) == 3:
            wx = jnp.outer(weights[0], weights[1])
            wx = jnp.outer(wx.flatten(), weights[2])
            if wx.shape == (1, 1):
                return wx.item()
            if 1 in wx.shape:
                return jnp.broadcast_to(wx, N).flatten()
            return wx.flatten()
        raise NotImplementedError

    def get_weights_inside_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]:
        """Return interior weights (exclude perimeter).

        Args:
            N: Total number of weights for each submesh.
            kind: List of the kind of sampling used for submeshes. If a single
                string is provided, it is applied to all submeshes.

        Returns:
            Array of interior weights.
        """
        x = self.get_all_weights(*N, kind=kind)
        mask = self.boundary_mask(*N, kind=kind)
        if not isinstance(x, Array):
            return x
        return x[mask == False]  # noqa: E712

    def get_weights_on_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]:
        """Return boundary weights.

        Args:
            N: Total number of points for each submesh.
            kind: List of the kind of sampling used for submeshes. If a single
                string is provided, it is applied to all submeshes.

        Returns:
            Array of boundary weights.
        """
        x = self.get_all_weights(*N, kind=kind)
        mask = self.boundary_mask(*N, kind=kind)
        if not isinstance(x, Array):
            return x
        return x[mask]


class TimeMarchingMesh(CartesianProductMesh):
    """Time-marching mesh as Cartesian product with time axis along the last
    dimension.

    Attributes:
        deltat: Time step size.
        timestep: Current time step index.
    """

    def __init__(self, *m0: BaseMesh) -> None:
        """Initialize time-marching mesh.

        Args:
            *m0: Submeshes to form the Cartesian product. The last submesh
                must be a Line representing the time axis. This Line represents
                the first time step; subsequent time steps are obtained by
                shifting this Line forward by a constant deltat corresponding
                to the length of the first time interval.
        """
        assert isinstance(m0[-1], Line), "Last submesh must be a Line (time axis)"
        self.deltat: float = float(m0[-1].right - m0[-1].left)
        self.timestep: int = 0
        self._spatial_mesh: ParameterMesh | None = None
        super().__init__(*m0)

    @property
    def dt(self):
        return jnp.hstack((jnp.zeros(self.dim - 1), self.deltat))[None, :]

    def update_time(self):
        """Shift time axis forward by deltat."""
        self.timestep += 1
        time_mesh = cast(Line, self.submeshes[-1])
        time_mesh.left += self.deltat
        time_mesh.right += self.deltat

    def get_points(
        self,
        *N: NPointsType,
        domain: DomainType = "all",
        kind: KindType = "uniform",
    ) -> Array:
        """Return sampled points.

        Args:
            N: Number of total points in mesh. Identical to length of returned
                array in case domain is 'all'. N represents all arguments needed
                for all submeshes.
            domain: 'inside' | 'boundary' | 'all' | 'initial-time' | 'end-time'
            kind: Sampling kind(s). List of kinds for each submesh.
        """
        if domain == "initial-time":
            xi = self.get_points(*N, domain="boundary", kind=kind)
            return xi[xi[:, -1] <= self.submeshes[-1].left + 1e-7]
        elif domain == "end-time":
            xi = self.get_points(*N, domain="boundary", kind=kind)
            return xi[xi[:, -1] >= self.submeshes[-1].right - 1e-7]
        return super().get_points(*N, domain=domain, kind=kind)

    def get_spatial_mesh(self) -> ParameterMesh:
        """Return spatial mesh (excluding time axis)."""
        if self._spatial_mesh is None:
            if len(self.submeshes) == 2:
                self._spatial_mesh = cast(ParameterMesh, self.submeshes[0])
            else:
                self._spatial_mesh = CartesianProductMesh(*self.submeshes[:-1])
        return self._spatial_mesh

    def get_points_at_time(
        self, *N: NPointsType, t: float = 0, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        mesh = self.get_spatial_mesh()
        if isinstance(mesh, MultiParameterMesh):
            x = mesh.get_points(*N, domain="all", kind=kind)
        else:
            x = mesh.get_points(
                cast(int, N[0]), domain="all", kind=cast(SampleMethodLike, kind)
            )
        return jnp.hstack((x, jnp.full((x.shape[0], 1), float(t))))


@dataclass
class Line(SingleParameterMesh):
    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))
    left: float
    right: float

    def __post_init__(self) -> None:
        self.left = float(self.left)
        self.right = float(self.right)
        if self.right <= self.left:
            raise ValueError(
                f"right ({self.right}) must be greater than left ({self.left})"
            )

    @property
    def dim(self) -> int:
        """Return spatial dimension of the mesh."""
        return 1

    def get_all_points(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array:
        """Return N points including boundaries.

        Args:
            N: Number of sample points (including boundaries).
            kind: Sampling strategy.

        Returns:
            Array of shape (N, 1) with coordinates.
        """
        if N < 2:
            raise ValueError("N must be >= 2 for line sampling")

        kind = _coerce_sample_method(kind)

        if kind == SampleMethod.UNIFORM:
            return jnp.linspace(self.left, self.right, N)[:, None]

        elif kind == SampleMethod.LEGENDRE:
            x = (1 + leggauss(N - 2)[0]) / 2  # leggauss(0) is ok.

        elif kind == SampleMethod.CHEBYSHEV:
            x = (
                1
                + jnp.cos(jnp.pi + (2 * jnp.arange(N - 2) + 1) * jnp.pi / (2 * (N - 2)))
            ) / 2

        elif kind == SampleMethod.RANDOM:
            x = jax.random.uniform(self.key, (N - 2,))

        else:
            raise NotImplementedError

        return jnp.hstack(
            (
                jnp.array([self.left]),
                self.left + x * (self.right - self.left),
                jnp.array([self.right]),
            )
        )[:, None]

    def get_points_inside_domain(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array:
        """Return interior points (exclude boundaries).

        Args:
            N: Total number of sample points (including 2 boundary points).
            kind: Sampling strategy.

        Returns:
            Array of shape (N - 2, 1) with interior coordinates.
        """
        return self.get_all_points(N, kind=kind)[1:-1]

    def get_points_on_domain(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array:
        """Return boundary endpoints.

        Args:
            N: Total number of points (ignored, as number of boundary points is
                always 2).
            kind: Sampling strategy. Ignored (kept for API consistency).

        Returns:
            Array [[0.0], [1.0]]
        """
        x = self.get_all_points(2, kind=kind)
        return jnp.vstack((x[0], x[-1]))

    def get_all_weights(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Array | Literal[1]:
        """Return quadrature weights for all N points including boundaries.

        Args:
            N: Number of sample points (including boundaries).
            kind: Sampling kind.

        Returns:
            1 for uniform/random (equal weights) or weight array otherwise.

        """
        if kind is None:
            kind = SampleMethod.UNIFORM
        kind = _coerce_sample_method(kind)

        if kind in (SampleMethod.UNIFORM, SampleMethod.RANDOM):
            return 1
        elif kind == SampleMethod.LEGENDRE:
            return jnp.hstack((jnp.array([1.0]), leggauss(N - 2)[1], jnp.array([1.0])))
        elif kind == SampleMethod.CHEBYSHEV:
            return jnp.hstack(
                (jnp.array([1.0]), jnp.pi / (N - 2) * jnp.ones(N - 2), jnp.array([1.0]))
            )
        raise NotImplementedError

    def get_weights_inside_domain(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Array | Literal[1]:
        """Return quadrature weights for interior points.

        Args:
            N: Total number of weights (including boundary).
            kind: Sampling kind.

        Returns:
            1 for uniform/random (equal weights) or weight array otherwise.
        """
        w = self.get_all_weights(N, kind=kind)
        if isinstance(w, jnp.ndarray):
            return w[1:-1]
        return w

    def get_weights_on_domain(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Literal[1]:
        """Return weights for boundary points (always 1 placeholder)."""
        return 1

    def to_shapely(self) -> LineString:
        """Return shapely LineString for the unit line [0, 1]."""
        return LineString([(self.left, 0.0), (self.right, 0.0)])

    def boundary_mask(
        self, N: int, kind: SampleMethodLike = SampleMethod.UNIFORM
    ) -> Array:
        """Return boolean mask for boundary points.

        Args:
            N: Total number of points.
            kind: Sampling kind (ignored).

        Returns:
            Boolean array of shape (N,) with True for boundary points.
        """
        return jnp.hstack(
            (jnp.array([True]), jnp.zeros(N - 2, dtype=bool), jnp.array([True]))
        )


@dataclass
class UnitLine(Line):
    left: float = field(init=False, default=0.0)
    right: float = field(init=False, default=1.0)


@dataclass
class ShapelyMesh(SingleParameterMesh, ABC):
    """Polygonal domain using Shapely for sampling.

    - Interior: rejection sampling from the bounding box using a prepared polygon
    - Boundary: length-proportional sampling along all boundary segments

    Subclasses must implement `make_polygon()`.

    Attributes:
        seed: Random seed for sampling.
        boundary_factor: Fraction of total points allocated to boundary.

    Note: Only 'random' sampling kind is supported for both interior
        and boundary points.

    """

    seed: int = 101
    boundary_factor: float = 0.25

    @property
    def dim(self) -> int:
        """Return spatial dimension of the mesh."""
        return 2

    @abstractmethod
    def make_polygon(self) -> Polygon: ...

    def boundary_mask(
        self, N: int, kind: SampleMethodLike = SampleMethod.RANDOM
    ) -> Array:
        """Return boolean mask for boundary points.

        Args:
            N: Total number of points.
            kind: Sampling method (only 'random' supported).

        Returns:
            Boolean array of shape (N,) with True for boundary points.
        """
        Ni = int(N * (1 - self.boundary_factor))
        Nb = N - Ni
        return jnp.hstack(
            (
                jnp.zeros(Ni, dtype=bool),
                jnp.ones(Nb, dtype=bool),
            )
        )

    def get_all_points(
        self, N: int, kind: SampleMethodLike = SampleMethod.RANDOM
    ) -> Array:
        """Return all points (N, 2) inside and on the polygon.

        Args:
            N: Total number of points (interior + boundary).
            kind: Sampling method (only 'random' supported).

        Returns:
            Array of shape (N, 2) with all points.
        """
        xi = self.get_points_inside_domain(N, kind=kind)
        xb = self.get_points_on_domain(N, kind=kind)
        return jnp.vstack((xi, xb))

    def get_points_inside_domain(
        self, N: int, kind: SampleMethodLike = SampleMethod.RANDOM
    ) -> Array:
        """Return interior points inside the domain.

        Args:
            N: Total number of points (interior + boundary).
            kind: Sampling method (only 'random' supported).

        Returns:
            Array of shape (Ni, 2) with interior points.
        """
        N = int(N * (1 - self.boundary_factor))
        poly = self.make_polygon()
        lo_x, lo_y, hi_x, hi_y = poly.bounds
        rng = np.random.default_rng(self.seed)

        pts = []
        # Draw points until enough are collected
        chunk = max(8192, N // 2)
        len_pts = lambda p: sum(len(pj) for pj in p)
        while len_pts(pts) < N:
            k = max(chunk, N - len_pts(pts))
            cand = np.empty((k, 2), dtype=float)
            cand[:, 0] = rng.uniform(lo_x, hi_x, size=k)
            cand[:, 1] = rng.uniform(lo_y, hi_y, size=k)
            from shapely import contains, points

            pgeom = points(cand[:, 0], cand[:, 1])
            mask = np.asarray(contains(poly, pgeom), dtype=bool)
            sel = cand[mask]
            if sel.size:
                need = N - len_pts(pts)
                pts.append(sel[:need])

        return jnp.vstack(pts)

    def get_points_on_domain(
        self,
        N: int,
        kind: SampleMethodLike = SampleMethod.RANDOM,
        specific_points: Array | np.ndarray | None = None,
    ) -> Array:
        """Return boundary points along the polygon edges.

        Args:
            N: Total number of points (interior + boundary).
            kind: Sampling method (only 'random' supported).
            specific_points: Optional array of specific boundary points to include.

        Returns:
            Array of shape (Nb, 2) with boundary points.
        """
        N = N - int(N * (1 - self.boundary_factor))
        N = N - (specific_points.shape[0] if specific_points is not None else 0)
        poly = self.make_polygon()
        rings = [poly.exterior] + list(poly.interiors)

        # Build edge list for all rings
        edges = []
        lengths = []
        for ring in rings:
            coords = np.asarray(ring.coords)  # closed ring; last point == first
            A = coords[:-1]
            B = coords[1:]
            segs = np.stack([A, B], axis=1)  # (m, 2, 2)
            L = np.linalg.norm(B - A, axis=1)
            edges.append(segs)
            lengths.append(L)

        edges = np.concatenate(edges, axis=0)
        lengths = np.concatenate(lengths, axis=0)
        probs = lengths / lengths.sum()

        rng = np.random.default_rng(self.seed + 1)  # different stream than interior
        counts = rng.multinomial(N, probs)

        pts = []
        for (a, b), m in zip(edges, counts, strict=True):
            if m == 0:
                continue
            t = rng.random(m)
            pts.append(a + t[:, None] * (b - a))
        pts = jnp.vstack(pts)

        return (
            jnp.vstack((pts, specific_points)) if specific_points is not None else pts
        )

    def get_all_weights(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Literal[1]:
        return 1

    def get_weights_inside_domain(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Literal[1]:
        return 1

    def get_weights_on_domain(
        self, N: int, kind: SampleMethodLike | None = None
    ) -> Literal[1]:
        return 1

    def plot_solution(self, X, values, xb=None, levels=30):
        """Plot solution over polygonal mesh using triangulation.
        Args:
            X      : all sample points (N, 2) = vstack((xi, xb))
            values : solution values at X (N,)
            xb     : optional boundary points to overlay as red dots
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        poly = self.make_polygon()

        tri = mtri.Triangulation(X[:, 0], X[:, 1])

        # Mask triangles whose centroid lies outside the polygon (handles holes)
        centroids = X[tri.triangles].mean(axis=1)
        from shapely import contains, points

        pgeom = points(centroids[:, 0], centroids[:, 1])
        inside = np.asarray(contains(poly, pgeom), dtype=bool)
        mask = ~inside  # mask True means hide triangle
        tri.set_mask(mask)

        fig, ax = plt.subplots(figsize=(6, 6))
        tpc = ax.tripcolor(tri, values, shading="gouraud", cmap="viridis")
        ax.tricontour(tri, values, levels=levels, colors="k", linewidths=0.5)
        if xb is not None:
            ax.plot(xb[:, 0], xb[:, 1], "r.", ms=2, label="boundary")
            ax.legend(loc="lower left")
        ax.set_aspect("equal")
        ax.set_title("Solution w(x,y)")
        fig.colorbar(tpc, ax=ax, shrink=0.8, label="w")
        plt.show()


@dataclass(kw_only=True)
class RectangleShapely(ShapelyMesh):
    """Rectangle as a Shapely polygon.

    Attributes:
        left: Left x-bound.
        right: Right x-bound.
        bottom: Lower y-bound.
        top: Upper y-bound.
    """

    left: float = field(init=True, default=0.0)
    right: float = field(init=True, default=1.0)
    bottom: float = field(init=True, default=0.0)
    top: float = field(init=True, default=1.0)

    def make_polygon(self) -> Polygon:
        return Polygon(
            [
                (self.left, self.bottom),
                (self.right, self.bottom),
                (self.right, self.top),
                (self.left, self.top),
                (self.left, self.bottom),
            ]
        )

    def get_points_on_domain(
        self,
        N: int,
        kind: SampleMethodLike = SampleMethod.RANDOM,
        specific_points: Array | np.ndarray | None = None,
    ) -> Array:
        if specific_points is None:
            specific_points = np.array(
                [
                    [self.left, self.bottom],
                    [self.right, self.bottom],
                    [self.right, self.top],
                    [self.left, self.top],
                ],
                dtype=float,
            )

        pts = super().get_points_on_domain(
            N, kind=kind, specific_points=specific_points
        )

        return pts


@dataclass
class Rectangle(CartesianProductMesh):
    """Affine-mapped rectangle [left, right] x [bottom, top].

    Attributes:
        key: PRNG key for random sampling.
        left: Left x-bound.
        right: Right x-bound.
        bottom: Lower y-bound.
        top: Upper y-bound.
    """

    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))
    left: float
    right: float
    bottom: float
    top: float

    def __post_init__(self) -> None:
        self.left = float(self.left)
        self.right = float(self.right)
        self.bottom = float(self.bottom)
        self.top = float(self.top)
        if not self.right > self.left:
            raise ValueError(
                f"right ({self.right}) must be greater than left ({self.left})"
            )
        if not self.top > self.bottom:
            raise ValueError(
                f"top ({self.top}) must be greater than bottom ({self.bottom})"
            )
        super().__init__(
            Line(self.left, self.right, key=self.key),
            Line(self.bottom, self.top, key=self.key),
        )

    def boundary_mask(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        x = self.get_all_points(*N, kind=kind)
        return (
            (abs(x[:, 0] - self.left) < 1e-8)
            | (abs(x[:, 0] - self.right) < 1e-8)
            | (abs(x[:, 1] - self.bottom) < 1e-8)
            | (abs(x[:, 1] - self.top) < 1e-8)
        )

    def get_all_points(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        """Return all points including bounaries

        Args:
            N: The number of points to sample in each direction
            kind: The kind of points to sample.

        Note:
            When kind is 'random', we interpret the first element of N as
            the total number of points to sample and the second as the number
            of boundary points. Sampling is done radomly in the Cartesian
            domain [left, right]^2, with boundary points placed along the
            edges, including the four corners.

            The number of boundary points will be floored to 4 * (N[1] // 4).

            When kind is None or a list of two SampleMethods, we use the
            standard Cartesian product of two line meshes. The default kind
            is 'uniform' for both dimensions if kind is None.

            If kind is a string other than 'random', we use that kind for both
            dimensions.

        Returns:
            Array of shape (N, 2) with coordinates.
        """
        if (
            isinstance(kind, str | SampleMethod)
            and _coerce_sample_method(kind) == SampleMethod.RANDOM
        ):
            Ni, Nx = cast(tuple[int, int], N)
            smesh = self.to_shapely()
            smesh.boundary_factor = Nx / Ni
            return smesh.get_all_points(Ni, kind=SampleMethod.RANDOM)

        kind_list = _normalize_kind(kind, 2)
        return super().get_all_points(*N, kind=kind_list)

    def get_all_weights(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array | Literal[1]:
        """Return all weights including bounaries

        Args:
            N: The number of weights to sample.
            kind: The kind of weights to sample.

        Note:
            When kind is 'random', the method returns 1.

            When kind is SampleMethod other than 'random' or a list of two
            SampleMethods, we use the standard Cartesian product of two line
            weights. The default kind is 'uniform' for both dimensions.

        Returns:
            Array of shape (N,) or Literal[1] with weights.
        """
        if (
            isinstance(kind, str | SampleMethod)
            and _coerce_sample_method(kind) == SampleMethod.RANDOM
        ):
            return 1
        kind_list = _normalize_kind(kind, 2)
        return super().get_all_weights(*N, kind=kind_list)

    def to_shapely(self) -> ShapelyMesh:
        """Return ShapelyMesh for the rectangle."""
        return RectangleShapely(
            left=self.left, right=self.right, bottom=self.bottom, top=self.top
        )


@dataclass
class UnitSquare(Rectangle):
    """Reference unit square [0, 1]^2.

    Attributes:
        key: PRNG key for random sampling.
    """

    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))
    left: float | int = field(init=False, default=0.0)
    right: float | int = field(init=False, default=1.0)
    bottom: float | int = field(init=False, default=0.0)
    top: float | int = field(init=False, default=1.0)


def points_along_axis(a: Array | float, b: Array | float) -> Array:
    """Return Cartesian product points between 1D arrays a and b.

    Args:
        a: Scalar or 1D array-like.
        b: Scalar or 1D array-like.

    Returns:
        Array of shape (len(a)*len(b), 2) listing all (a_i, b_j) pairs.
    """
    a = jnp.atleast_1d(a)
    b = jnp.atleast_1d(b)
    return jnp.array(jnp.meshgrid(a, b, indexing="ij")).reshape((2, -1)).T


class AnnulusPolar(Rectangle):
    """Annulus in polar coordinates: radius in [r_in, r_out], theta in [0, 2π).

    Sampling in theta wraps for interior points (exclude duplicate 2π).
    """

    def __init__(self, radius_inner: float, radius_outer: float) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        Rectangle.__init__(self, radius_inner, radius_outer, 0, 2 * jnp.pi)

    def get_all_points(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        x = Rectangle.get_all_points(self, *N, kind=kind)
        return x[(abs(x[:, 1] - 2 * jnp.pi) > 1e-8)]

    def get_points_inside_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        x = AnnulusPolar.get_all_points(self, *N, kind=kind)
        return x[
            (abs(x[:, 0] - self.radius_inner) > 1e-6)
            & (abs(x[:, 0] - self.radius_outer) > 1e-6)
        ]

    def get_points_on_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        x = AnnulusPolar.get_all_points(self, *N, kind=kind)
        return x[
            (abs(x[:, 0] - self.radius_inner) < 1e-6)
            | (abs(x[:, 0] - self.radius_outer) < 1e-6)
        ]


class Annulus(AnnulusPolar):
    """Cartesian annulus converted from polar samples.

    Interior/boundary sampling occurs in polar coordinates and is then
    mapped to Cartesian (x, y).
    """

    def __init__(self, radius_inner: float, radius_outer: float) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        AnnulusPolar.__init__(self, radius_inner, radius_outer)

    def convert_to_cartesian(self, xc) -> Array:
        """Convert polar (r, θ) points to Cartesian (x, y) using JAX ops."""
        r = xc[:, 0]
        theta = xc[:, 1]
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        return jnp.column_stack((x, y))

    def get_all_points(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        xc = AnnulusPolar.get_all_points(self, *N, kind=kind)
        return self.convert_to_cartesian(xc)

    def get_points_inside_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        xc = AnnulusPolar.get_points_inside_domain(self, *N, kind=kind)
        return self.convert_to_cartesian(xc)

    def get_points_on_domain(
        self, *N: NPointsType, kind: KindType = SampleMethod.UNIFORM
    ) -> Array:
        xc = AnnulusPolar.get_points_on_domain(self, *N, kind=kind)
        return self.convert_to_cartesian(xc)


@dataclass
class Square_with_hole(ShapelyMesh):
    """Square domain with a circular hole.

    The outer boundary is a square defined by corners (left, bottom) and (right, top).
    The hole is a circle with center (cx, cy) and radius r. Sampling uses:
      - Interior: rejection sampling from the bounding box using a prepared polygon
      - Boundary: length-proportional sampling along all boundary segments
          (outer + hole)

    Attributes:
        left, right, bottom, top: Square bounds (default: [-1, 1] x [-1, 1]).
        cx, cy, r: Circle center and radius (default: (0.3, 0.0), r=0.4).
        hole_resolution: Polygonization resolution for the circle boundary.
        seed: Seed passed to NumPy's default_rng for reproducibility.
    """

    left: float = -1.0
    right: float = 1.0
    bottom: float = -1.0
    top: float = 1.0
    cx: float = 0.3
    cy: float = 0.0
    r: float = 0.4
    hole_resolution: int = 128

    def make_polygon(self) -> Polygon:
        from shapely.geometry import Point, Polygon

        outer = [
            (self.left, self.bottom),
            (self.right, self.bottom),
            (self.right, self.top),
            (self.left, self.top),
        ]
        hole = Point(self.cx, self.cy).buffer(self.r, quad_segs=self.hole_resolution)
        poly = Polygon(outer, holes=[list(hole.exterior.coords)[:-1]])
        if not poly.is_valid or poly.area <= 0.0:
            raise ValueError(
                "Invalid polygon configuration (check square bounds and hole)."
            )
        return poly

    def get_points_on_domain(
        self,
        N: int,
        kind: SampleMethodLike = SampleMethod.RANDOM,
        specific_points: Array | np.ndarray | None = None,
    ) -> Array:
        if specific_points is None:
            specific_points = np.array(
                [
                    [self.left, self.bottom],
                    [self.right, self.bottom],
                    [self.right, self.top],
                    [self.left, self.top],
                ],
                dtype=float,
            )
        pts = super().get_points_on_domain(
            N, kind=kind, specific_points=specific_points
        )
        return pts


@dataclass
class Circle_with_hole(ShapelyMesh):
    """Circular domain with a circular hole.

    The outer boundary is a circle defined by center (Cx, Cy) and radius R.
    The hole is a circle with center (cx, cy) and radius r. Sampling uses:
      - Interior: rejection sampling from the bounding box using a prepared polygon
      - Boundary: length-proportional sampling along all boundary segments

    Attributes:
        Cx, Cy, R: Outer circle center and radius (default: (0.0, 0.0), R=1).
        cx, cy, r: Inner circle center and radius (default: (0.3, 0.0), r=0.4).
        inner_hole_resolution: Polygonization resolution for inner circle.
        outer_hole_resolution: Polygonization resolution for outer circle.
        seed: Seed passed to NumPy's default_rng for reproducibility.
    """

    Cx: float = 0.0
    Cy: float = 0.0
    R: float = 1.0
    cx: float = 0.3
    cy: float = 0.0
    r: float = 0.4
    inner_hole_resolution: int = 64
    outer_hole_resolution: int = 256

    def make_polygon(self) -> Polygon:
        from shapely.geometry import Point

        outer = Point(self.Cx, self.Cy).buffer(
            self.R, quad_segs=self.outer_hole_resolution
        )
        hole = Point(self.cx, self.cy).buffer(
            self.r, quad_segs=self.inner_hole_resolution
        )
        poly = outer.difference(hole)
        if not poly.is_valid or poly.area <= 0.0:
            raise ValueError(
                "Invalid polygon configuration (check circle bounds and hole)."
            )
        return Polygon(poly)


@dataclass
class Triangle(ShapelyMesh):
    """Triangular domain.

    Attributes:

        seed: Seed passed to NumPy's default_rng for reproducibility.
    """

    def make_polygon(self) -> Polygon:
        from shapely.geometry import Polygon

        triangle = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ]
        poly = Polygon(triangle)
        return poly

    def get_points_on_domain(
        self,
        N: int,
        kind: SampleMethodLike = SampleMethod.RANDOM,
        specific_points: Array | np.ndarray | None = None,
    ) -> Array:
        if specific_points is None:
            specific_points = np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            )
        pts = super().get_points_on_domain(
            N, kind=kind, specific_points=specific_points
        )
        return pts


@dataclass
class Lshape(ShapelyMesh):
    """L-shaped polygonal domain.

    Sampling uses:
      - Interior: rejection sampling from the bounding box using a prepared polygon
      - Boundary: length-proportional sampling along all boundary segments

    Attributes:
        seed: Seed passed to NumPy's default_rng for reproducibility.
    """

    left: float = -1.0
    right: float = 1.0
    bottom: float = -1.0
    top: float = 1.0
    Lx: float = 1.0
    Ly: float = 1.0

    def make_polygon(self) -> Polygon:
        from shapely.geometry import Polygon

        outer = [
            (self.left, self.top),
            (self.left, self.bottom),
            (self.right, self.bottom),
            (self.right, self.bottom + self.Ly),
            (self.left + self.Lx, self.bottom + self.Ly),
            (self.left + self.Lx, self.top),
        ]
        poly = Polygon(outer)
        if not poly.is_valid or poly.area <= 0.0:
            raise ValueError("Invalid polygon configuration for L-shape.")
        return poly

    def get_points_on_domain(
        self,
        N: int,
        kind: SampleMethodLike = SampleMethod.RANDOM,
        specific_points: Array | np.ndarray | None = None,
    ) -> Array:
        specific_points = np.array(
            [
                [self.left, self.bottom],
                [self.right, self.bottom],
                [self.left, self.top],
                [self.left + self.Lx, self.bottom + self.Ly],
                [self.left + self.Lx, self.top],
                [self.right, self.bottom + self.Ly],
            ],
            dtype=float,
        )
        return super().get_points_on_domain(
            N, kind=kind, specific_points=specific_points
        )


# Experimental!
class UnionMesh(BaseMesh):  # pragma: no cover
    """Union of multiple meshes for composite domains.

    Attributes:
        meshes: Tuple of mesh objects (e.g., Line, Rectangle, ShapelyMesh).
    """

    def __init__(self, meshes: tuple) -> None:
        self.meshes = meshes
        self.intersection_points = jnp.vstack(
            [self.meshes[i].get_points_on_domain() for i in range(len(self.meshes))]
        )

    def get_points_inside_domain(
        self,
        N: int | tuple[int, ...],
        kind: SampleMethod = SampleMethod.UNIFORM,
        domain: int | None = None,
    ) -> list[Array]:
        """Return interior points from domain.

        Args:
            N: Number of interior points for the specified mesh.
            kind: Sampling kind for all meshes.
            domain: The domain index to sample from.

        Returns:
            Array with interior points (N, d)
        """
        if isinstance(domain, int):
            return self.meshes[domain].get_points_inside_domain(N, kind=kind)
        assert isinstance(N, tuple)
        pts = []
        for i in range(len(self.meshes)):
            zi = self.meshes[i].get_points_inside_domain(N[i], kind=kind)
            pts.append(zi)
        return pts

    def get_points_on_domain(
        self,
        N: int,
        kind: SampleMethod = SampleMethod.UNIFORM,
    ) -> tuple:
        """Return boundary points from all meshes.

        Args:
            N: Tuple of number of boundary points for each mesh.
            kind: Sampling kind for all meshes.

        Returns:
            Arrays with boundary points (N, d)
        """
        pts = [jnp.expand_dims(self.meshes[0].get_points_on_domain(N)[0], -1)]
        for _ in range(1, len(self.meshes) - 1):
            pts.append(None)
        pts.append(jnp.expand_dims(self.meshes[-1].get_points_on_domain(N)[-1], -1))
        return tuple(pts)

    def get_points_on_intersection(
        self, N: int, kind: SampleMethod = SampleMethod.UNIFORM
    ) -> tuple:
        """Return internal boundary points

        Args:
            N: Tuple of number of boundary points for each mesh.
            kind: Sampling kind for all meshes.

        Returns:
            Arrays with boundary points (N, d)
        """
        pts = [jnp.expand_dims(self.meshes[0].get_points_on_domain()[1], -1)]
        for i in range(1, len(self.meshes) - 1):
            pts.append(self.meshes[i].get_points_on_domain())
        pts.append(jnp.expand_dims(self.meshes[-1].get_points_on_domain()[0], -1))
        return tuple(pts)
