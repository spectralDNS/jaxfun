"""Sampling meshes for 1D/2D reference domains and mapped geometric regions.

Supported domains:
- UnitLine / Line
- UnitSquare / Rectangle
- Annulus (polar -> Cartesian conversion)

Sampling kinds (interior / boundary):
- 'uniform'   : Equidistant interior points (excludes boundary)
- 'legendre'  : Gauss–Legendre nodes (mapped to [0,1])
- 'chebyshev' : Chebyshev nodes of the first kind (mapped to [0,1])
- 'random'    : Pseudorandom uniform samples

Weights:
Return 1 when uniform (each point equal) or arrays for quadrature-based kinds.
"""

from dataclasses import dataclass, field
from numbers import Number
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from flax import nnx
from jax.typing import ArrayLike

from jaxfun.typing import Array, SampleMethod
from jaxfun.utils import leggauss


@dataclass
class UnitLine:
    """Reference 1D line domain [0, 1].

    Attributes:
        key: PRNG key (nnx Rngs) used for random sampling.
    """

    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))

    def get_points_inside_domain(self, N: int, kind: SampleMethod = "uniform") -> Array:
        """Return interior points (exclude 0 and 1).

        Args:
            N: Number of interior sample points (excludes boundaries).
            kind: Sampling strategy: uniform | legendre | chebyshev | random.

        Returns:
            Array of shape (N, 1) with interior coordinates in (0,1).
        """
        if kind == "uniform":
            return jnp.linspace(0, 1, N + 2)[1:-1, None]

        elif kind == "legendre":
            return (1 + leggauss(N)[0][:, None]) / 2

        elif kind == "chebyshev":
            return (1 + jnp.cos(jnp.pi + (2 * jnp.arange(N) + 1) * jnp.pi / (2 * N)))[
                :, None
            ] / 2

        elif kind == "random":
            return jax.random.uniform(self.key, (N, 1))

        raise NotImplementedError

    def get_points_on_domain(self, N: int = 2, kind: SampleMethod = "uniform") -> Array:
        """Return boundary endpoints.

        Args:
            N: Number of boundary points (ignored, always 2).
            kind: Ignored (kept for API consistency).

        Returns:
            Array [[0.0],[1.0]].
        """
        return jnp.array([[0.0], [1.0]])

    def get_weights_inside_domain(
        self, N: int, kind: SampleMethod = "uniform"
    ) -> Array | Literal[1]:
        """Return quadrature weights for interior points.

        Args:
            N: Number of interior weights.
            kind: Sampling kind.

        Returns:
            1 for uniform/random (equal weights) or weight array otherwise.
        """
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            return leggauss(N)[1] * N
        elif kind == "chebyshev":
            return jnp.pi / N * jnp.ones(N)
        raise NotImplementedError

    def get_weights_on_domain(
        self, N: int, kind: SampleMethod = "uniform"
    ) -> Literal[1]:
        """Return weights for boundary points (always 1 placeholder)."""
        return 1


@dataclass
class Line(UnitLine):
    """Affine-mapped 1D line [left, right].

    Attributes:
        left: Left boundary.
        right: Right boundary.
    """

    left: Number
    right: Number

    def __post_init__(self):
        """Validate and coerce boundaries to float."""
        self.left = float(self.left)
        self.right = float(self.right)
        if not self.right > self.left:
            raise ValueError(
                f"right ({self.right}) must be greater than left ({self.left})"
            )

    def get_points_inside_domain(self, N: int, kind: SampleMethod = "uniform") -> Array:
        """Return interior points mapped from reference (0,1)."""
        x = super().get_points_inside_domain(N, kind)
        return self.left + (self.right - self.left) * x

    def get_points_on_domain(self, N: int = 2, kind: SampleMethod = "uniform") -> Array:
        """Return boundary endpoints [[left],[right]]."""
        return jnp.array([[self.left], [self.right]], dtype=float)


@dataclass
class UnitSquare:
    """Reference unit square [0, 1]^2.

    Attributes:
        key: PRNG key for random sampling.
    """

    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))

    def get_points_inside_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform"
    ) -> Array:
        """Return interior points (exclude perimeter).

        Args:
            Nx: Number of interior points along x.
            Ny: Number of interior points along y.
            kind: uniform | legendre | chebyshev | random.

        Returns:
            Array (Nx*Ny, 2) of interior coordinates.
        """
        if kind == "uniform":
            x = jnp.linspace(0, 1, Nx + 2)[1:-1]
            y = jnp.linspace(0, 1, Ny + 2)[1:-1]

        elif kind == "legendre":
            x = (1 + leggauss(Nx)[0]) / 2
            y = (1 + leggauss(Ny)[0]) / 2

        elif kind == "chebyshev":
            x = (1 + jnp.cos(jnp.pi + (2 * jnp.arange(Nx) + 1) * jnp.pi / (2 * Nx))) / 2
            y = (1 + jnp.cos(jnp.pi + (2 * jnp.arange(Ny) + 1) * jnp.pi / (2 * Ny))) / 2

        else:
            assert kind == "random", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            return jax.random.uniform(self.key, (Nx * Ny, 2))

        return jnp.array(jnp.meshgrid(x, y, indexing="ij")).reshape((2, -1)).T

    def get_points_on_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array:
        """Return boundary points in counterclockwise order.

        Args:
            Nx: Number of boundary points along x.
            Ny: Number of boundary points along y.
            kind: uniform | legendre | chebyshev | random.
            corners: If True, append the 4 corner points explicitly.

        Returns:
            Array (#boundary_pts, 2).
        """

        if kind == "uniform":
            x = np.linspace(0, 1, Nx + 2)[1:-1]
            y = np.linspace(0, 1, Ny + 2)[1:-1]
            xy = np.vstack((np.hstack((x, x, y, y)),) * 2).T

        elif kind == "legendre":
            x = (1 + leggauss(Nx)[0]) / 2
            y = (1 + leggauss(Ny)[0]) / 2
            xy = np.vstack((np.hstack((x, x, y, y)),) * 2).T

        elif kind == "chebyshev":
            x = (1 + np.cos(np.pi + (2 * np.arange(Nx) + 1) * np.pi / (2 * Nx))) / 2
            y = (1 + np.cos(np.pi + (2 * np.arange(Ny) + 1) * np.pi / (2 * Ny))) / 2
            xy = np.vstack((np.hstack((x, x, y, y)),) * 2).T

        else:
            assert kind == "random", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            if Nx == 1 or Ny == 1:
                M = jnp.sqrt(max(Nx, Ny)).astype(int)
                xy = np.array(jax.random.uniform(self.key, (4 * M, 2)))
                Nx = M
                Ny = M
            else:
                xy = np.array(jax.random.uniform(self.key, (2 * (Nx + Ny), 2)))

        if corners:
            c = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
            xy = np.vstack((xy, c))

        xy[:Nx, 1] = 0
        xy[Nx : 2 * Nx, 1] = 1
        xy[2 * Nx : (2 * Nx + Ny), 0] = 0
        xy[(2 * Nx + Ny) : (2 * Nx + 2 * Ny), 0] = 1
        return jnp.array(xy)

    def get_weights_inside_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform"
    ) -> Array | Literal[1]:
        """Return quadrature weights for interior nodes.

        Args:
            Nx: Number of interior points along x.
            Ny: Number of interior points along y.
            kind: Sampling kind.

        Returns:
            1 for uniform/random or flattened tensor-product weight array.
        """
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            wx = leggauss(Nx)[1] * Nx
            wy = leggauss(Ny)[1] * Ny
            return jnp.outer(wx, wy).flatten()
        else:
            assert kind == "chebyshev", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            wx = jnp.pi / Nx * jnp.ones(Nx)
            wy = jnp.pi / Ny * jnp.ones(Ny)
            return jnp.outer(wx, wy).flatten()

    def get_weights_on_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array | Literal[1]:
        """Return weights for boundary nodes.

        Args:
            Nx: Number of boundary points along x.
            Ny: Number of boundary points along y.
            kind: Sampling kind.
            corners: Whether 4 corner points are present (affects length).

        Returns:
            1 for uniform/random else 1D weight array.
        """
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            wx = leggauss(Nx)[1] * (2 * Nx + 2 * Ny)
            wy = leggauss(Ny)[1] * (2 * Nx + 2 * Ny)
            w = jnp.hstack((wx, wx, wy, wy))
            if corners:
                w = jnp.hstack((w, jnp.ones(4)))
            return w
        else:
            assert kind == "chebyshev", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            wx = jnp.pi / Nx * (2 * Nx + 2 * Ny) * jnp.ones(Nx)
            wy = jnp.pi / Ny * (2 * Nx + 2 * Ny) * jnp.ones(Ny)
            w = jnp.hstack((wx, wx, wy, wy))
            if corners:
                w = jnp.hstack((w, jnp.ones(4)))
            return w


@dataclass
class Rectangle(UnitSquare):
    """Affine-mapped rectangle [left, right] x [bottom, top].

    Attributes:
        left: Left x-bound.
        right: Right x-bound.
        bottom: Lower y-bound.
        top: Upper y-bound.
    """

    left: Number
    right: Number
    bottom: Number
    top: Number

    def __post_init__(self):
        """Validate and coerce rectangle bounds."""
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

    def get_points_inside_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform"
    ) -> Array:
        """Return interior points mapped from UnitSquare."""
        mesh = super().get_points_inside_domain(Nx, Ny, kind)
        x = self.left + (self.right - self.left) * mesh[:, 0]
        y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
        return jnp.array([x, y]).T

    def get_points_on_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array:
        """Return boundary points mapped from UnitSquare."""
        mesh = super().get_points_on_domain(Nx, Ny, kind, corners=corners)
        x = self.left + (self.right - self.left) * mesh[:, 0]
        y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
        return jnp.array([x, y]).T


def points_along_axis(a: Number | Array, b: Array | Number) -> Array:
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

    def __init__(self, radius_inner: Number, radius_outer: Number) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        Rectangle.__init__(self, radius_inner, radius_outer, 0, 2 * jnp.pi)

    def get_points_inside_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform"
    ) -> Array:
        """Return interior polar points (r, θ)."""
        if kind == "uniform":
            x = jnp.linspace(0, 1, Nx + 2)[1:-1]
            y = jnp.linspace(0, 1, Ny + 1)[:-1]  # wrap around periodic
            mesh = jnp.array(jnp.meshgrid(x, y, indexing="ij")).reshape((2, -1)).T
            x = self.left + (self.right - self.left) * mesh[:, 0]
            y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
            return jnp.array([x, y]).T

        return Rectangle.get_points_inside_domain(self, Nx, Ny, kind)

    def get_points_on_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform", corners: bool = False
    ) -> Array:
        """Return boundary polar points (r=inner/outer)."""
        if kind == "uniform":
            y = np.linspace(0, 1, Ny + 1)[:-1]
            xy = np.vstack((np.hstack((y, y)),) * 2).T
            xy[:Nx, 0] = 0
            xy[Nx : 2 * Nx, 0] = 1
            x = self.left + (self.right - self.left) * xy[:, 0]
            y = self.bottom + (self.top - self.bottom) * xy[:, 1]
            return jnp.array((x, y)).T

        return super().get_points_on_domain(Nx, Ny, kind, False)[2 * Nx :]


class Annulus(AnnulusPolar):
    """Cartesian annulus converted from polar samples.

    Interior/boundary sampling occurs in polar coordinates and is then
    mapped to Cartesian (x, y).
    """

    def __init__(self, radius_inner: Number, radius_outer: Number) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        AnnulusPolar.__init__(self, radius_inner, radius_outer)

    def convert_to_cartesian(self, xc) -> Array:
        """Convert polar (r, θ) points to Cartesian (x, y)."""
        r, theta = sp.symbols("r,theta", real=True, positive=True)
        rv = (r * sp.cos(theta), r * sp.sin(theta))
        mesh = []
        for xi in rv:
            mesh.append(sp.lambdify((r, theta), xi, modules="jax")(*xc.T))
        return jnp.array(mesh).T

    def get_points_inside_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform"
    ) -> Array:
        """Return interior Cartesian points."""
        xc = AnnulusPolar.get_points_inside_domain(self, Nx, Ny, kind)
        return self.convert_to_cartesian(xc)

    def get_points_on_domain(
        self, Nx: int, Ny: int, kind: SampleMethod = "uniform", corners: bool = False
    ) -> Array:
        """Return boundary Cartesian points."""
        xc = AnnulusPolar.get_points_on_domain(self, Nx, Ny, kind, False)
        return self.convert_to_cartesian(xc)


@dataclass
class ShapelyMesh:
    """Polygonal domain using Shapely for sampling.

    - Interior: rejection sampling from the bounding box using a prepared polygon
    - Boundary: length-proportional sampling along all boundary segments

    """

    seed: int = 101

    def make_polygon(self):
        raise NotImplementedError

    def get_points_inside_domain(self, N: int, kind: SampleMethod = "random") -> Array:
        """Return interior points (N, 2) inside the domain."""
        assert kind in ("random",), (
            "Only 'random' interior sampling is supported for polygons."
        )
        from shapely.geometry import Point
        from shapely.prepared import prep

        poly = self.make_polygon()
        prepared = prep(poly)
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
            mask = np.fromiter(
                (prepared.contains(Point(x, y)) for x, y in cand),
                count=k,
                dtype=bool,
            )
            sel = cand[mask]
            if sel.size:
                need = N - len_pts(pts)
                pts.append(sel[:need])

        return jnp.asarray(np.vstack(pts))

    def get_points_on_domain(
        self, N: int, kind: SampleMethod = "random", corners: bool = False
    ) -> Array:
        """Return boundary points (N, 2) along the polygon edges."""
        assert kind in ("random",), (
            "Only 'random' boundary sampling is supported for polygons."
        )

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

        return jnp.asarray(np.vstack(pts))

    def plot_solution(self, X, values, xb=None, levels=30):
        """Plot solution over polygonal mesh using triangulation.
        Args:
            X      : all sample points (N, 2) = vstack((xi, xb))
            values : solution values at X (N,)
            xb     : optional boundary points to overlay as red dots
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from shapely.geometry import Point
        from shapely.prepared import prep

        poly = self.make_polygon()

        tri = mtri.Triangulation(X[:, 0], X[:, 1])

        # Mask triangles whose centroid lies outside the polygon (handles holes)
        prepared = prep(poly)
        centroids = X[tri.triangles].mean(axis=1)
        mask = np.array([not prepared.contains(Point(c[0], c[1])) for c in centroids])
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

    def make_polygon(self):
        from shapely.geometry import Point, Polygon

        outer = [
            (self.left, self.bottom),
            (self.right, self.bottom),
            (self.right, self.top),
            (self.left, self.top),
        ]
        hole = Point(self.cx, self.cy).buffer(self.r, resolution=self.hole_resolution)
        poly = Polygon(outer, holes=[list(hole.exterior.coords)[:-1]])
        if not poly.is_valid or poly.area <= 0.0:
            raise ValueError(
                "Invalid polygon configuration (check square bounds and hole)."
            )
        return poly

    def get_points_on_domain(
        self, N: int, kind: SampleMethod = "random", corners: bool = False
    ) -> Array:
        """Return boundary points (N, 2) along outer square and circular
        hole.
        """
        pts = super().get_points_on_domain(N, kind)

        if corners:
            c = np.array(
                [
                    [self.left, self.bottom],
                    [self.right, self.bottom],
                    [self.right, self.top],
                    [self.left, self.top],
                ],
                dtype=float,
            )
            return jnp.asarray(np.vstack([pts, c]))
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

    def make_polygon(self):
        from shapely.geometry import Point

        outer = Point(self.Cx, self.Cy).buffer(
            self.R, resolution=self.outer_hole_resolution
        )
        hole = Point(self.cx, self.cy).buffer(
            self.r, resolution=self.inner_hole_resolution
        )
        poly = outer.difference(hole)
        if not poly.is_valid or poly.area <= 0.0:
            raise ValueError(
                "Invalid polygon configuration (check circle bounds and hole)."
            )
        return poly


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

    def make_polygon(self):
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
        self, N: int, kind: SampleMethod = "random", corners: bool = False
    ) -> Array:
        """Return boundary points (N, 2) along outer square and circular
        hole.
        """
        pts = super().get_points_on_domain(N, kind)

        if corners:
            c = np.array(
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
            return jnp.asarray(np.vstack([pts, c]))
        return pts
