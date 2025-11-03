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
        N: Number of interior sample points (excludes boundaries).
        key: PRNG key (nnx Rngs) used for random sampling.
    """

    N: int
    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return interior points (exclude 0 and 1).

        Args:
            kind: Sampling strategy: uniform | legendre | chebyshev | random.

        Returns:
            Array of shape (N, 1) with interior coordinates in (0,1).
        """
        if kind == "uniform":
            return jnp.linspace(0, 1, self.N + 2)[1:-1, None]

        elif kind == "legendre":
            return (1 + leggauss(self.N)[0][:, None]) / 2

        elif kind == "chebyshev":
            return (
                1
                + jnp.cos(jnp.pi + (2 * jnp.arange(self.N) + 1) * jnp.pi / (2 * self.N))
            )[:, None] / 2

        elif kind == "random":
            return jax.random.uniform(self.key, (self.N, 1))

        raise NotImplementedError

    def get_points_on_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return boundary endpoints.

        Args:
            kind: Ignored (kept for API consistency).

        Returns:
            Array [[0.0],[1.0]].
        """
        return jnp.array([[0.0], [1.0]])

    def get_weights_inside_domain(
        self, kind: SampleMethod = "uniform"
    ) -> Array | Literal[1]:
        """Return quadrature weights for interior points.

        Args:
            kind: Sampling kind.

        Returns:
            1 for uniform/random (equal weights) or weight array otherwise.
        """
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            return leggauss(self.N)[1] * self.N
        elif kind == "chebyshev":
            return jnp.pi / self.N * jnp.ones(self.N)
        raise NotImplementedError

    def get_weights_on_domain(self, kind: SampleMethod = "uniform") -> Literal[1]:
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

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return interior points mapped from reference (0,1)."""
        x = super().get_points_inside_domain(kind)
        return self.left + (self.right - self.left) * x

    def get_points_on_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return boundary endpoints [[left],[right]]."""
        return jnp.array([[self.left], [self.right]], dtype=float)


@dataclass
class UnitSquare:
    """Reference unit square [0, 1]^2.

    Attributes:
        Nx: Number of interior points along x.
        Ny: Number of interior points along y.
        key: PRNG key for random sampling.
    """

    Nx: int
    Ny: int
    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return interior points (exclude perimeter).

        Args:
            kind: uniform | legendre | chebyshev | random.

        Returns:
            Array (Nx*Ny, 2) of interior coordinates.
        """
        if kind == "uniform":
            x = jnp.linspace(0, 1, self.Nx + 2)[1:-1]
            y = jnp.linspace(0, 1, self.Ny + 2)[1:-1]

        elif kind == "legendre":
            x = (1 + leggauss(self.Nx)[0]) / 2
            y = (1 + leggauss(self.Ny)[0]) / 2

        elif kind == "chebyshev":
            x = (
                1
                + jnp.cos(
                    jnp.pi + (2 * jnp.arange(self.Nx) + 1) * jnp.pi / (2 * self.Nx)
                )
            ) / 2
            y = (
                1
                + jnp.cos(
                    jnp.pi + (2 * jnp.arange(self.Ny) + 1) * jnp.pi / (2 * self.Ny)
                )
            ) / 2

        else:
            assert kind == "random", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            return jax.random.uniform(self.key, (self.Nx * self.Ny, 2))

        return jnp.array(jnp.meshgrid(x, y, indexing="ij")).reshape((2, -1)).T

    def get_points_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array:
        """Return boundary points in counterclockwise order.

        Args:
            kind: uniform | legendre | chebyshev | random.
            corners: If True, append the 4 corner points explicitly.

        Returns:
            Array (#boundary_pts, 2).
        """
        Nx = self.Nx
        Ny = self.Ny

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
        self, kind: SampleMethod = "uniform"
    ) -> Array | Literal[1]:
        """Return quadrature weights for interior nodes.

        Args:
            kind: Sampling kind.

        Returns:
            1 for uniform/random or flattened tensor-product weight array.
        """
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            wx = leggauss(self.Nx)[1] * self.Nx
            wy = leggauss(self.Ny)[1] * self.Ny
            return jnp.outer(wx, wy).flatten()
        else:
            assert kind == "chebyshev", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            wx = jnp.pi / self.Nx * jnp.ones(self.Nx)
            wy = jnp.pi / self.Ny * jnp.ones(self.Ny)
            return jnp.outer(wx, wy).flatten()

    def get_weights_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array | Literal[1]:
        """Return weights for boundary nodes.

        Args:
            kind: Sampling kind.
            corners: Whether 4 corner points are present (affects length).

        Returns:
            1 for uniform/random else 1D weight array.
        """
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            wx = leggauss(self.Nx)[1] * (2 * self.Nx + 2 * self.Ny)
            wy = leggauss(self.Ny)[1] * (2 * self.Nx + 2 * self.Ny)
            w = jnp.hstack((wx, wx, wy, wy))
            if corners:
                w = jnp.hstack((w, jnp.ones(4)))
            return w
        else:
            assert kind == "chebyshev", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            wx = jnp.pi / self.Nx * (2 * self.Nx + 2 * self.Ny) * jnp.ones(self.Nx)
            wy = jnp.pi / self.Ny * (2 * self.Nx + 2 * self.Ny) * jnp.ones(self.Ny)
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

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return interior points mapped from UnitSquare."""
        mesh = super().get_points_inside_domain(kind)
        x = self.left + (self.right - self.left) * mesh[:, 0]
        y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
        return jnp.array([x, y]).T

    def get_points_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array:
        """Return boundary points mapped from UnitSquare."""
        mesh = super().get_points_on_domain(kind, corners=corners)
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

    def __init__(
        self, Nx: int, Ny: int, radius_inner: Number, radius_outer: Number
    ) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        Rectangle.__init__(self, Nx, Ny, radius_inner, radius_outer, 0, 2 * jnp.pi)

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return interior polar points (r, θ)."""
        if kind == "uniform":
            x = jnp.linspace(0, 1, self.Nx + 2)[1:-1]
            y = jnp.linspace(0, 1, self.Ny + 1)[:-1]  # wrap around periodic
            mesh = jnp.array(jnp.meshgrid(x, y, indexing="ij")).reshape((2, -1)).T
            x = self.left + (self.right - self.left) * mesh[:, 0]
            y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
            return jnp.array([x, y]).T

        return Rectangle.get_points_inside_domain(self, kind)

    def get_points_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = False
    ) -> Array:
        """Return boundary polar points (r=inner/outer)."""
        if kind == "uniform":
            y = np.linspace(0, 1, self.Ny + 1)[:-1]
            xy = np.vstack((np.hstack((y, y)),) * 2).T
            xy[: self.Nx, 0] = 0
            xy[self.Nx : 2 * self.Nx, 0] = 1
            x = self.left + (self.right - self.left) * xy[:, 0]
            y = self.bottom + (self.top - self.bottom) * xy[:, 1]
            return jnp.array((x, y)).T

        return super().get_points_on_domain(kind, False)[2 * self.Nx :]


class Annulus(AnnulusPolar):
    """Cartesian annulus converted from polar samples.

    Interior/boundary sampling occurs in polar coordinates and is then
    mapped to Cartesian (x, y).
    """

    def __init__(
        self, Nx: int, Ny: int, radius_inner: Number, radius_outer: Number
    ) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        AnnulusPolar.__init__(self, Nx, Ny, radius_inner, radius_outer)

    def convert_to_cartesian(self, xc) -> Array:
        """Convert polar (r, θ) points to Cartesian (x, y)."""
        r, theta = sp.symbols("r,theta", real=True, positive=True)
        rv = (r * sp.cos(theta), r * sp.sin(theta))
        mesh = []
        for xi in rv:
            mesh.append(sp.lambdify((r, theta), xi, modules="jax")(*xc.T))
        return jnp.array(mesh).T

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        """Return interior Cartesian points."""
        xc = AnnulusPolar.get_points_inside_domain(self, kind)
        return self.convert_to_cartesian(xc)

    def get_points_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = False
    ) -> Array:
        """Return boundary Cartesian points."""
        xc = AnnulusPolar.get_points_on_domain(self, kind, False)
        return self.convert_to_cartesian(xc)
