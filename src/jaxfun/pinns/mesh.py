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
    N: int
    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
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
        return jnp.array([[0.0], [1.0]])

    def get_weights_inside_domain(
        self, kind: SampleMethod = "uniform"
    ) -> Array | Literal[1]:
        if kind in ("uniform", "random"):
            return 1
        elif kind == "legendre":
            return leggauss(self.N)[1] * self.N
        elif kind == "chebyshev":
            return jnp.pi / self.N * jnp.ones(self.N)
        raise NotImplementedError

    def get_weights_on_domain(self, kind: SampleMethod = "uniform") -> Literal[1]:
        return 1


@dataclass
class Line(UnitLine):
    left: Number
    right: Number

    def __post_init__(self):
        self.left = float(self.left)
        self.right = float(self.right)
        if not self.right > self.left:
            raise ValueError(
                f"right ({self.right}) must be greater than left ({self.left})"
            )

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        x = super().get_points_inside_domain(kind)
        return self.left + (self.right - self.left) * x

    def get_points_on_domain(self, kind: SampleMethod = "uniform") -> Array:
        return jnp.array([[self.left], [self.right]], dtype=float)


@dataclass
class UnitSquare:
    Nx: int
    Ny: int
    key: ArrayLike = field(kw_only=True, default_factory=nnx.rnglib.Rngs(101))

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
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
        if kind == "uniform":
            x = np.linspace(0, 1, self.Nx + 2)[1:-1]
            y = np.linspace(0, 1, self.Ny + 2)[1:-1]
            xy = np.vstack((np.hstack((x, x, y, y)),) * 2).T

        elif kind == "legendre":
            x = (1 + leggauss(self.Nx)[0]) / 2
            y = (1 + leggauss(self.Ny)[0]) / 2
            xy = np.vstack((np.hstack((x, x, y, y)),) * 2).T

        elif kind == "chebyshev":
            x = (
                1 + np.cos(np.pi + (2 * np.arange(self.Nx) + 1) * np.pi / (2 * self.Nx))
            ) / 2
            y = (
                1 + np.cos(np.pi + (2 * np.arange(self.Ny) + 1) * np.pi / (2 * self.Ny))
            ) / 2
            xy = np.vstack((np.hstack((x, x, y, y)),) * 2).T

        else:
            assert kind == "random", (
                "Only 'uniform', 'legendre', 'chebyshev' and 'random' are supported"
            )
            xy = np.array(jax.random.uniform(self.key, (2 * (self.Nx + self.Ny), 2)))

        if corners:
            c = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
            xy = np.vstack((xy, c))

        xy[: self.Nx, 1] = 0
        xy[self.Nx : 2 * self.Nx, 1] = 1
        xy[2 * self.Nx : (2 * self.Nx + self.Ny), 0] = 0
        xy[(2 * self.Nx + self.Ny) : (2 * self.Nx + 2 * self.Ny), 0] = 1
        return jnp.array(xy)

    def get_weights_inside_domain(
        self, kind: SampleMethod = "uniform"
    ) -> Array | Literal[1]:
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
    left: Number
    right: Number
    bottom: Number
    top: Number

    def __post_init__(self):
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
        mesh = super().get_points_inside_domain(kind)
        x = self.left + (self.right - self.left) * mesh[:, 0]
        y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
        return jnp.array([x, y]).T

    def get_points_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = True
    ) -> Array:
        mesh = super().get_points_on_domain(kind, corners=corners)
        x = self.left + (self.right - self.left) * mesh[:, 0]
        y = self.bottom + (self.top - self.bottom) * mesh[:, 1]
        return jnp.array([x, y]).T


def points_along_axis(a: Number | Array, b: Array | Number) -> Array:
    a = jnp.atleast_1d(a)
    b = jnp.atleast_1d(b)
    return jnp.array(jnp.meshgrid(a, b, indexing="ij")).reshape((2, -1)).T


class AnnulusPolar(Rectangle):
    def __init__(
        self, Nx: int, Ny: int, radius_inner: Number, radius_outer: Number
    ) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        Rectangle.__init__(self, Nx, Ny, radius_inner, radius_outer, 0, 2 * jnp.pi)

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
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
    def __init__(
        self, Nx: int, Ny: int, radius_inner: Number, radius_outer: Number
    ) -> None:
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        AnnulusPolar.__init__(self, Nx, Ny, radius_inner, radius_outer)

    def convert_to_cartesian(self, xc) -> Array:
        r, theta = sp.symbols("r,theta", real=True, positive=True)
        rv = (r * sp.cos(theta), r * sp.sin(theta))
        mesh = []
        for xi in rv:
            mesh.append(sp.lambdify((r, theta), xi, modules="jax")(*xc.T))
        return jnp.array(mesh).T

    def get_points_inside_domain(self, kind: SampleMethod = "uniform") -> Array:
        xc = AnnulusPolar.get_points_inside_domain(self, kind)
        return self.convert_to_cartesian(xc)

    def get_points_on_domain(
        self, kind: SampleMethod = "uniform", corners: bool = False
    ) -> Array:
        xc = AnnulusPolar.get_points_on_domain(self, kind, False)
        return self.convert_to_cartesian(xc)
