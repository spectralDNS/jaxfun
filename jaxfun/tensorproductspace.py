from functools import partial
from typing import Iterable
from scipy import sparse as scipy_sparse
import jax
from jax import Array
import jax.numpy as jnp
from jaxfun.Basespace import BaseSpace
from jaxfun.coordinates import CoordSys
from jaxfun.utils.common import lambdify, eliminate_near_zeros

tensor_product_symbol = "\u2297"
multiplication_sign = "\u00d7"


class TensorProductSpace:
    def __init__(
        self,
        spaces: list[BaseSpace],
        coordinates: CoordSys = None,
        name: str = None,
    ) -> None:
        from jaxfun.arguments import CartCoordSys

        self.spaces = spaces
        self.name = name
        self.system = (
            CartCoordSys("N")[len(spaces)] if coordinates is None else coordinates
        )
        self.tensorname = tensor_product_symbol.join([b.name for b in spaces])
        self.spacemap = {
            key: val for key, val in zip(self.system._base_scalars, spaces)
        }

    def __len__(self) -> int:
        return len(self.spaces)

    def __iter__(self) -> Iterable[BaseSpace]:
        return iter(self.spaces)

    def __getitem__(self, i: int) -> BaseSpace:
        return self.spaces[i]

    @property
    def rank(self):
        return 0

    def mesh(
        self, kind: str = "quadrature", N: int = 0, broadcast: bool = True
    ) -> Array:
        """Return mesh in the domain of self"""
        mesh = []
        for ax, space in enumerate(self.spaces):
            X = space.mesh(kind, N)
            mesh.append(self.broadcast_to_ndims(X, ax) if broadcast else X)
        return tuple(mesh)

    def cartesian_mesh(self, kind: str = "quadrature", N: int = 0):
        rv = self.system._position_vector
        x = self.system.base_scalars()
        xj = self.mesh(kind, N, True)
        mesh = []
        for r in rv:
            mesh.append(lambdify(x, r, modules="jax")(*xj))
        return tuple(mesh)

    def broadcast_to_ndims(self, x: Array, axis: int = 0):
        """Return 1D array ``x`` as an array of shape according to self"""
        s = [jnp.newaxis] * len(self)
        s[axis] = slice(None)
        return x[tuple(s)]

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def evaluate(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        if kind == "quadrature":
            mesh = [s.quad_points_and_weights()[0] for s in self]
        else:
            mesh = [
                jnp.linspace(
                    float(d.reference_domain.lower), float(d.reference_domain.upper), N
                )
                for d in self
            ]
        dim: int = len(self)
        if dim == 2:
            for i, (xi, ax) in enumerate(zip(mesh, range(dim))):
                axi: int = dim - 1 - ax
                c = jax.vmap(
                    self.spaces[i].evaluate, in_axes=(None, axi), out_axes=axi
                )(xi, c)
        else:
            for i, (xi, ax) in enumerate(zip(mesh, range(dim))):
                ax0, ax1 = set(range(dim)) - set((ax,))
                c = jax.vmap(
                    jax.vmap(
                        self.spaces[i].evaluate, in_axes=(None, ax0), out_axes=ax0
                    ),
                    in_axes=(None, ax1),
                    out_axes=ax1,
                )(xi, c)
        return c


class VectorTensorProductSpace:
    def __init__(
        self,
        tensorspace: TensorProductSpace | tuple[TensorProductSpace],
        name: str = None,
    ) -> None:
        if not isinstance(tensorspace, tuple):
            assert isinstance(tensorspace, TensorProductSpace)
            tensorspace = (tensorspace,) * len(tensorspace)
        self.tensorspaces = tensorspace
        self.system = self.tensorspaces[0].system
        self.name = name
        self.tensorname = multiplication_sign.join([b.name for b in self.tensorspaces])

    def __len__(self) -> int:
        return len(self.tensorspaces)

    def __iter__(self) -> Iterable[TensorProductSpace]:
        return iter(self.tensorspaces)

    def __getitem__(self, i: int) -> TensorProductSpace:
        return self.tensorspaces[i]

    @property
    def rank(self):
        return 1


class TPMatrix:
    def __init__(
        self,
        mats: list[Array],
        scale: float,
        test_space: TensorProductSpace,
        trial_space: TensorProductSpace,
    ) -> None:
        self.mats = mats
        self.scale = scale
        self.test_space = test_space
        self.trial_space = trial_space


def tpmats_to_scipy_sparse_list(A: list[TPMatrix]) -> list[scipy_sparse.csc_array]:
    return [
        scipy_sparse.csc_array(eliminate_near_zeros(mat, 1000))
        for a in A
        for mat in a.mats
    ]
