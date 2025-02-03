import copy
import itertools
from collections.abc import Iterable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax import Array
from scipy import sparse as scipy_sparse

from jaxfun.Basespace import BaseSpace
from jaxfun.composite import BCGeneric, Composite, DirectSum
from jaxfun.coordinates import CoordSys
from jaxfun.Fourier import Fourier
from jaxfun.utils.common import eliminate_near_zeros, lambdify

tensor_product_symbol = "\u2297"
multiplication_sign = "\u00d7"


class TensorProductSpace:
    def __init__(
        self,
        spaces: list[BaseSpace],
        system: CoordSys = None,
        name: str = None,
    ) -> None:
        from jaxfun.arguments import CartCoordSys, x, y, z

        system = (
            CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[len(spaces)])
            if system is None
            else system
        )
        self.spaces = spaces
        self.name = name
        self.system = system
        self.tensorname = tensor_product_symbol.join([b.name for b in spaces])

    def __len__(self) -> int:
        return len(self.spaces)

    def __iter__(self) -> Iterable[BaseSpace]:
        return iter(self.spaces)

    def __getitem__(self, i: int) -> BaseSpace:
        return self.spaces[i]

    @property
    def dims(self) -> int:
        return len(self)

    @property
    def rank(self) -> int:
        return 0

    def shape(self) -> tuple[int]:
        return tuple([n.N + 1 for n in self.spaces])

    def mesh(
        self,
        kind: str = "quadrature",
        N: tuple[int] | None = None,
        broadcast: bool = True,
    ) -> Array:
        """Return mesh in the domain of self"""
        mesh = []
        if N is None:
            N = tuple([s.N for s in self.spaces])
        for ax, space in enumerate(self.spaces):
            X = space.mesh(kind, N[ax])
            mesh.append(self.broadcast_to_ndims(X, ax) if broadcast else X)
        return tuple(mesh)

    def cartesian_mesh(
        self, kind: str = "quadrature", N: tuple[int] | None = None
    ) -> tuple[Array]:
        rv = self.system._position_vector
        x = self.system.base_scalars()
        xj = self.mesh(kind, N, True)
        mesh = []
        for r in rv:
            mesh.append(lambdify(x, r, modules="jax")(*xj))
        return tuple(mesh)

    def broadcast_to_ndims(self, x: Array, axis: int = 0) -> Array:
        """Return 1D array ``x`` as an array of shape according to self"""
        s = [jnp.newaxis] * len(self)
        s[axis] = slice(None)
        return x[tuple(s)]

    @partial(jax.jit, static_argnums=0)
    def evaluate(self, x: list[Array], c: Array) -> Array:
        """Evaluate on a given tensor product mesh"""
        dim: int = len(self)
        if dim == 2:
            for i, (xi, ax) in enumerate(zip(x, range(dim), strict=False)):
                axi: int = dim - 1 - ax
                c = jax.vmap(
                    self.spaces[i].evaluate, in_axes=(None, axi), out_axes=axi
                )(self.spaces[i].map_reference_domain(xi), c)
        else:
            for i, (xi, ax) in enumerate(zip(x, range(dim), strict=False)):
                ax0, ax1 = set(range(dim)) - set((ax,))
                c = jax.vmap(
                    jax.vmap(
                        self.spaces[i].evaluate, in_axes=(None, ax0), out_axes=ax0
                    ),
                    in_axes=(None, ax1),
                    out_axes=ax1,
                )(self.spaces[i].map_reference_domain(xi), c)
        return c

    def get_padded(self, N: tuple[int]) -> "TensorProductSpace":
        paddedspaces = [s.get_padded(n) for s, n in zip(self.spaces, N, strict=False)]
        return TensorProductSpace(
            paddedspaces, system=self.system, name=self.name + "p"
        )

    def backward(
        self, c: Array, kind: str = "quadrature", N: tuple[int] | None = None
    ) -> Array:
        dim: int = len(self)
        has_fft = jnp.any(
            jnp.array([isinstance(space.orthogonal, Fourier) for space in self.spaces])
        )

        # padding in Fourier requires additional effort because we are using the FFT
        # and padding with jnp.fft is not padding the highest wavenumbers, but simply
        # the end of the array.
        if (
            N is not None
            and has_fft
            and jnp.any(
                jnp.array(
                    [n > space.N for n, space in zip(N, self.spaces, strict=False)]
                )
            ).item()
        ):
            shape = list(c.shape)
            for ax, space in enumerate(self.spaces):
                if isinstance(space, Fourier) and N[ax] > space.N:  # padding
                    shape[ax] = N[ax]
            c0 = np.zeros(shape, dtype=c.dtype)
            sl = [slice(0, c.shape[ax]) for ax in range(dim)]
            for ax, space in enumerate(self.spaces):
                if isinstance(space, Fourier):
                    sl[ax] = space.wavenumbers()
                c0[tuple(sl)] = c
            c = jnp.array(c0)

        return self._backward(c, kind, N)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _backward(
        self, c: Array, kind: str = "quadrature", N: tuple[int] | None = None
    ) -> Array:
        dim: int = len(self)

        if dim == 2:
            for ax in range(dim):
                axi: int = dim - 1 - ax
                backward = partial(
                    self.spaces[ax].backward,
                    kind=kind,
                    N=self.spaces[ax].N if N is None else N[ax],
                )
                c = jax.vmap(backward, in_axes=axi, out_axes=axi)(c)
        else:
            for ax in range(dim):
                backward = partial(
                    self.spaces[ax].backward,
                    kind=kind,
                    N=self.spaces[ax] if N is None else N[ax],
                )
                ax0, ax1 = set(range(dim)) - set((ax,))
                c = jax.vmap(
                    jax.vmap(backward, in_axes=ax0, out_axes=ax0),
                    in_axes=ax1,
                    out_axes=ax1,
                )(c)
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
    def rank(self) -> int:
        return 1


def TensorProduct(
    spaces: list[BaseSpace | DirectSum], system: CoordSys = None, name: str = "T"
) -> TensorProductSpace | list[TensorProductSpace]:
    from jaxfun.arguments import CartCoordSys, x, y, z

    system = (
        CartCoordSys("N", {1: (x,), 2: (x, y), 3: (x, y, z)}[len(spaces)])
        if system is None
        else system
    )
    # Use the same coordinates in BaseSpaces as in TensorProductSpace
    for i, space in enumerate(spaces):
        space.system = system.sub_system(i)
        if isinstance(space, Composite):
            space.orthogonal.system = system.sub_system(i)
        if isinstance(space, DirectSum):
            space.spaces[0].system = system.sub_system(i)
            if isinstance(space.spaces[0], Composite):
                space.spaces[0].orthogonal.system = system.sub_system(i)
            space.spaces[1].system = system.sub_system(i)
            space.spaces[1].orthogonal.system = system.sub_system(i)

    if isinstance(spaces[0], DirectSum) or isinstance(spaces[1], DirectSum):
        return DirectSumTPS(spaces, system, name)
    return TensorProductSpace(spaces, system, name)


class DirectSumTPS(TensorProductSpace):
    """TensorProductSpace where one or more 1D space contains non-zero boundary
    conditions and thus a DirectSum space.

    Creates a list of TensorProductSpaces.

    """

    def __init__(
        self,
        spaces: list[BaseSpace | DirectSum],
        system: CoordSys = None,
        name: str = None,
    ) -> None:
        from jaxfun.inner import project1D

        self.spaces = spaces
        self.system = system
        self.name = name
        self.bndvals = {}
        self.tensorname = tensor_product_symbol.join([b.name for b in spaces])

        for space in spaces:
            if space.bcs is None:
                continue
            if space.bcs.is_homogeneous():
                continue
            if isinstance(space, DirectSum):
                s0 = space.spaces[1]
                for val in s0.bcs.values():
                    for key, v in val.items():
                        if len(sp.sympify(v).free_symbols) > 0:
                            val[key] = system.expr_psi_to_base_scalar(v)

        two_inhomogeneous = False
        if isinstance(spaces[0], DirectSum) and isinstance(spaces[1], DirectSum):
            bcspaces = (spaces[0].spaces[1], spaces[1].spaces[1])
            two_inhomogeneous = bcspaces
            bc0, bc1 = bcspaces
            bc0bcs = copy.deepcopy(bc0.bcs)
            bc1bcs = copy.deepcopy(bc1.bcs)

            lr = lambda bcz, z: {"left": bcz.domain.lower, "right": bcz.domain.upper}[z]
            # Set boundary values for boundary spaces
            bcall = []
            for bcthis, bcother, zother in zip(
                [bc0bcs, bc1bcs], [bc1bcs, bc0bcs], [bc1, bc0], strict=False
            ):
                bcall.append([])
                df = 2.0 / (zother.domain.upper - zother.domain.lower)
                for bcval in bcthis.orderedvals():
                    bcs = copy.deepcopy(bcother)
                    for lr_other, bco in bcs.items():
                        z = lr(zother, lr_other)
                        for key in bco:
                            if key == "D":
                                bco[key] = float(
                                    system.expr_base_scalar_to_psi(bcval).subs(
                                        zother.system._psi[0], z
                                    )
                                )
                            elif key[0] == "N":
                                var = zother.system._psi[0]
                                nd = 1 if len(key) == 1 else int(key[1])
                                bco[key] = float(
                                    (
                                        system.expr_base_scalar_to_psi(bcval).diff(
                                            var, nd
                                        )
                                        / df**nd
                                    ).subs(var, z)
                                )
                    bcall[-1].append(bcs)

            self.bndvals[bcspaces] = jnp.array([z.orderedvals() for z in bcall[0]])

        self.tpspaces = self.split(spaces)
        # Compute all the known coefficients of all spaces
        for tensorspace in self.tpspaces:
            otherspaces = [p for p in tensorspace if not isinstance(p, BCGeneric)]
            bcspaces = [p for p in tensorspace if isinstance(p, BCGeneric)]
            bcsindex = [
                i for i, p in enumerate(tensorspace) if isinstance(p, BCGeneric)
            ]
            if len(otherspaces) == 0:
                continue

            elif len(otherspaces) == 1:
                assert len(bcspaces) == 1
                bcspace = bcspaces[0]
                otherspace = otherspaces[0]
                uh = []
                for j, bc in enumerate(bcspace.bcs.orderedvals()):
                    otherspace = otherspaces[0]
                    if two_inhomogeneous:
                        bco = copy.deepcopy(two_inhomogeneous[(bcsindex[0] + 1) % 2])
                        bco.bcs = bcall[bcsindex[0]][j]
                        otherspace = otherspace + bco
                    uh.append(project1D(bc, otherspace))

                if bcsindex[0] == 0:
                    self.bndvals[tensorspace] = jnp.array(uh)
                else:
                    self.bndvals[tensorspace] = jnp.array(uh).T

    def split(self, spaces: list[BaseSpace | DirectSum]) -> dict:
        f = []
        for space in spaces:
            if isinstance(space, DirectSum):
                f.append(space)
            else:
                f.append([space])
        tensorspaces = itertools.product(*f)
        return {
            s: TensorProductSpace(s, name=self.name + f"{i}", system=self.system)
            for i, s in enumerate(tensorspaces)
        }

    def backward(self, c: Array, kind: str = "quadrature", N: int = 0) -> Array:
        a = []
        for f, v in self.tpspaces.items():
            if jnp.any(jnp.array([isinstance(s, BCGeneric) for s in v.spaces])):
                a.append(v.backward(self.bndvals[f], kind=kind, N=N))
            else:
                a.append(v.backward(c, kind=kind, N=N))
        return jnp.sum(jnp.array(a), axis=0)


# NOTE: Could this be a DataClass / NamedTuple?
class TPMatrix:  # noqa: B903
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


def tpmats_to_scipy_sparse_list(
    A: list[TPMatrix], tol: int = 100
) -> list[scipy_sparse.csc_array]:
    return [
        scipy_sparse.csc_array(eliminate_near_zeros(mat, tol))
        for a in A
        for mat in a.mats
    ]
