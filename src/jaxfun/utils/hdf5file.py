"""HDF5 snapshot output with an XDMF sidecar, for ParaView and VisIt.

One HDF5 file per run holds the tensor-product mesh once and one group per
snapshot. Beside it sits a small XDMF file that describes those arrays to
ParaView as a temporal collection of rectilinear grids, which is what makes the
time slider work. The sidecar is regenerated from the HDF5 file's own contents
after every snapshot, so a run that is killed mid-flight still leaves a pair of
files that open.

Open the `.xdmf` file in ParaView, not the `.h5`. Read either with h5py::

    with h5py.File("snapshots.h5") as f:
        z = f["mesh/z"][:]  # one array per axis
        U = f["snapshots/0000001000/U"][:]  # (nz, ny, nx, 3), t in .attrs

Fields go in with jaxfun's `(x, y, z)` axis order and are stored reversed, as
`(z, y, x)`, with any vector components innermost: that is the order XDMF and
VTK index a rectilinear mesh in.
"""

from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from jaxfun.galerkin.tensorproductspace import TensorProductSpace
    from jaxfun.typing import ArrayLike, MeshKind

__all__ = ("HDF5File", "xdmf_from_hdf5")

# XDMF rather than VTKHDF, which is where VTK's own development goes and would
# need no sidecar, because these meshes are rectilinear -- uniform along Fourier
# directions, clustered along Chebyshev or Legendre ones -- and VTKHDF supports
# neither RectilinearGrid nor StructuredGrid; ParaView 6.1 rejects both. The
# alternatives are worse: resampling onto a uniform ImageData mesh blurs exactly
# the near-wall layer the clustering buys, and an UnstructuredGrid pays for
# explicit connectivity to say what three coordinate arrays already say. If
# VTKHDF grows rectilinear support, only the sidecar has to change.
#
# The XML is spelled the XDMF2 way (`TopologyType=`, `NumberType=`) rather than
# with the XDMF3 `Type=` shorthand: every reader in circulation accepts the
# former, including the Xdmf2-only reader in the VTK wheel the test suite pulls
# in, while only newer ones accept the latter.
#
# Two conventions meet in the sidecar and must not be conflated. `Dimensions` is
# slowest-varying first ("Nz Ny Nx"), while the VXVYVZ geometry lists its
# coordinate arrays in natural axis order, X then Y then Z -- the opposite
# nesting. Hence the transpose on the way in: get it backwards and ParaView
# renders the wall-normal direction as the streamwise one, silently and
# plausibly, which is what `tests/utils/test_hdf5file.py` pins down with three
# different extents per axis.
_TOPOLOGY = {1: "1DRectMesh", 2: "2DRectMesh", 3: "3DRectMesh"}
_GEOMETRY = {1: "VX", 2: "VXVY", 3: "VXVYVZ"}
_MESH_GROUP = "mesh"
_SNAPSHOT_GROUP = "snapshots"
_STEP_KEY = "{:010d}"


def _h5py() -> Any:
    """Return the h5py module, or explain how to install it."""
    try:
        import h5py
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "HDF5 snapshots need h5py, which jaxfun does not install by default. "
            "Add it with `uv sync --extra io`, `uv add h5py` or "
            "`pip install 'jaxfun[io]'`."
        ) from err
    return h5py


def _axis_names(dim: int) -> tuple[str, ...]:
    """Return the mesh dataset names for a `dim`-dimensional grid."""
    return ("x", "y", "z")[:dim] if dim <= 3 else tuple(f"x{i}" for i in range(dim))


def _to_xdmf_order(a: np.ndarray, dtype: Any = None) -> np.ndarray:
    """Return `a` reversed from (x, y, z) to the (z, y, x) order XDMF expects."""
    return np.ascontiguousarray(a.transpose(*reversed(range(a.ndim))), dtype=dtype)


class HDF5File:
    """Write snapshots of fields on a tensor-product mesh, for ParaView.

    The mesh is taken from the function space once, at construction; every
    `write` adds one group of fields at one time. Opening an existing file
    appends to it, which is what a restart from a checkpoint needs, and the
    stored mesh is checked against the space so a restart onto a different grid
    fails instead of mixing two meshes in one series.

    Args:
        path: The HDF5 file. The sidecar is written beside it with an `.xdmf`
            suffix, referring to the HDF5 file by basename so the pair can be
            moved together.
        space: The space the fields live on; only its mesh is used.
        N: Per-axis point counts, as `TensorProductSpace.mesh` takes them.
            Defaults to each axis's quadrature points, which is what
            a plain `backward` transform produces.
        kind: `MeshKind.QUADRATURE` (the default) or `MeshKind.UNIFORM`, which
            must match how the fields being written were evaluated.
        mode: `"a"` to append to an existing file, `"w"` to replace it.
        dtype: Stored precision of the fields. float32 by default, which is
            plenty for pictures and halves the file. The mesh is always stored
            in float64.
        wrap_axes: Axes to close by appending a copy of the first plane at the
            far end of the domain. A Fourier mesh excludes its right endpoint,
            so without this ParaView draws the box one cell short and the
            periodic seam shows. Only correct for an axis that really is
            periodic over the whole domain; on any other axis it invents a
            plane of data.
        wall_axes: Axes to extend with one plane at each end of the domain,
            holding `wall_values`. Gauss quadrature points exclude the
            endpoints, so a wall-bounded direction otherwise has no wall in the
            picture. Only correct when the field takes that constant value on
            the boundary -- true for a space built with homogeneous Dirichlet
            conditions, false in general.
        wall_values: Per-field `(value at the lower end, value at the upper
            end)` used on every axis in `wall_axes`. Defaults to zero, for
            homogeneous Dirichlet data.
        compression: Passed to `h5py.create_dataset`, e.g. `"gzip"`. Off by
            default: it costs wall-clock time in the solver's critical path and
            buys little on turbulent fields.
        name: Name of the temporal collection, as it appears in ParaView.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        space: TensorProductSpace,
        *,
        N: tuple[int | None, ...] | None = None,
        kind: MeshKind | str = "quadrature",
        mode: Literal["a", "w"] = "a",
        dtype: Any = np.float32,
        wrap_axes: Sequence[int] = (),
        wall_axes: Sequence[int] = (),
        wall_values: Mapping[str, tuple[float, float]] | None = None,
        compression: str | None = None,
        name: str = "TimeSeries",
    ) -> None:
        coords = [np.asarray(x, dtype=np.float64) for x in space.mesh(kind, N, False)]
        domains = [
            (float(s.domain.lower), float(s.domain.upper)) for s in space.basespaces
        ]
        self._init(
            path,
            coords,
            domains,
            mode=mode,
            dtype=dtype,
            wrap_axes=wrap_axes,
            wall_axes=wall_axes,
            wall_values=wall_values,
            compression=compression,
            name=name,
        )

    @classmethod
    def from_coords(
        cls,
        path: str | os.PathLike,
        coords: Sequence[ArrayLike],
        domains: Sequence[tuple[float, float]] | None = None,
        **kwargs: Any,
    ) -> HDF5File:
        """Return a writer for a mesh given as one coordinate array per axis.

        The way in when the mesh is already on the host, or when the caller must
        keep every JAX computation off a single process. `domains` defaults to
        the span of each coordinate array, which makes `wrap_axes` and
        `wall_axes` no-ops unless it is given.
        """
        self = cls.__new__(cls)
        x = [np.asarray(c, dtype=np.float64).ravel() for c in coords]
        self._init(
            path,
            x,
            list(domains) if domains is not None else [(c[0], c[-1]) for c in x],
            **kwargs,
        )
        return self

    def _init(
        self,
        path: str | os.PathLike,
        coords: list[np.ndarray],
        domains: Sequence[tuple[float, float]],
        *,
        mode: Literal["a", "w"] = "a",
        dtype: Any = np.float32,
        wrap_axes: Sequence[int] = (),
        wall_axes: Sequence[int] = (),
        wall_values: Mapping[str, tuple[float, float]] | None = None,
        compression: str | None = None,
        name: str = "TimeSeries",
    ) -> None:
        h5py = _h5py()
        self.path = Path(path)
        self.dim = len(coords)
        if self.dim not in _TOPOLOGY:
            raise ValueError(
                f"an XDMF rectilinear mesh is 1D, 2D or 3D, not {self.dim}D"
            )
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"fields must be float32 or float64, not {self.dtype}")
        self.shape = tuple(len(c) for c in coords)
        self.wrap_axes = tuple(sorted(set(wrap_axes)))
        self.wall_axes = tuple(sorted(set(wall_axes)))
        if set(self.wrap_axes) & set(self.wall_axes):
            raise ValueError("an axis cannot be both wrapped and walled")
        for ax in self.wrap_axes + self.wall_axes:
            if not 0 <= ax < self.dim:
                raise ValueError(f"axis {ax} is out of range for a {self.dim}D mesh")
        self.wall_values = dict(wall_values or {})
        self.compression = compression
        self.name = name
        self._domains = [(float(a), float(b)) for a, b in domains]
        self._coords = [self._augment_coord(c, ax) for ax, c in enumerate(coords)]
        self._names = _axis_names(self.dim)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: Any = h5py.File(self.path, mode)
        self._write_mesh()

    # -- mesh ------------------------------------------------------------------

    def _augment_coord(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Return coordinate `x` extended as `wrap_axes`/`wall_axes` ask."""
        lo, hi = self._domains[axis]
        if axis in self.wrap_axes:
            return np.concatenate([x, [hi]])
        if axis in self.wall_axes:
            return np.concatenate([[lo], x, [hi]])
        return x

    def _augment_field(self, a: np.ndarray, values: tuple[float, float]) -> np.ndarray:
        """Return field `a` extended to match the stored mesh."""
        for axis in self.wrap_axes:
            first = np.take(a, [0], axis=axis)
            a = np.concatenate([a, first], axis=axis)
        for axis in self.wall_axes:
            plane = np.shape(a)[:axis] + (1,) + np.shape(a)[axis + 1 :]
            a = np.concatenate(
                [
                    np.full(plane, values[0], a.dtype),
                    a,
                    np.full(plane, values[1], a.dtype),
                ],
                axis=axis,
            )
        return a

    @property
    def mesh_shape(self) -> tuple[int, ...]:
        """Shape of the stored mesh, after any wrapping or walling."""
        return tuple(len(c) for c in self._coords)

    def _write_mesh(self) -> None:
        """Write the mesh, or check it against what the file already holds."""
        if _MESH_GROUP in self._f:
            g = self._f[_MESH_GROUP]
            stored = [np.asarray(g[n]) for n in self._names if n in g]
            if len(stored) != self.dim or any(
                s.shape != c.shape or not np.allclose(s, c)
                for s, c in zip(stored, self._coords, strict=True)
            ):
                have = "x".join(str(len(s)) for s in stored)
                want = "x".join(str(len(c)) for c in self._coords)
                raise ValueError(
                    f"{self.path} holds snapshots on a {have} mesh, this run writes "
                    f"{want}; use a different file or remove this one"
                )
            return
        g = self._f.create_group(_MESH_GROUP)
        for n, c in zip(self._names, self._coords, strict=True):
            g.create_dataset(n, data=c, dtype=np.float64)
        self._f.attrs["dim"] = self.dim
        self._f.attrs["axes"] = list(self._names)
        self._f.attrs["wrap_axes"] = list(self.wrap_axes)
        self._f.attrs["wall_axes"] = list(self.wall_axes)
        self._f.attrs["creator"] = "jaxfun.utils.hdf5file"
        self._f.flush()

    # -- writing ---------------------------------------------------------------

    @property
    def steps(self) -> list[int]:
        """The snapshot steps currently in the file, ascending."""
        if _SNAPSHOT_GROUP not in self._f:
            return []
        return sorted(int(g.attrs["step"]) for g in self._f[_SNAPSHOT_GROUP].values())

    @property
    def xdmf_path(self) -> Path:
        """The sidecar ParaView opens."""
        return self.path.with_suffix(".xdmf")

    def write(
        self,
        fields: Mapping[str, ArrayLike],
        time: float,
        step: int | None = None,
    ) -> None:
        """Write one snapshot and regenerate the sidecar.

        Args:
            fields: Field name to array. An array shaped like the mesh is a
                scalar; one shaped `(d, *mesh_shape)` is a vector, stored as a
                single `(..., d)` dataset so ParaView sees one vector attribute.
            time: Simulation time, which is what the time slider shows.
            step: Snapshot key. Defaults to one past the highest step in the
                file. Writing an existing step replaces it.
        """
        if step is None:
            existing = self.steps
            step = existing[-1] + 1 if existing else 0
        step = int(step)
        # Convert every field before touching the file, so that a bad one leaves
        # the file as it was rather than an old snapshot deleted or a new one
        # half written.
        arrays = {field: self._prepare(field, value) for field, value in fields.items()}
        snaps = self._f.require_group(_SNAPSHOT_GROUP)
        key = _STEP_KEY.format(step)
        if key in snaps:
            warnings.warn(
                f"{self.path}: replacing snapshot {step}; HDF5 does not reclaim the "
                "space, so run h5repack if this happens often",
                stacklevel=2,
            )
            del snaps[key]
        g = snaps.create_group(key)
        g.attrs["step"] = step
        g.attrs["time"] = float(time)
        for field, a in arrays.items():
            g.create_dataset(field, data=a, compression=self.compression)
        self._f.flush()
        self._write_xdmf()

    def _prepare(self, field: str, value: ArrayLike) -> np.ndarray:
        """Return `value` checked against the mesh and laid out for storage."""
        a = np.asarray(value)
        if a.shape == self.shape:
            pass
        elif a.ndim == self.dim + 1 and a.shape[1:] == self.shape:
            # (d, *mesh) -> (*mesh, d): a component axis, innermost in XDMF.
            a = np.moveaxis(a, 0, -1)
        else:
            raise ValueError(
                f"field {field!r} has shape {a.shape}; expected {self.shape} for a "
                f"scalar or {(self.dim, *self.shape)} for a vector"
            )
        walls = self.wall_values.get(field, (0.0, 0.0))
        a = self._augment_field(a, walls)
        if a.ndim > self.dim:  # keep the component axis innermost
            comps = [_to_xdmf_order(a[..., i]) for i in range(a.shape[-1])]
            a = np.stack(comps, axis=-1).astype(self.dtype, copy=False)
            return np.ascontiguousarray(a)
        return _to_xdmf_order(a, self.dtype)

    def _write_xdmf(self) -> None:
        """Rewrite the sidecar from the HDF5 file's contents, atomically."""
        xml = _xdmf_tree(self._f, self.path.name, self.name)
        tmp = self.xdmf_path.with_suffix(".xdmf.tmp")
        tmp.write_bytes(xml)
        os.replace(tmp, self.xdmf_path)

    def close(self) -> None:
        """Close the HDF5 file. The sidecar is already up to date."""
        # An h5py file object is falsy once closed, so this is idempotent.
        if getattr(self, "_f", None):
            self._f.close()

    def __enter__(self) -> HDF5File:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"HDF5File({str(self.path)!r}, mesh={'x'.join(map(str, self.mesh_shape))}, "
            f"snapshots={len(self.steps)})"
        )


# -- the sidecar ---------------------------------------------------------------


def _data_item(
    parent: ET.Element, dims: Sequence[int], precision: int, ref: str
) -> None:
    """Add a `<DataItem>` pointing into the HDF5 file."""
    item = ET.SubElement(
        parent,
        "DataItem",
        Format="HDF",
        NumberType="Float",
        Precision=str(precision),
        Dimensions=" ".join(str(d) for d in dims),
    )
    item.text = ref


def _xdmf_tree(f: Any, h5name: str, name: str) -> bytes:
    """Return the XDMF document describing every snapshot in open file `f`."""
    coords = [f[f"{_MESH_GROUP}/{n}"] for n in _axis_names(int(f.attrs["dim"]))]
    dim = len(coords)
    # Dimensions is slowest-varying first; the geometry lists X, Y, Z.
    grid_dims = [len(c) for c in reversed(coords)]
    root = ET.Element(
        "Xdmf", Version="3.0", attrib={"xmlns:xi": "http://www.w3.org/2001/XInclude"}
    )
    domain = ET.SubElement(root, "Domain")
    series = ET.SubElement(
        domain, "Grid", Name=name, GridType="Collection", CollectionType="Temporal"
    )
    snaps = f.get(_SNAPSHOT_GROUP, {})
    for key in sorted(snaps, key=lambda k: int(snaps[k].attrs["step"])):
        g = snaps[key]
        grid = ET.SubElement(series, "Grid", Name=f"grid_{key}", GridType="Uniform")
        ET.SubElement(grid, "Time", Value=f"{float(g.attrs['time']):.16g}")
        ET.SubElement(
            grid,
            "Topology",
            TopologyType=_TOPOLOGY[dim],
            Dimensions=" ".join(str(d) for d in grid_dims),
        )
        geometry = ET.SubElement(grid, "Geometry", GeometryType=_GEOMETRY[dim])
        for n, c in zip(_axis_names(dim), coords, strict=True):
            _data_item(geometry, (len(c),), 8, f"{h5name}:/{_MESH_GROUP}/{n}")
        for field in sorted(g):
            ds = g[field]
            vector = ds.ndim == dim + 1
            attribute = ET.SubElement(
                grid,
                "Attribute",
                Name=field,
                AttributeType="Vector" if vector else "Scalar",
                Center="Node",
            )
            _data_item(
                attribute,
                ds.shape,
                ds.dtype.itemsize,
                f"{h5name}:/{_SNAPSHOT_GROUP}/{key}/{field}",
            )
    ET.indent(root, "  ")
    return (
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n' + ET.tostring(root) + b"\n"
    )


def xdmf_from_hdf5(path: str | os.PathLike, name: str = "TimeSeries") -> Path:
    """Rebuild the XDMF sidecar for an existing HDF5 snapshot file.

    `HDF5File.write` already does this after every snapshot; this is for a file
    whose sidecar was lost, or one moved next to a differently named HDF5 file.

    Returns:
        The path written.
    """
    h5py = _h5py()
    path = Path(path)
    with h5py.File(path, "r") as f:
        xml = _xdmf_tree(f, path.name, name)
    out = path.with_suffix(".xdmf")
    out.write_bytes(xml)
    return out
