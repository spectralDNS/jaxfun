import xml.etree.ElementTree as ET
from typing import cast
from xml.etree.ElementTree import Element

import numpy as np
import pytest

from jaxfun.galerkin import FunctionSpace, TensorProduct, TensorProductSpace
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.utils import Domain
from jaxfun.utils.hdf5file import HDF5File, xdmf_from_hdf5

h5py = pytest.importorskip("h5py")

# Distinct extents in every direction: a transposed write then either raises on
# shape or moves the variation to the wrong axis. Equal extents would let
# (x, y, z) and (z, y, x) pass interchangeably, which is the whole risk here.
NX, NY, NZ = 4, 6, 10
DIRICHLET = {"left": {"D": 0}, "right": {"D": 0}}


def space_3d() -> TensorProductSpace:
    Fx = FunctionSpace(NX, Fourier, domain=Domain(0, 2), name="Fx")
    Fy = FunctionSpace(NY, Fourier, domain=Domain(0, 3), name="Fy")
    D = FunctionSpace(NZ, Legendre, bcs=DIRICHLET, name="D")
    return TensorProduct(Fx, Fy, D, name="V3")


def space_2d() -> TensorProductSpace:
    Fx = FunctionSpace(NX, Fourier, domain=Domain(0, 2), name="Fx")
    D = FunctionSpace(NY, Legendre, bcs=DIRICHLET, name="D")
    return TensorProduct(Fx, D, name="V2")


def coords(V: TensorProductSpace) -> list[np.ndarray]:
    return [np.asarray(x) for x in V.mesh(broadcast=False)]


def ramps(V: TensorProductSpace) -> dict[str, np.ndarray]:
    """One field varying along each axis only, so ordering is observable."""
    x = coords(V)
    shape = tuple(len(c) for c in x)
    return {
        f"f{ax}": np.broadcast_to(
            c.reshape((1,) * ax + (-1,) + (1,) * (len(x) - ax - 1)), shape
        ).copy()
        for ax, c in enumerate(x)
    }


def test_index_order_3d(tmp_path) -> None:
    V = space_3d()
    x, y, z = coords(V)
    fields = ramps(V)

    with HDF5File(tmp_path / "t.h5", V, dtype=np.float64) as f:
        f.write(fields, time=0.5, step=7)
        assert f.steps == [7]

    with h5py.File(tmp_path / "t.h5") as h:
        g = h["snapshots/0000000007"]
        assert g.attrs["time"] == 0.5 and g.attrs["step"] == 7
        for name in fields:
            # Stored slowest-axis-first: (z, y, x), not (x, y, z).
            assert g[name].shape == (NZ, NY, NX)
        assert np.allclose(g["f0"][0, 0, :], x)  # x varies fastest
        assert np.allclose(g["f1"][0, :, 0], y)
        assert np.allclose(g["f2"][:, 0, 0], z)  # z varies slowest
        assert np.ptp(g["f0"][:, :, 0]) == 0  # f0 is flat in y and z
        assert np.ptp(g["f2"][0]) == 0  # f2 is flat in x and y
        for name, c in zip(("x", "y", "z"), (x, y, z), strict=True):
            assert np.allclose(h[f"mesh/{name}"], c)
            assert h[f"mesh/{name}"].dtype == np.float64

    root = ET.parse(tmp_path / "t.xdmf").getroot()
    topology = root.find(".//Topology")
    assert topology is not None
    assert topology.get("TopologyType") == "3DRectMesh"
    assert topology.get("Dimensions") == f"{NZ} {NY} {NX}"
    geometry = root.find(".//Geometry")
    assert geometry is not None
    assert geometry.get("GeometryType") == "VXVYVZ"
    # The geometry lists X, Y, Z -- the opposite nesting from Dimensions.
    assert [d.get("Dimensions") for d in geometry] == [str(NX), str(NY), str(NZ)]
    assert [d.text for d in geometry] == [f"t.h5:/mesh/{n}" for n in "xyz"]
    attribute = root.find(".//Attribute")
    assert attribute is not None
    assert attribute.get("AttributeType") == "Scalar"
    assert attribute.get("Center") == "Node"
    assert attribute[0].get("Dimensions") == f"{NZ} {NY} {NX}"
    assert attribute[0].get("Precision") == "8"


def test_index_order_2d(tmp_path) -> None:
    V = space_2d()
    x, y = coords(V)
    with HDF5File(tmp_path / "t.h5", V) as f:
        f.write(ramps(V), time=0.0, step=0)

    with h5py.File(tmp_path / "t.h5") as h:
        g = h["snapshots/0000000000"]
        assert g["f0"].shape == (NY, NX)
        assert np.allclose(g["f0"][0, :], x)
        assert np.allclose(g["f1"][:, 0], y)

    root = ET.parse(tmp_path / "t.xdmf").getroot()
    assert root is not None
    assert (
        cast(Element[str], root.find(".//Topology")).get("TopologyType") == "2DRectMesh"
    )
    assert (
        cast(Element[str], root.find(".//Topology")).get("Dimensions") == f"{NY} {NX}"
    )
    geometry = root.find(".//Geometry")
    assert geometry is not None
    assert geometry.get("GeometryType") == "VXVY"
    assert [d.get("Dimensions") for d in geometry] == [str(NX), str(NY)]


def test_vector_field(tmp_path) -> None:
    V = space_3d()
    u = np.stack([np.asarray(v) for v in ramps(V).values()])
    with HDF5File(tmp_path / "t.h5", V) as f:
        f.write({"U": u}, time=1.0, step=1)

    with h5py.File(tmp_path / "t.h5") as h:
        ds = h["snapshots/0000000001/U"]
        assert ds.shape == (NZ, NY, NX, 3)
        assert ds.dtype == np.float32
        # Components stay in (u, v, w) order and keep their own axis ordering.
        for component in range(3):
            assert np.allclose(ds[..., component], u[component].transpose(2, 1, 0))

    attribute = ET.parse(tmp_path / "t.xdmf").getroot().find(".//Attribute")
    assert attribute is not None
    assert attribute.get("AttributeType") == "Vector"
    assert attribute[0].get("Dimensions") == f"{NZ} {NY} {NX} 3"
    assert attribute[0].get("Precision") == "4"


def test_float32_is_the_default(tmp_path) -> None:
    V = space_3d()
    with HDF5File(tmp_path / "t.h5", V) as f:
        f.write(ramps(V), time=0.0, step=0)
    with h5py.File(tmp_path / "t.h5") as h:
        assert h["snapshots/0000000000/f0"].dtype == np.float32
        assert h["mesh/x"].dtype == np.float64


def test_append_across_sessions(tmp_path) -> None:
    V, path = space_3d(), tmp_path / "t.h5"
    fields = ramps(V)
    with HDF5File(path, V) as f:
        f.write(fields, time=0.1, step=10)
        # The sidecar is complete before close(), so a killed run still opens.
        assert len(ET.parse(f.xdmf_path).getroot().findall(".//Grid/Grid")) == 1
        f.write(fields, time=0.2, step=20)
    with HDF5File(path, V) as f:
        assert f.steps == [10, 20]
        f.write(fields, time=0.3)  # step defaults to one past the last
        assert f.steps == [10, 20, 21]

    grids = ET.parse(tmp_path / "t.xdmf").getroot().findall(".//Grid/Grid")
    assert [g.get("Name") for g in grids] == [
        "grid_0000000010",
        "grid_0000000020",
        "grid_0000000021",
    ]
    assert [cast(Element[str], g.find("Time")).get("Value") for g in grids] == [
        "0.1",
        "0.2",
        "0.3",
    ]


def test_rewritten_step_replaces(tmp_path) -> None:
    V = space_3d()
    shape = (NX, NY, NZ)
    with HDF5File(tmp_path / "t.h5", V) as f:
        f.write({"u": np.zeros(shape)}, time=0.0, step=3)
        with pytest.warns(UserWarning, match="replacing snapshot 3"):
            f.write({"u": np.ones(shape)}, time=0.5, step=3)
        assert f.steps == [3]
    with h5py.File(tmp_path / "t.h5") as h:
        assert np.all(np.asarray(h["snapshots/0000000003/u"]) == 1.0)
    assert len(ET.parse(tmp_path / "t.xdmf").getroot().findall(".//Grid/Grid")) == 1


def test_grid_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "t.h5"
    with HDF5File(path, space_3d()) as f:
        f.write(ramps(space_3d()), time=0.0, step=0)
    Fx = FunctionSpace(NX, Fourier, domain=Domain(0, 2), name="Fx")
    Fy = FunctionSpace(NY, Fourier, domain=Domain(0, 3), name="Fy")
    D = FunctionSpace(NZ + 2, Legendre, bcs=DIRICHLET, name="D")
    with pytest.raises(ValueError, match="holds snapshots on a"):
        HDF5File(path, TensorProduct(Fx, Fy, D, name="Vother"))


def test_wrap_and_wall_axes(tmp_path) -> None:
    V = space_3d()
    x, _, z = coords(V)
    fields = ramps(V)
    with HDF5File(
        tmp_path / "t.h5", V, wrap_axes=(0, 1), wall_axes=(2,), dtype=np.float64
    ) as f:
        assert f.mesh_shape == (NX + 1, NY + 1, NZ + 2)
        f.write(fields, time=0.0, step=0)

    with h5py.File(tmp_path / "t.h5") as h:
        assert np.isclose(h["mesh/x"][-1], 2.0)  # the periodic domain closes
        assert np.allclose(h["mesh/x"][:-1], x)
        assert np.allclose(h["mesh/z"][[0, -1]], (-1.0, 1.0))  # walls appear
        assert np.allclose(h["mesh/z"][1:-1], z)
        u = np.asarray(h["snapshots/0000000000/f2"])
        assert u.shape == (NZ + 2, NY + 1, NX + 1)
        assert np.all(u[0] == 0.0) and np.all(u[-1] == 0.0)
        # The wrapped planes repeat the first, so the seam closes.
        assert np.allclose(u[:, :, 0], u[:, :, -1])
        assert np.allclose(u[:, 0, :], u[:, -1, :])


def test_wall_values_override(tmp_path) -> None:
    V = space_3d()
    with HDF5File(
        tmp_path / "t.h5", V, wall_axes=(2,), wall_values={"T": (1.0, -1.0)}
    ) as f:
        f.write({"T": np.zeros((NX, NY, NZ)), "u": np.zeros((NX, NY, NZ))}, time=0.0)
    with h5py.File(tmp_path / "t.h5") as h:
        T = np.asarray(h["snapshots/0000000000/T"])
        assert np.all(T[0] == 1.0) and np.all(T[-1] == -1.0)
        u = np.asarray(h["snapshots/0000000000/u"])
        assert np.all(u[0] == 0.0) and np.all(u[-1] == 0.0)


def test_bad_field_shape_raises(tmp_path) -> None:
    V = space_3d()
    with (
        HDF5File(tmp_path / "t.h5", V) as f,
        pytest.raises(ValueError, match="expected"),
    ):
        f.write({"u": np.zeros((NZ, NY, NX))}, time=0.0)


def test_from_coords_and_rebuild_sidecar(tmp_path) -> None:
    x = np.linspace(0, 1, NX, endpoint=False)
    z = np.linspace(-0.9, 0.9, NZ)
    path = tmp_path / "t.h5"
    with HDF5File.from_coords(
        path, (x, z), domains=[(0.0, 1.0), (-1.0, 1.0)], wrap_axes=(0,), wall_axes=(1,)
    ) as f:
        f.write({"u": np.ones((NX, NZ))}, time=2.0, step=5)
    (tmp_path / "t.xdmf").unlink()
    out = xdmf_from_hdf5(path, name="Rebuilt")
    root = ET.parse(out).getroot()
    assert root is not None
    assert out == tmp_path / "t.xdmf"
    assert cast(Element[str], root.find(".//Grid")).get("Name") == "Rebuilt"
    assert (
        cast(Element[str], root.find(".//Topology")).get("Dimensions")
        == f"{NZ + 2} {NX + 1}"
    )
    assert cast(Element[str], root.find(".//Grid/Grid/Time")).get("Value") == "2"
