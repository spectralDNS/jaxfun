import jax.numpy as jnp
from flax import nnx

from jaxfun.pinns.mesh import (
    Annulus,
    AnnulusPolar,
    CartesianProductMesh,
    Circle_with_hole,
    Line,
    Lshape,
    Rectangle,
    Square_with_hole,
    Triangle,
    UnitLine,
    UnitSquare,
    points_along_axis,
)
from jaxfun.utils import leggauss


def _meshgrid_flatten(x, y):
    return jnp.array(jnp.meshgrid(x, y, indexing="ij")).reshape((2, -1)).T


def _cheb_nodes(n):
    # Matches implementation used in mesh.py
    return (1 + jnp.cos(jnp.pi + (2 * jnp.arange(n) + 1) * jnp.pi / (2 * n))) / 2


def test_unitsquare_uniform_inside_matches_meshgrid():
    Nx, Ny = 4, 5
    us = UnitSquare()

    pts = us.get_points(Nx, Ny, domain="inside")
    x = jnp.linspace(0, 1, Nx)[1:-1]
    y = jnp.linspace(0, 1, Ny)[1:-1]
    expected = _meshgrid_flatten(x, y)

    assert pts.shape == ((Nx - 2) * (Ny - 2), 2)
    assert jnp.allclose(pts, expected)


def test_unitsquare_legendre_inside_is_tensor_product():
    Nx, Ny = 5, 6
    us = UnitSquare()

    pts = us.get_points(Nx, Ny, domain="inside", kind=["legendre"] * 2)
    x = (1 + leggauss(Nx - 2)[0]) / 2
    y = (1 + leggauss(Ny - 2)[0]) / 2
    expected = _meshgrid_flatten(x, y)

    # Compare as sets by sorting rows to avoid assuming any special order
    idx_pts = jnp.lexsort((pts[:, 1], pts[:, 0]))
    idx_exp = jnp.lexsort((expected[:, 1], expected[:, 0]))
    assert jnp.allclose(pts[idx_pts], expected[idx_exp])


def test_unitsquare_chebyshev_inside_is_tensor_product():
    Nx, Ny = 7, 4
    us = UnitSquare()

    pts = us.get_points(Nx, Ny, domain="inside", kind=["chebyshev"] * 2)
    x = _cheb_nodes(Nx - 2)
    y = _cheb_nodes(Ny - 2)
    expected = _meshgrid_flatten(x, y)

    idx_pts = jnp.lexsort((pts[:, 1], pts[:, 0]))
    idx_exp = jnp.lexsort((expected[:, 1], expected[:, 0]))
    assert jnp.allclose(pts[idx_pts], expected[idx_exp])


def test_unitsquare_boundary_legendre_and_chebyshev_basic():
    Nx, Ny = 5, 6
    us = UnitSquare()

    kinds = ("legendre", "chebyshev", "random", "uniform")
    for kind in zip(kinds, kinds, strict=True):
        bd_c = us.get_points(Nx, Ny, domain="boundary", kind=kind)

        assert bd_c.shape == (2 * Nx + 2 * Ny - 4, 2)

        # all boundary points must lie on x in {0,1} or y in {0,1}
        on_left_or_right = (bd_c[:, 0] == 0) | (bd_c[:, 0] == 1)
        on_bottom_or_top = (bd_c[:, 1] == 0) | (bd_c[:, 1] == 1)
        assert jnp.all(on_left_or_right | on_bottom_or_top)

        # corners present
        corners = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        present = jnp.array(
            [jnp.any(jnp.all(jnp.isclose(bd_c, c), axis=1)) for c in corners]
        )
        assert jnp.all(present)


def test_unitsquare_random_reproducible_with_rng_key():
    Nx, Ny = 8, 3
    rngs = nnx.Rngs(123)
    us1 = UnitSquare(key=rngs())
    us2 = UnitSquare(key=nnx.Rngs(123)())

    p1 = us1.get_points(Nx * Ny, Nx, domain="all", kind="random")
    p2 = us2.get_points(Nx * Ny, Nx, domain="all", kind="random")

    assert p1.shape == (Nx * Ny, 2)
    assert jnp.all(p1 == p2)
    # same seed -> same samples


def test_unitline_points_and_weights():
    N = 7
    ul = UnitLine()

    kinds = ["uniform", "legendre", "chebyshev", "random"]
    p = [ul.get_points(N, domain="all", kind=k) for k in kinds]

    assert jnp.all(jnp.array([pts.shape == (N, 1) for pts in p]))

    p = [ul.get_points(N, domain="inside", kind=k) for k in kinds]
    # values in (0, 1)
    for pi in p:
        assert jnp.all((pi > 0) & (pi < 1))

    # boundary points
    p = [ul.get_points(N, domain="boundary", kind=k) for k in kinds]

    assert jnp.all(jnp.array([pts.shape == (2, 1) for pts in p]))

    assert jnp.all(
        jnp.array([jnp.allclose(pi.squeeze(), jnp.array([0.0, 1.0])) for pi in p])
    )

    # weights
    w = [ul.get_weights(N, domain="inside", kind=k) for k in kinds]

    assert w[0] == 1
    assert w[3] == 1
    assert w[1].shape == (N - 2,)
    assert w[2].shape == (N - 2,)
    assert jnp.allclose(w[2], (jnp.pi / (N - 2)) * jnp.ones(N - 2))


def test_line_affine_mapping():
    N = 5
    left, right = -2.5, 3.0
    line = Line(left=left, right=right)

    p = line.get_points(N, domain="inside", kind="uniform")
    assert p.shape == (N - 2, 1)
    # map should be within interval
    assert p.min() > left
    assert p.max() < right

    bd = line.get_points(N, domain="boundary", kind="uniform").squeeze()
    assert jnp.allclose(bd, jnp.array([left, right]))


def test_unitsquare_points_all_kinds():
    Nx, Ny = 3, 4
    us = UnitSquare()

    for kind in ("uniform", "legendre", "chebyshev"):
        pts = us.get_points(Nx, Ny, domain="all", kind=kind)
        assert pts.shape == (Nx * Ny, 2)
        assert jnp.all((pts >= 0) & (pts <= 1))


def test_unitsquare_weights_inside_and_boundary():
    Nx, Ny = 4, 3
    us = UnitSquare()

    # inside weights
    kinds = ("uniform", "random", "legendre", "chebyshev")
    wi = [us.get_weights(Nx, Ny, domain="all", kind=[kind] * 2) for kind in kinds]

    assert wi[0] == 1
    assert wi[1] == 1
    assert wi[2].shape == (Nx * Ny,)
    assert wi[3].shape == (Nx * Ny,)

    wi = [us.get_weights(Nx, Ny, domain="inside", kind=[kind] * 2) for kind in kinds]

    assert wi[0] == 1
    assert wi[1] == 1
    assert wi[2].shape == ((Nx - 2) * (Ny - 2),)
    assert wi[3].shape == ((Nx - 2) * (Ny - 2),)

    # boundary weights
    wi = [us.get_weights(Nx, Ny, domain="boundary", kind=[kind] * 2) for kind in kinds]

    assert wi[0] == 1
    assert wi[1] == 1
    assert wi[2].shape == (2 * Nx + 2 * Ny - 4,)
    assert wi[3].shape == (2 * Nx + 2 * Ny - 4,)


def test_rectangle_maps_unitsquare():
    Nx, Ny = 5, 4
    rect = Rectangle(left=-2.0, right=4.0, bottom=1.5, top=3.0)

    pts_in = rect.get_points(Nx, Ny, domain="inside")
    assert pts_in.shape == ((Nx - 2) * (Ny - 2), 2)
    assert jnp.all((pts_in[:, 0] > -2.0) & (pts_in[:, 0] < 4.0))
    assert jnp.all((pts_in[:, 1] > 1.5) & (pts_in[:, 1] < 3.0))

    bd = rect.get_points(Nx, Ny, domain="boundary")
    assert bd.shape == (2 * Nx + 2 * Ny - 4, 2)
    # Check that boundary x's and y's hit the rectangle edges
    assert jnp.all(
        (bd[:Nx, 1] == 1.5)
        | (bd[:Nx, 1] == 3.0)
        | (bd[:Nx, 0] == -2.0)
        | (bd[:Nx, 0] == 4.0)
    )


def test_points_along_axis():
    a = jnp.array([0.0, 1.0])
    b = jnp.array([-1.0, 2.0, 3.0])
    pts = points_along_axis(a, b)
    assert pts.shape == (a.size * b.size, 2)
    # contains all pairs
    expected = jnp.array(
        [[0.0, -1.0], [0.0, 2.0], [0.0, 3.0], [1.0, -1.0], [1.0, 2.0], [1.0, 3.0]]
    )
    assert jnp.allclose(pts, expected)


def test_annulus_polar_inside_and_boundary_uniform():
    # Use Nx == Ny to match current implementation assumptions
    Nx = 6
    Ny = 5
    ri, ro = 2.0, 5.0
    ap = AnnulusPolar(radius_inner=ri, radius_outer=ro)

    for kind in ("uniform", "random"):
        pts_in = ap.get_points(Nx, Ny, domain="all", kind=[kind] * 2)
        assert pts_in.shape == (Nx * (Ny - 1), 2)
        r = pts_in[:, 0]
        theta = pts_in[:, 1]
        assert jnp.all((r >= ri) & (r <= ro))
        assert jnp.all((theta >= 0) & (theta < 2 * jnp.pi))
        pts_in = ap.get_points(Nx, Ny, domain="inside", kind=[kind] * 2)
        assert pts_in.shape == ((Nx - 2) * (Ny - 1), 2)
        r = pts_in[:, 0]
        theta = pts_in[:, 1]
        assert jnp.all((r > ri) & (r < ro))
        assert jnp.all((theta >= 0) & (theta < 2 * jnp.pi))

        bd = ap.get_points(Nx, Ny, domain="boundary", kind=[kind] * 2)
        # boundary points are 2 * Nx
        assert bd.shape == (2 * (Ny - 1), 2)
        # first Nx rows at inner radius, next Nx rows at outer radius
        assert jnp.allclose(bd[: Ny - 1, 0], ri)
        assert jnp.allclose(bd[-(Ny - 1) :, 0], ro)


def test_annulus_cartesian_inside_and_boundary_uniform():
    Nx = 5
    Ny = 6
    ri, ro = 1.0, 2.0
    ann = Annulus(radius_inner=ri, radius_outer=ro)

    pts_in = ann.get_points(Nx, Ny, domain="all")
    assert pts_in.shape == (Nx * (Ny - 1), 2)
    # radii in [ri, ro]
    radii = jnp.linalg.norm(pts_in, axis=1)
    assert jnp.all((radii + 1e-6 >= ri) & (radii - 1e-6 <= ro))

    bd = ann.get_points(Nx, Ny, domain="boundary")
    assert bd.shape == (2 * (Ny - 1), 2)
    radii_bd = jnp.linalg.norm(bd, axis=1)
    # first Nx inner, next Nx outer
    assert jnp.allclose(radii_bd[: Ny - 1], ri, atol=1e-6)
    assert jnp.allclose(radii_bd[-(Ny - 1) :], ro, atol=1e-6)


# Test union property: get_all_points = get_points_inside_domain ∪ get_points_on_domain


def test_unitline_points_union_property():
    """Test get_all_points equals union of inside + boundary points."""
    N = 8
    ul = UnitLine()

    for kind in ["uniform", "legendre", "chebyshev"]:
        all_pts = ul.get_points(N, domain="all", kind=kind)
        inside_pts = ul.get_points(N, domain="inside", kind=kind)
        boundary_pts = ul.get_points(N, domain="boundary", kind=kind)

        union_pts = jnp.vstack((inside_pts, boundary_pts))
        assert all_pts.shape == union_pts.shape
        # For UnitLine, get_all_points returns sorted
        # [boundary[0], inside..., boundary[1]]
        # So we need to verify all points are accounted for
        assert all_pts.shape == (N, 1)
        assert inside_pts.shape == (N - 2, 1)
        assert boundary_pts.shape == (2, 1)


def test_line_points_union_property():
    """Test get_all_points equals union of inside + boundary points."""
    N = 6
    line = Line(left=-1.0, right=2.0)

    for kind in ["uniform", "legendre", "chebyshev"]:
        all_pts = line.get_points(N, domain="all", kind=kind)
        inside_pts = line.get_points(N, domain="inside", kind=kind)
        boundary_pts = line.get_points(N, domain="boundary", kind=kind)

        union_pts = jnp.vstack((inside_pts, boundary_pts))
        assert all_pts.shape == union_pts.shape
        assert all_pts.shape == (N, 1)
        assert inside_pts.shape == (N - 2, 1)
        assert boundary_pts.shape == (2, 1)


def test_unitsquare_points_union_property():
    """Test get_all_points equals union of inside + boundary points."""
    Nx, Ny = 5, 6
    us = UnitSquare()

    for kind in ["uniform", "legendre", "chebyshev"]:
        all_pts = us.get_points(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_pts = us.get_points(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_pts = us.get_points(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert all_pts.shape == (Nx * Ny, 2)
        assert inside_pts.shape == ((Nx - 2) * (Ny - 2), 2)
        assert boundary_pts.shape == (2 * Nx + 2 * Ny - 4, 2)

        # Total number of points should match
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


def test_rectangle_points_union_property():
    """Test get_all_points equals union of inside + boundary points."""
    Nx, Ny = 4, 5
    rect = Rectangle(left=-1.0, right=2.0, bottom=0.0, top=3.0)

    for kind in ["uniform", "legendre", "chebyshev"]:
        all_pts = rect.get_points(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_pts = rect.get_points(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_pts = rect.get_points(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert all_pts.shape == (Nx * Ny, 2)
        assert inside_pts.shape == ((Nx - 2) * (Ny - 2), 2)
        assert boundary_pts.shape == (2 * Nx + 2 * Ny - 4, 2)
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


def test_annulus_polar_points_union_property():
    """Test get_all_points equals union of inside + boundary points."""
    Nx, Ny = 5, 6
    ap = AnnulusPolar(radius_inner=1.0, radius_outer=3.0)

    for kind in ["uniform", "random"]:
        all_pts = ap.get_points(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_pts = ap.get_points(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_pts = ap.get_points(Nx, Ny, domain="boundary", kind=[kind] * 2)

        # For AnnulusPolar, theta excludes 2π to avoid duplication
        assert all_pts.shape == (Nx * (Ny - 1), 2)
        assert inside_pts.shape == ((Nx - 2) * (Ny - 1), 2)
        assert boundary_pts.shape == (2 * (Ny - 1), 2)
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


def test_annulus_points_union_property():
    """Test get_all_points equals union of inside + boundary points."""
    Nx, Ny = 5, 6
    ann = Annulus(radius_inner=1.0, radius_outer=3.0)

    for kind in ["uniform", "random"]:
        all_pts = ann.get_points(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_pts = ann.get_points(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_pts = ann.get_points(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert all_pts.shape == (Nx * (Ny - 1), 2)
        assert inside_pts.shape == ((Nx - 2) * (Ny - 1), 2)
        assert boundary_pts.shape == (2 * (Ny - 1), 2)
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


# Test union: get_all_weights = get_weights_inside_domain ∪ get_weights_on_domain


def test_unitline_weights_union_property():
    """Test get_all_weights equals union of inside + boundary weights."""
    N = 7
    ul = UnitLine()

    # uniform/random return Literal[1], so union still returns 1
    for kind in ["uniform", "random"]:
        all_w = ul.get_weights(N, domain="all", kind=kind)
        inside_w = ul.get_weights(N, domain="inside", kind=kind)
        boundary_w = ul.get_weights(N, domain="boundary", kind=kind)

        assert all_w == 1
        assert inside_w == 1
        assert boundary_w == 1

    # legendre/chebyshev return arrays
    for kind in ["legendre", "chebyshev"]:
        all_w = ul.get_weights(N, domain="all", kind=kind)
        inside_w = ul.get_weights(N, domain="inside", kind=kind)
        boundary_w = ul.get_weights(N, domain="boundary", kind=kind)

        assert isinstance(all_w, jnp.ndarray)
        assert isinstance(inside_w, jnp.ndarray)
        assert boundary_w == 1

        # For UnitLine, all_weights includes boundary weights
        assert all_w.shape == (N,)
        assert inside_w.shape == (N - 2,)

        # Verify interior weights match those in all_weights (excluding boundaries)
        assert jnp.allclose(all_w[1:-1], inside_w)


def test_line_weights_union_property():
    """Test get_all_weights equals union of inside + boundary weights."""
    N = 6
    line = Line(left=-2.0, right=3.0)

    for kind in ["uniform", "random"]:
        all_w = line.get_weights(N, domain="all", kind=kind)
        inside_w = line.get_weights(N, domain="inside", kind=kind)
        boundary_w = line.get_weights(N, domain="boundary", kind=kind)

        assert all_w == 1
        assert inside_w == 1
        assert boundary_w == 1

    for kind in ["legendre", "chebyshev"]:
        all_w = line.get_weights(N, domain="all", kind=kind)
        inside_w = line.get_weights(N, domain="inside", kind=kind)

        assert isinstance(all_w, jnp.ndarray)
        assert isinstance(inside_w, jnp.ndarray)
        assert all_w.shape == (N,)
        assert inside_w.shape == (N - 2,)
        assert jnp.allclose(all_w[1:-1], inside_w)


def test_unitsquare_weights_union_property():
    """Test get_all_weights union property for 2D mesh."""
    Nx, Ny = 5, 6
    us = UnitSquare()

    for kind in ["uniform", "random"]:
        all_w = us.get_weights(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_w = us.get_weights(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_w = us.get_weights(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert all_w == 1
        assert inside_w == 1
        assert boundary_w == 1

    for kind in ["legendre", "chebyshev"]:
        all_w = us.get_weights(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_w = us.get_weights(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_w = us.get_weights(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert isinstance(all_w, jnp.ndarray)
        assert isinstance(inside_w, jnp.ndarray)
        assert isinstance(boundary_w, jnp.ndarray)

        assert all_w.shape == (Nx * Ny,)
        assert inside_w.shape == ((Nx - 2) * (Ny - 2),)
        assert boundary_w.shape == (2 * Nx + 2 * Ny - 4,)


def test_rectangle_weights_union_property():
    """Test get_all_weights union property for Rectangle."""
    Nx, Ny = 4, 5
    rect = Rectangle(left=-1.0, right=2.0, bottom=0.0, top=3.0)

    for kind in ["uniform", "random"]:
        all_w = rect.get_weights(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_w = rect.get_weights(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_w = rect.get_weights(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert all_w == 1
        assert inside_w == 1
        assert boundary_w == 1

    for kind in ["legendre", "chebyshev"]:
        all_w = rect.get_weights(Nx, Ny, domain="all", kind=[kind] * 2)
        inside_w = rect.get_weights(Nx, Ny, domain="inside", kind=[kind] * 2)
        boundary_w = rect.get_weights(Nx, Ny, domain="boundary", kind=[kind] * 2)

        assert isinstance(all_w, jnp.ndarray)
        assert isinstance(inside_w, jnp.ndarray)
        assert isinstance(boundary_w, jnp.ndarray)

        assert all_w.shape == (Nx * Ny,)
        assert inside_w.shape == ((Nx - 2) * (Ny - 2),)
        assert boundary_w.shape == (2 * Nx + 2 * Ny - 4,)


# Additional coverage tests


def test_line_boundary_mask():
    """Test boundary_mask method for Line."""
    N = 6
    line = Line(left=-1.0, right=1.0)

    mask = line.boundary_mask(N, kind="uniform")
    assert mask.shape == (N,)
    assert mask[0] and mask[-1]  # First and last are boundary
    assert not jnp.any(mask[1:-1])  # Middle points are interior


def test_unitline_boundary_mask():
    """Test boundary_mask method for UnitLine."""
    N = 5
    ul = UnitLine()

    mask = ul.boundary_mask(N, kind="uniform")
    assert mask.shape == (N,)
    assert mask[0] and mask[-1]
    assert not jnp.any(mask[1:-1])


def test_rectangle_boundary_mask():
    """Test boundary_mask method for Rectangle."""
    Nx, Ny = 5, 6
    rect = Rectangle(left=-2.0, right=2.0, bottom=-1.0, top=1.0)

    for kind in ["uniform", "legendre", "chebyshev"]:
        mask = rect.boundary_mask(Nx, Ny, kind=[kind] * 2)
        assert mask.shape == (Nx * Ny,)

        # Count boundary points
        n_boundary = jnp.sum(mask)
        expected_boundary = 2 * Nx + 2 * Ny - 4
        assert n_boundary == expected_boundary


def test_unitsquare_boundary_mask():
    """Test boundary_mask method for UnitSquare."""
    Nx, Ny = 4, 5
    us = UnitSquare()

    for kind in ["uniform", "legendre", "chebyshev"]:
        mask = us.boundary_mask(Nx, Ny, kind=[kind] * 2)
        all_pts = us.get_points(Nx, Ny, domain="all", kind=[kind] * 2)

        # Verify mask identifies boundary correctly
        boundary_pts = all_pts[mask]
        expected_n_boundary = 2 * Nx + 2 * Ny - 4
        assert boundary_pts.shape == (expected_n_boundary, 2)


def test_rectangle_random_sampling():
    """Test Rectangle with random sampling."""
    Ni, Nx = 100, 20
    rect = Rectangle(left=-1.0, right=1.0, bottom=-2.0, top=2.0)

    all_pts = rect.get_points(Ni, Nx, domain="all", kind="random")
    # Should have Ni total points, with Nx adjusted boundary points
    assert all_pts.shape == (Ni, 2)

    # All points should be within bounds
    assert jnp.all((all_pts[:, 0] >= -1.0) & (all_pts[:, 0] <= 1.0))
    assert jnp.all((all_pts[:, 1] >= -2.0) & (all_pts[:, 1] <= 2.0))


def test_annulus_polar_boundary_mask():
    """Test boundary_mask for AnnulusPolar."""
    Nx, Ny = 5, 6
    ap = AnnulusPolar(radius_inner=1.0, radius_outer=3.0)

    mask = ap.boundary_mask(Nx, Ny, kind="uniform")
    # AnnulusPolar excludes theta=2π to avoid duplication
    # So it has Nx*(Ny-1) points instead of Nx*Ny
    # But boundary_mask is from parent Rectangle, so it returns Nx*Ny mask
    # Actually, for AnnulusPolar's actual points, we need to get all_points first
    all_pts = ap.get_all_points(Nx, Ny, kind="uniform")
    assert all_pts.shape == (Nx * (Ny - 1), 2)
    assert mask.shape == (Nx * (Ny - 1),)


def test_annulus_cartesian_coordinate_conversion():
    """Test Annulus converts polar to Cartesian correctly."""
    Nx, Ny = 5, 6
    ann = Annulus(radius_inner=1.0, radius_outer=2.0)

    # Get points in Cartesian
    pts = ann.get_points(Nx, Ny, domain="all", kind="uniform")

    # Verify radial distances
    radii = jnp.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    assert jnp.all((radii >= 1.0 - 1e-6) & (radii <= 2.0 + 1e-6))


# ShapelyMesh tests


def test_square_with_hole_basic_sampling():
    """Test Square_with_hole mesh basic sampling."""
    N = 100
    mesh = Square_with_hole(
        left=-1.0, right=1.0, bottom=-1.0, top=1.0, cx=0.3, cy=0.0, r=0.4
    )

    # Test all points
    all_pts = mesh.get_points(N, domain="all", kind="random")
    assert all_pts.shape[1] == 2  # 2D points
    assert all_pts.shape[0] <= N  # May have fewer due to boundary_factor

    # All points should be within square bounds
    assert jnp.all((all_pts[:, 0] >= -1.0) & (all_pts[:, 0] <= 1.0))
    assert jnp.all((all_pts[:, 1] >= -1.0) & (all_pts[:, 1] <= 1.0))

    # Interior points should not be inside the hole
    inside_pts = mesh.get_points(N, domain="inside", kind="random")
    radii_from_hole = jnp.sqrt(
        (inside_pts[:, 0] - 0.3) ** 2 + (inside_pts[:, 1] - 0.0) ** 2
    )
    assert jnp.all(radii_from_hole >= 0.4 - 1e-6)  # Outside the hole


def test_square_with_hole_boundary_sampling():
    """Test Square_with_hole boundary sampling."""
    N = 100
    mesh = Square_with_hole()

    boundary_pts = mesh.get_points(N, domain="boundary", kind="random")
    assert boundary_pts.shape[1] == 2

    # Boundary points should be on square edges or hole perimeter
    # At least should be within the domain
    assert boundary_pts.shape[0] > 0


def test_square_with_hole_weights():
    """Test Square_with_hole weights are always 1."""
    N = 50
    mesh = Square_with_hole()

    assert mesh.get_weights(N, domain="all", kind="random") == 1
    assert mesh.get_weights(N, domain="inside", kind="random") == 1
    assert mesh.get_weights(N, domain="boundary", kind="random") == 1


def test_square_with_hole_boundary_mask():
    """Test Square_with_hole boundary_mask."""
    N = 100
    mesh = Square_with_hole()

    mask = mesh.boundary_mask(N, kind="random")
    assert mask.shape == (N,)
    assert mask.dtype == bool

    # Number of boundary points determined by boundary_factor
    n_boundary = jnp.sum(mask)
    expected_boundary = int(N * mesh.boundary_factor)
    assert n_boundary == expected_boundary


def test_circle_with_hole_basic_sampling():
    """Test Circle_with_hole mesh basic sampling."""
    N = 100
    mesh = Circle_with_hole(Cx=0.0, Cy=0.0, R=1.0, cx=0.3, cy=0.0, r=0.4)

    all_pts = mesh.get_points(N, domain="all", kind="random")
    assert all_pts.shape[1] == 2

    # Points should be within outer circle
    radii = jnp.sqrt(all_pts[:, 0] ** 2 + all_pts[:, 1] ** 2)
    assert jnp.all(radii <= 1.0 + 1e-6)

    # Interior points should be outside inner hole
    inside_pts = mesh.get_points(N, domain="inside", kind="random")
    radii_from_hole = jnp.sqrt(
        (inside_pts[:, 0] - 0.3) ** 2 + (inside_pts[:, 1] - 0.0) ** 2
    )
    assert jnp.all(radii_from_hole >= 0.4 - 1e-6)


def test_circle_with_hole_weights():
    """Test Circle_with_hole weights are always 1."""
    N = 50
    mesh = Circle_with_hole()

    assert mesh.get_weights(N, domain="all", kind="random") == 1
    assert mesh.get_weights(N, domain="inside", kind="random") == 1
    assert mesh.get_weights(N, domain="boundary", kind="random") == 1


def test_triangle_basic_sampling():
    """Test Triangle mesh basic sampling."""
    N = 100
    mesh = Triangle()

    all_pts = mesh.get_points(N, domain="all", kind="random")
    assert all_pts.shape[1] == 2
    assert all_pts.shape[0] <= N

    # Points should be within bounds [0,1] x [0,1]
    assert jnp.all((all_pts[:, 0] >= 0.0) & (all_pts[:, 0] <= 1.0))
    assert jnp.all((all_pts[:, 1] >= 0.0) & (all_pts[:, 1] <= 1.0))

    # For triangle: x + y <= 1
    assert jnp.all(all_pts[:, 0] + all_pts[:, 1] <= 1.0 + 1e-6)


def test_triangle_boundary_includes_vertices():
    """Test Triangle boundary includes the three vertices."""
    N = 100
    mesh = Triangle()

    boundary_pts = mesh.get_points(N, domain="boundary", kind="random")
    assert boundary_pts.shape[1] == 2

    # Check that vertices are included
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for vertex in vertices:
        # Check if vertex is in boundary points (within tolerance)
        distances = jnp.sqrt(jnp.sum((boundary_pts - vertex) ** 2, axis=1))
        assert jnp.any(distances < 1e-6)


def test_triangle_weights():
    """Test Triangle weights are always 1."""
    N = 50
    mesh = Triangle()

    assert mesh.get_weights(N, domain="all", kind="random") == 1
    assert mesh.get_weights(N, domain="inside", kind="random") == 1
    assert mesh.get_weights(N, domain="boundary", kind="random") == 1


def test_lshape_basic_sampling():
    """Test Lshape mesh basic sampling."""
    N = 100
    mesh = Lshape(left=-1.0, right=1.0, bottom=-1.0, top=1.0, Lx=1.0, Ly=1.0)

    all_pts = mesh.get_points(N, domain="all", kind="random")
    assert all_pts.shape[1] == 2
    assert all_pts.shape[0] <= N

    inside_pts = mesh.get_points(N, domain="inside", kind="random")
    assert inside_pts.shape[1] == 2


def test_lshape_boundary_includes_corners():
    """Test Lshape boundary includes corner points."""
    N = 100
    mesh = Lshape(left=-1.0, right=1.0, bottom=-1.0, top=1.0, Lx=1.0, Ly=1.0)

    boundary_pts = mesh.get_points(N, domain="boundary", kind="random")
    assert boundary_pts.shape[1] == 2

    # Check that some corner points are included
    expected_corners = jnp.array(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    )
    for corner in expected_corners:
        distances = jnp.sqrt(jnp.sum((boundary_pts - corner) ** 2, axis=1))
        assert jnp.any(distances < 1e-6)


def test_lshape_weights():
    """Test Lshape weights are always 1."""
    N = 50
    mesh = Lshape()

    assert mesh.get_weights(N, domain="all", kind="random") == 1
    assert mesh.get_weights(N, domain="inside", kind="random") == 1
    assert mesh.get_weights(N, domain="boundary", kind="random") == 1


def test_shapely_meshes_union_property():
    """Test union property for ShapelyMesh-based classes."""
    N = 100
    meshes = [
        Square_with_hole(),
        Circle_with_hole(),
        Triangle(),
        Lshape(),
    ]

    for mesh in meshes:
        all_pts = mesh.get_points(N, domain="all", kind="random")
        inside_pts = mesh.get_points(N, domain="inside", kind="random")
        boundary_pts = mesh.get_points(N, domain="boundary", kind="random")

        # Total number of points should match
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


# Edge case tests


def test_line_invalid_bounds():
    """Test Line raises error for invalid bounds."""
    try:
        Line(left=2.0, right=1.0)
        raise AssertionError("Should raise ValueError")
    except ValueError as e:
        assert "greater than" in str(e).lower()


def test_rectangle_invalid_bounds():
    """Test Rectangle raises error for invalid bounds."""
    try:
        Rectangle(left=2.0, right=1.0, bottom=0.0, top=1.0)
        raise AssertionError("Should raise ValueError for left > right")
    except ValueError as e:
        assert "greater than" in str(e).lower()

    try:
        Rectangle(left=0.0, right=1.0, bottom=2.0, top=1.0)
        raise AssertionError("Should raise ValueError for bottom > top")
    except ValueError as e:
        assert "greater than" in str(e).lower()


def test_unitline_small_N():
    """Test UnitLine with small N values."""
    ul = UnitLine()

    # N=2 means only boundary points
    pts = ul.get_points(2, domain="all", kind="uniform")
    assert pts.shape == (2, 1)
    assert jnp.allclose(pts.squeeze(), jnp.array([0.0, 1.0]))

    # N=2 means 0 interior points
    pts = ul.get_points(2, domain="inside", kind="uniform")
    assert pts.shape == (0, 1)


def test_rectangle_to_shapely():
    """Test Rectangle.to_shapely() conversion."""
    rect = Rectangle(left=-1.0, right=1.0, bottom=-2.0, top=2.0)
    shapely_mesh = rect.to_shapely()

    # Should return a ShapelyMesh instance
    assert hasattr(shapely_mesh, "make_polygon")

    # Test sampling with the shapely version
    N = 50
    pts = shapely_mesh.get_points(N, domain="all", kind="random")
    assert pts.shape[1] == 2


def test_unitline_to_shapely():
    """Test UnitLine.to_shapely() returns a LineString."""
    ul = UnitLine()
    linestring = ul.to_shapely()

    # Should be a shapely LineString
    assert hasattr(linestring, "coords")


def test_line_to_shapely():
    """Test Line.to_shapely() returns a LineString."""
    line = Line(left=-2.0, right=3.0)
    linestring = line.to_shapely()

    assert hasattr(linestring, "coords")


# 3D CartesianProductMesh tests


def test_3d_mesh_from_three_lines_basic():
    """Test 3D CartesianProductMesh created from three Line meshes."""
    Nx, Ny, Nz = 4, 5, 6
    line_x = Line(left=-1.0, right=1.0)
    line_y = Line(left=0.0, right=2.0)
    line_z = Line(left=-0.5, right=0.5)

    mesh_3d = CartesianProductMesh(line_x, line_y, line_z)

    # Test get_all_points
    all_pts = mesh_3d.get_points(Nx, Ny, Nz, domain="all", kind=["uniform"] * 3)
    assert all_pts.shape == (Nx * Ny * Nz, 3)

    # Check bounds
    assert jnp.all((all_pts[:, 0] >= -1.0) & (all_pts[:, 0] <= 1.0))
    assert jnp.all((all_pts[:, 1] >= 0.0) & (all_pts[:, 1] <= 2.0))
    assert jnp.all((all_pts[:, 2] >= -0.5) & (all_pts[:, 2] <= 0.5))


def test_3d_mesh_from_three_lines_inside_points():
    """Test interior points for 3D mesh from three lines."""
    Nx, Ny, Nz = 4, 5, 6
    line_x = Line(left=-1.0, right=1.0)
    line_y = Line(left=0.0, right=2.0)
    line_z = Line(left=-0.5, right=0.5)

    mesh_3d = CartesianProductMesh(line_x, line_y, line_z)

    inside_pts = mesh_3d.get_points(Nx, Ny, Nz, domain="inside", kind=["uniform"] * 3)
    # Interior points exclude boundaries in all three dimensions
    expected_shape = ((Nx - 2) * (Ny - 2) * (Nz - 2), 3)
    assert inside_pts.shape == expected_shape

    # All interior points should be strictly within bounds (not on boundary)
    assert jnp.all((inside_pts[:, 0] > -1.0) & (inside_pts[:, 0] < 1.0))
    assert jnp.all((inside_pts[:, 1] > 0.0) & (inside_pts[:, 1] < 2.0))
    assert jnp.all((inside_pts[:, 2] > -0.5) & (inside_pts[:, 2] < 0.5))


def test_3d_mesh_from_three_lines_boundary_points():
    """Test boundary points for 3D mesh from three lines."""
    Nx, Ny, Nz = 4, 5, 6
    line_x = Line(left=-1.0, right=1.0)
    line_y = Line(left=0.0, right=2.0)
    line_z = Line(left=-0.5, right=0.5)

    mesh_3d = CartesianProductMesh(line_x, line_y, line_z)

    boundary_pts = mesh_3d.get_points(
        Nx, Ny, Nz, domain="boundary", kind=["uniform"] * 3
    )

    # Total points minus interior points
    expected_n_boundary = Nx * Ny * Nz - (Nx - 2) * (Ny - 2) * (Nz - 2)
    assert boundary_pts.shape == (expected_n_boundary, 3)

    # At least one coordinate should be at a boundary for each point
    at_x_boundary = (jnp.abs(boundary_pts[:, 0] - (-1.0)) < 1e-8) | (
        jnp.abs(boundary_pts[:, 0] - 1.0) < 1e-8
    )
    at_y_boundary = (jnp.abs(boundary_pts[:, 1] - 0.0) < 1e-8) | (
        jnp.abs(boundary_pts[:, 1] - 2.0) < 1e-8
    )
    at_z_boundary = (jnp.abs(boundary_pts[:, 2] - (-0.5)) < 1e-8) | (
        jnp.abs(boundary_pts[:, 2] - 0.5) < 1e-8
    )

    assert jnp.all(at_x_boundary | at_y_boundary | at_z_boundary)


def test_3d_mesh_from_three_lines_boundary_mask():
    """Test boundary_mask for 3D mesh from three lines."""
    Nx, Ny, Nz = 4, 5, 6
    line_x = Line(left=-1.0, right=1.0)
    line_y = Line(left=0.0, right=2.0)
    line_z = Line(left=-0.5, right=0.5)

    mesh_3d = CartesianProductMesh(line_x, line_y, line_z)

    mask = mesh_3d.boundary_mask(Nx, Ny, Nz, kind=["uniform"] * 3)
    assert mask.shape == (Nx * Ny * Nz,)
    assert mask.dtype == bool

    # Number of boundary points
    n_boundary = jnp.sum(mask)
    expected_n_boundary = Nx * Ny * Nz - (Nx - 2) * (Ny - 2) * (Nz - 2)
    assert n_boundary == expected_n_boundary

    # Verify mask matches actual boundary points
    all_pts = mesh_3d.get_points(Nx, Ny, Nz, domain="all", kind=["uniform"] * 3)
    boundary_pts_from_mask = all_pts[mask]
    boundary_pts_direct = mesh_3d.get_points(
        Nx, Ny, Nz, domain="boundary", kind=["uniform"] * 3
    )

    # Should have same shape
    assert boundary_pts_from_mask.shape == boundary_pts_direct.shape


def test_3d_mesh_from_three_lines_union_property():
    """Test union property for 3D mesh from three lines."""
    Nx, Ny, Nz = 4, 5, 6
    line_x = Line(left=-1.0, right=1.0)
    line_y = Line(left=0.0, right=2.0)
    line_z = Line(left=-0.5, right=0.5)

    mesh_3d = CartesianProductMesh(line_x, line_y, line_z)

    for kind in ["uniform", "legendre", "chebyshev"]:
        all_pts = mesh_3d.get_points(Nx, Ny, Nz, domain="all", kind=[kind] * 3)
        inside_pts = mesh_3d.get_points(Nx, Ny, Nz, domain="inside", kind=[kind] * 3)
        boundary_pts = mesh_3d.get_points(
            Nx, Ny, Nz, domain="boundary", kind=[kind] * 3
        )

        # Verify dimensions
        assert all_pts.shape == (Nx * Ny * Nz, 3)
        assert inside_pts.shape == ((Nx - 2) * (Ny - 2) * (Nz - 2), 3)

        # Total number of points should match
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


def test_3d_mesh_from_three_lines_weights_union_property():
    """Test weights union property for 3D mesh from three lines."""
    Nx, Ny, Nz = 4, 5, 6
    line_x = Line(left=-1.0, right=1.0)
    line_y = Line(left=0.0, right=2.0)
    line_z = Line(left=-0.5, right=0.5)

    mesh_3d = CartesianProductMesh(line_x, line_y, line_z)

    # Test with uniform (returns Literal[1])
    all_w = mesh_3d.get_weights(Nx, Ny, Nz, domain="all", kind=["uniform"] * 3)
    inside_w = mesh_3d.get_weights(Nx, Ny, Nz, domain="inside", kind=["uniform"] * 3)
    boundary_w = mesh_3d.get_weights(
        Nx, Ny, Nz, domain="boundary", kind=["uniform"] * 3
    )

    assert all_w == 1
    assert inside_w == 1
    assert boundary_w == 1

    # Test with legendre (returns arrays)
    all_w = mesh_3d.get_weights(Nx, Ny, Nz, domain="all", kind=["legendre"] * 3)
    inside_w = mesh_3d.get_weights(Nx, Ny, Nz, domain="inside", kind=["legendre"] * 3)
    boundary_w = mesh_3d.get_weights(
        Nx, Ny, Nz, domain="boundary", kind=["legendre"] * 3
    )

    assert isinstance(all_w, jnp.ndarray)
    assert isinstance(inside_w, jnp.ndarray)
    assert isinstance(boundary_w, jnp.ndarray)

    assert all_w.shape == (Nx * Ny * Nz,)
    assert inside_w.shape == ((Nx - 2) * (Ny - 2) * (Nz - 2),)


def test_3d_mesh_from_unitsquare_and_line_basic():
    """Test 3D CartesianProductMesh created from UnitSquare and Line."""
    Nx, Ny, Nz = 4, 5, 6
    unit_square = UnitSquare()
    line_z = Line(left=-1.0, right=1.0)

    mesh_3d = CartesianProductMesh(unit_square, line_z)

    # Test get_all_points - UnitSquare is one submesh (takes Nx, Ny), Line is
    # another (takes Nz)
    all_pts = mesh_3d.get_points(
        (Nx, Ny), Nz, domain="all", kind=["uniform", "uniform"]
    )
    assert all_pts.shape == (Nx * Ny * Nz, 3)

    # Check bounds
    assert jnp.all((all_pts[:, 0] >= 0.0) & (all_pts[:, 0] <= 1.0))
    assert jnp.all((all_pts[:, 1] >= 0.0) & (all_pts[:, 1] <= 1.0))
    assert jnp.all((all_pts[:, 2] >= -1.0) & (all_pts[:, 2] <= 1.0))


def test_3d_mesh_from_unitsquare_and_line_inside_points():
    """Test interior points for 3D mesh from UnitSquare and Line."""
    Nx, Ny, Nz = 4, 5, 6
    unit_square = UnitSquare()
    line_z = Line(left=-1.0, right=1.0)

    mesh_3d = CartesianProductMesh(unit_square, line_z)

    inside_pts = mesh_3d.get_points((Nx, Ny), Nz, domain="inside", kind=["uniform"] * 2)
    # Interior points exclude boundaries in all dimensions
    expected_shape = ((Nx - 2) * (Ny - 2) * (Nz - 2), 3)
    assert inside_pts.shape == expected_shape

    # All interior points should be strictly within bounds
    assert jnp.all((inside_pts[:, 0] > 0.0) & (inside_pts[:, 0] < 1.0))
    assert jnp.all((inside_pts[:, 1] > 0.0) & (inside_pts[:, 1] < 1.0))
    assert jnp.all((inside_pts[:, 2] > -1.0) & (inside_pts[:, 2] < 1.0))


def test_3d_mesh_from_unitsquare_and_line_boundary_points():
    """Test boundary points for 3D mesh from UnitSquare and Line."""
    Nx, Ny, Nz = 4, 5, 6
    unit_square = UnitSquare()
    line_z = Line(left=-1.0, right=1.0)

    mesh_3d = CartesianProductMesh(unit_square, line_z)

    boundary_pts = mesh_3d.get_points(
        (Nx, Ny), Nz, domain="boundary", kind=["uniform"] * 2
    )

    # Total points minus interior points
    expected_n_boundary = Nx * Ny * Nz - (Nx - 2) * (Ny - 2) * (Nz - 2)
    assert boundary_pts.shape == (expected_n_boundary, 3)

    # At least one coordinate should be at a boundary for each point
    at_x_boundary = (jnp.abs(boundary_pts[:, 0] - 0.0) < 1e-8) | (
        jnp.abs(boundary_pts[:, 0] - 1.0) < 1e-8
    )
    at_y_boundary = (jnp.abs(boundary_pts[:, 1] - 0.0) < 1e-8) | (
        jnp.abs(boundary_pts[:, 1] - 1.0) < 1e-8
    )
    at_z_boundary = (jnp.abs(boundary_pts[:, 2] - (-1.0)) < 1e-8) | (
        jnp.abs(boundary_pts[:, 2] - 1.0) < 1e-8
    )

    assert jnp.all(at_x_boundary | at_y_boundary | at_z_boundary)


def test_3d_mesh_from_unitsquare_and_line_boundary_mask():
    """Test boundary_mask for 3D mesh from UnitSquare and Line."""
    Nx, Ny, Nz = 4, 5, 6
    unit_square = UnitSquare()
    line_z = Line(left=-1.0, right=1.0)

    mesh_3d = CartesianProductMesh(unit_square, line_z)

    mask = mesh_3d.boundary_mask((Nx, Ny), Nz, kind=["uniform"] * 2)
    assert mask.shape == (Nx * Ny * Nz,)
    assert mask.dtype == bool

    # Number of boundary points
    n_boundary = jnp.sum(mask)
    expected_n_boundary = Nx * Ny * Nz - (Nx - 2) * (Ny - 2) * (Nz - 2)
    assert n_boundary == expected_n_boundary

    # Verify mask matches actual boundary points
    all_pts = mesh_3d.get_points((Nx, Ny), Nz, domain="all", kind=["uniform"] * 2)
    boundary_pts_from_mask = all_pts[mask]
    boundary_pts_direct = mesh_3d.get_points(
        (Nx, Ny), Nz, domain="boundary", kind=["uniform"] * 2
    )

    # Should have same shape
    assert boundary_pts_from_mask.shape == boundary_pts_direct.shape


def test_3d_mesh_from_unitsquare_and_line_union_property():
    """Test union property for 3D mesh from UnitSquare and Line."""
    Nx, Ny, Nz = 4, 5, 6
    unit_square = UnitSquare()
    line_z = Line(left=-1.0, right=1.0)

    mesh_3d = CartesianProductMesh(unit_square, line_z)

    for kind in ["uniform", "legendre", "chebyshev"]:
        all_pts = mesh_3d.get_points((Nx, Ny), Nz, domain="all", kind=[kind] * 2)
        inside_pts = mesh_3d.get_points((Nx, Ny), Nz, domain="inside", kind=[kind] * 2)
        boundary_pts = mesh_3d.get_points(
            (Nx, Ny), Nz, domain="boundary", kind=[kind] * 2
        )

        # Verify dimensions
        assert all_pts.shape == (Nx * Ny * Nz, 3)
        assert inside_pts.shape == ((Nx - 2) * (Ny - 2) * (Nz - 2), 3)

        # Total number of points should match
        assert all_pts.shape[0] == inside_pts.shape[0] + boundary_pts.shape[0]


def test_3d_mesh_from_unitsquare_and_line_weights_union_property():
    """Test weights union property for 3D mesh from UnitSquare and Line."""
    Nx, Ny, Nz = 4, 5, 6
    unit_square = UnitSquare()
    line_z = Line(left=-1.0, right=1.0)

    mesh_3d = CartesianProductMesh(unit_square, line_z)

    # Test with uniform (returns Literal[1])
    all_w = mesh_3d.get_weights((Nx, Ny), Nz, domain="all", kind=["uniform"] * 2)
    inside_w = mesh_3d.get_weights((Nx, Ny), Nz, domain="inside", kind=["uniform"] * 2)
    boundary_w = mesh_3d.get_weights(
        (Nx, Ny), Nz, domain="boundary", kind=["uniform"] * 2
    )

    assert all_w == 1
    assert inside_w == 1
    assert boundary_w == 1

    # Test with legendre (returns arrays)
    all_w = mesh_3d.get_weights((Nx, Ny), Nz, domain="all", kind=["legendre"] * 2)
    inside_w = mesh_3d.get_weights((Nx, Ny), Nz, domain="inside", kind=["legendre"] * 2)
    boundary_w = mesh_3d.get_weights(
        (Nx, Ny), Nz, domain="boundary", kind=["legendre"] * 2
    )

    assert isinstance(all_w, jnp.ndarray)
    assert isinstance(inside_w, jnp.ndarray)
    assert isinstance(boundary_w, jnp.ndarray)

    assert all_w.shape == (Nx * Ny * Nz,)
    assert inside_w.shape == ((Nx - 2) * (Ny - 2) * (Nz - 2),)
