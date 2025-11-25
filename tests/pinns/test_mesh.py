import jax.numpy as jnp
from flax import nnx

from jaxfun.pinns.mesh import (
    Annulus,
    AnnulusPolar,
    Line,
    Rectangle,
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

    pts = us.get_points_inside_domain(Nx, Ny, "uniform")
    x = jnp.linspace(0, 1, Nx + 2)[1:-1]
    y = jnp.linspace(0, 1, Ny + 2)[1:-1]
    expected = _meshgrid_flatten(x, y)

    assert pts.shape == (Nx * Ny, 2)
    assert jnp.allclose(pts, expected)


def test_unitsquare_legendre_inside_is_tensor_product():
    Nx, Ny = 3, 6
    us = UnitSquare()

    pts = us.get_points_inside_domain(Nx, Ny, "legendre")
    x = (1 + leggauss(Nx)[0]) / 2
    y = (1 + leggauss(Ny)[0]) / 2
    expected = _meshgrid_flatten(x, y)

    # Compare as sets by sorting rows to avoid assuming any special order
    idx_pts = jnp.lexsort((pts[:, 1], pts[:, 0]))
    idx_exp = jnp.lexsort((expected[:, 1], expected[:, 0]))
    assert jnp.allclose(pts[idx_pts], expected[idx_exp])


def test_unitsquare_chebyshev_inside_is_tensor_product():
    Nx, Ny = 7, 4
    us = UnitSquare()

    pts = us.get_points_inside_domain(Nx, Ny, "chebyshev")
    x = _cheb_nodes(Nx)
    y = _cheb_nodes(Ny)
    expected = _meshgrid_flatten(x, y)

    idx_pts = jnp.lexsort((pts[:, 1], pts[:, 0]))
    idx_exp = jnp.lexsort((expected[:, 1], expected[:, 0]))
    assert jnp.allclose(pts[idx_pts], expected[idx_exp])


def test_unitsquare_boundary_legendre_and_chebyshev_basic():
    Nx, Ny = 5, 3
    us = UnitSquare()

    for kind in ("legendre", "chebyshev", "random", "uniform"):
        bd_c = us.get_points_on_domain(Nx, Ny, kind, corners=True)
        bd_nc = us.get_points_on_domain(Nx, Ny, kind, corners=False)

        # shapes: 2*Nx + 2*Ny (+4 corners)
        assert bd_nc.shape == (2 * Nx + 2 * Ny, 2)
        assert bd_c.shape == (2 * Nx + 2 * Ny + 4, 2)

        # all boundary points must lie on x in {0,1} or y in {0,1}
        on_left_or_right = (bd_c[:, 0] == 0) | (bd_c[:, 0] == 1)
        on_bottom_or_top = (bd_c[:, 1] == 0) | (bd_c[:, 1] == 1)
        assert jnp.all(on_left_or_right | on_bottom_or_top)

        # corners present when requested
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

    p1 = us1.get_points_inside_domain(Nx, Ny, "random")
    p2 = us2.get_points_inside_domain(Nx, Ny, "random")

    assert p1.shape == (Nx * Ny, 2)
    assert jnp.all(p1 == p2)
    # same seed -> same samples


def test_unitline_points_and_weights():
    N = 7
    ul = UnitLine()

    # inside points
    pu = ul.get_points_inside_domain(N, "uniform")
    pl = ul.get_points_inside_domain(N, "legendre")
    pc = ul.get_points_inside_domain(N, "chebyshev")
    pr = ul.get_points_inside_domain(N, "random")

    assert pu.shape == (N, 1)
    assert pl.shape == (N, 1)
    assert pc.shape == (N, 1)
    assert pr.shape == (N, 1)

    # values in (0, 1)
    for p in (pu, pl, pc, pr):
        assert jnp.all((p > 0) & (p < 1))

    # boundary points
    bd = ul.get_points_on_domain("uniform")
    assert bd.shape == (2, 1)
    assert jnp.allclose(bd.squeeze(), jnp.array([0.0, 1.0]))

    # weights
    w_u = ul.get_weights_inside_domain(N, "uniform")
    w_r = ul.get_weights_inside_domain(N, "random")
    w_l = ul.get_weights_inside_domain(N, "legendre")
    w_c = ul.get_weights_inside_domain(N, "chebyshev")

    assert w_u == 1
    assert w_r == 1
    assert w_l.shape == (N,)
    assert w_c.shape == (N,)
    assert jnp.allclose(w_c, (jnp.pi / N) * jnp.ones(N))

    assert ul.get_weights_on_domain(N, "uniform") == 1


def test_line_affine_mapping():
    N = 5
    left, right = -2.5, 3.0
    line = Line(left=left, right=right)

    p = line.get_points_inside_domain(N, "uniform")
    assert p.shape == (N, 1)
    # map should be within interval
    assert p.min() > left
    assert p.max() < right

    bd = line.get_points_on_domain(N, "uniform").squeeze()
    assert jnp.allclose(bd, jnp.array([left, right]))


def test_unitsquare_inside_points_all_kinds():
    Nx, Ny = 3, 4
    us = UnitSquare()

    for kind in ("uniform", "legendre", "chebyshev", "random"):
        pts = us.get_points_inside_domain(Nx, Ny, kind)
        assert pts.shape == (Nx * Ny, 2)
        assert jnp.all((pts >= 0) & (pts <= 1))


def test_unitsquare_boundary_points_and_corners():
    Nx, Ny = 3, 2
    us = UnitSquare()

    # with corners
    bd = us.get_points_on_domain(Nx, Ny, "uniform", corners=True)
    assert bd.shape == (2 * Nx + 2 * Ny + 4, 2)

    # first Nx: y=0, next Nx: y=1, next Ny: x=0, last Ny: x=1
    assert jnp.allclose(bd[:Nx, 1], 0.0)
    assert jnp.allclose(bd[Nx : 2 * Nx, 1], 1.0)
    assert jnp.allclose(bd[2 * Nx : 2 * Nx + Ny, 0], 0.0)
    assert jnp.allclose(bd[2 * Nx + Ny : 2 * Nx + 2 * Ny, 0], 1.0)

    # last 4 are corners
    corners = bd[-4:]
    expected_c = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert jnp.allclose(jnp.sort(corners, axis=0), jnp.sort(expected_c, axis=0))

    # without corners
    bd_nc = us.get_points_on_domain(Nx, Ny, "uniform", corners=False)
    assert bd_nc.shape == (2 * Nx + 2 * Ny, 2)


def test_unitsquare_weights_inside_and_boundary():
    Nx, Ny = 4, 3
    us = UnitSquare()

    # inside weights
    wi_u = us.get_weights_inside_domain(Nx, Ny, "uniform")
    wi_r = us.get_weights_inside_domain(Nx, Ny, "random")
    wi_l = us.get_weights_inside_domain(Nx, Ny, "legendre")
    wi_c = us.get_weights_inside_domain(Nx, Ny, "chebyshev")

    assert wi_u == 1
    assert wi_r == 1
    assert wi_l.shape == (Nx * Ny,)
    assert wi_c.shape == (Nx * Ny,)

    # boundary weights
    for kind in ("uniform", "legendre", "chebyshev", "random"):
        for corners in (False, True):
            wb_l = us.get_weights_on_domain(Nx, Ny, kind, corners=corners)
            base_len = 2 * Nx + 2 * Ny
            if kind in ("uniform", "random"):
                assert wb_l == 1
            else:
                assert wb_l.shape == (base_len + (4 if corners else 0),)
                if corners:
                    assert jnp.allclose(wb_l[-4:], 1.0)


def test_rectangle_maps_unitsquare():
    Nx, Ny = 2, 3
    rect = Rectangle(left=-2.0, right=4.0, bottom=1.5, top=3.0)

    pts_in = rect.get_points_inside_domain(Nx, Ny, "uniform")
    assert pts_in.shape == (Nx * Ny, 2)
    assert jnp.all((pts_in[:, 0] > -2.0) & (pts_in[:, 0] < 4.0))
    assert jnp.all((pts_in[:, 1] > 1.5) & (pts_in[:, 1] < 3.0))

    bd = rect.get_points_on_domain(Nx, Ny, "uniform", corners=True)
    assert bd.shape == (2 * Nx + 2 * Ny + 4, 2)
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
    Nx = Ny = 4
    ri, ro = 2.0, 5.0
    ap = AnnulusPolar(radius_inner=ri, radius_outer=ro)

    for kind in ("uniform", "random"):
        pts_in = ap.get_points_inside_domain(Nx, Ny, kind)
        assert pts_in.shape == (Nx * Ny, 2)
        r = pts_in[:, 0]
        theta = pts_in[:, 1]
        assert jnp.all((r > ri) & (r < ro))
        assert jnp.all((theta >= 0) & (theta < 2 * jnp.pi))
        pts_in = ap.get_points_inside_domain(Nx, Ny, kind)
        assert pts_in.shape == (Nx * Ny, 2)
        r = pts_in[:, 0]
        theta = pts_in[:, 1]
        assert jnp.all((r > ri) & (r < ro))
        assert jnp.all((theta >= 0) & (theta < 2 * jnp.pi))

        bd = ap.get_points_on_domain(Nx, Ny, kind, corners=False)
        # boundary points are 2 * Ny
        assert bd.shape == (2 * Ny, 2)
        # first Nx rows at inner radius, next Nx rows at outer radius
        assert jnp.allclose(bd[:Nx, 0], ri)
        assert jnp.allclose(bd[Nx : 2 * Nx, 0], ro)


def test_annulus_cartesian_inside_and_boundary_uniform():
    Nx = Ny = 3
    ri, ro = 1.0, 2.0
    ann = Annulus(radius_inner=ri, radius_outer=ro)

    pts_in = ann.get_points_inside_domain(Nx, Ny, "uniform")
    assert pts_in.shape == (Nx * Ny, 2)
    # radii in [ri, ro]
    radii = jnp.linalg.norm(pts_in, axis=1)
    assert jnp.all((radii > ri) & (radii < ro))

    bd = ann.get_points_on_domain(Nx, Ny, "uniform", corners=False)
    assert bd.shape == (2 * Ny, 2)
    radii_bd = jnp.linalg.norm(bd, axis=1)
    # first Nx inner, next Nx outer
    assert jnp.allclose(radii_bd[:Nx], ri, atol=1e-6)
    assert jnp.allclose(radii_bd[Nx : 2 * Nx], ro, atol=1e-6)
