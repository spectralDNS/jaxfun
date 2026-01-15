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

    for kind in ("uniform", "legendre", "chebyshev", "random"):
        pts = us.get_points(Nx, Ny, domain="all", kind=[kind] * 2)
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
    assert jnp.all((radii >= ri) & (radii <= ro))

    bd = ann.get_points(Nx, Ny, domain="boundary")
    assert bd.shape == (2 * (Ny - 1), 2)
    radii_bd = jnp.linalg.norm(bd, axis=1)
    # first Nx inner, next Nx outer
    assert jnp.allclose(radii_bd[: Ny - 1], ri, atol=1e-6)
    assert jnp.allclose(radii_bd[-(Ny - 1) :], ro, atol=1e-6)
