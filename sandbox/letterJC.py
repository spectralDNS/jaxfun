import numpy as np
from shapely.geometry import (  # for readable buffer styles
    CAP_STYLE,
    JOIN_STYLE,
    LineString,
    Point,
    Polygon,
)


def make_letter_J_from_centerline(
    *,
    height: float = 10.0,      # total letter height
    width: float = 6.0,        # horizontal extent of the top bar (to the right)
    thickness: float = 1.2,    # stroke thickness (final polygon width)
    hook_radius: float = 3.0,  # radius of the bottom hook (centerline arc)
    arc_resolution: int = 64,  # number of points sampling the arc
    top_bar_left: float = 0.0  # where the top bar starts on the left
) -> tuple[LineString, Polygon]:
    """
    Build a 'J' glyph as:
      1) a centerline polyline/arc (LineString),
      2) a filled polygon via buffer(thickness/2).

    Geometry:
      - Top bar centerline is at y = height - thickness/2 so the buffered
        polygon’s flat cap aligns near y = height when cap_style=FLAT.
      - Vertical stem sits at x = width (right side).
      - Bottom hook is a semicircle arc centered at (width - hook_radius, hook_radius),
        sweeping from angle 0 down to -π (clockwise), which forms the classic J-hook.

    Returns:
      (centerline: LineString, outline: Polygon)
    """
    assert hook_radius > thickness * 0.5, "hook_radius must exceed half the thickness."

    y_bar = height - thickness * 0.5
    cx, cy = width - hook_radius, hook_radius

    # Centerline segments:
    # 1) Top horizontal bar (left -> right)
    p0 = (top_bar_left, y_bar)
    p1 = (width, y_bar)

    # 2) Vertical stem (down to the hook start)
    p2 = (width, hook_radius)  # this equals the arc start when angle = 0

    # 3) Bottom hook arc: angles from 0 to -pi (clockwise)
    ang = np.linspace(0.0, -np.pi, arc_resolution)
    arc_xy = np.column_stack([cx + hook_radius * np.cos(ang),
                              cy + hook_radius * np.sin(ang)])
    # Build full centerline polyline (duplicate p2 is fine; LineString tolerates it)
    coords = [p0, p1, p2] + [tuple(p) for p in arc_xy]
    centerline = LineString(coords)

    # Buffer the centerline to get a filled J shape
    outline = centerline.buffer(
        thickness / 2.0,
        cap_style=CAP_STYLE.round,    # flat ends for bar/arc endpoints
        join_style=JOIN_STYLE.round, # rounded joins around corners/arc
        resolution=arc_resolution,   # arc smoothness
    )

    return centerline, outline


# Example usage
if __name__ == "__main__":
    cl, poly = make_letter_J_from_centerline(
        height=10.0, width=6.0, thickness=1.2, hook_radius=3.0, arc_resolution=96
    )
    print("Centerline length:", cl.length, "Polygon area:", poly.area)

    from shapely.prepared import prep

    prepJ = prep(poly)
    xmin, ymin, xmax, ymax = poly.bounds
    M = 5000
    cand = np.column_stack(
        [
            np.random.uniform(xmin, xmax, M),
            np.random.uniform(ymin, ymax, M),
        ]
    )
    mask = np.array([prepJ.contains(Point(x, y)) for x, y in cand])
    xi = cand[mask]
    print("Interior sample shape:", xi.shape)

    def sample_boundary(poly: Polygon, n: int) -> np.ndarray:
        ring = poly.exterior
        coords = np.asarray(ring.coords)
        segs = np.stack([coords[:-1], coords[1:]], axis=1)
        lengths = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
        probs = lengths / lengths.sum()
        counts = np.random.multinomial(n, probs)
        pts = []
        for (a, b), m in zip(segs, counts, strict=True):
            if m == 0:
                continue
            t = np.random.rand(m)
            pts.append(a + t[:, None] * (b - a))
        return np.vstack(pts)

    xb = sample_boundary(poly, 500)

    # Optional quick plot with matplotlib
    import matplotlib.pyplot as plt
    x, y = cl.xy
    px, py = poly.exterior.xy
    plt.figure(figsize=(5, 6))
    plt.plot(x, y, "r--", lw=2, label="centerline")
    plt.fill(px, py, alpha=0.4, label="buffered J")
    plt.axis("equal"); plt.legend(); plt.title("Letter J from centerline + buffer")
    plt.show()