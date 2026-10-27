r"""
Curvilinear coordinate systems, and the hints SymPy needs to simplify them.

jaxfun derives the metric of a coordinate system from its map to Cartesian
coordinates, and SymPy simplifies the result. Simplification is not an exact
science: a square root needs to know the sign of its argument, and identities
that mix circular and hyperbolic functions are often not found. A coordinate
system can therefore be given two kinds of hints:

* ``assumptions``, used where a square root is taken, that is, for the volume
  element sqrt(g) and the scaling factors h_i. For example
  ``sp.Q.positive(sp.sin(theta))`` for spherical coordinates.
* ``replace``, (pattern, replacement) pairs applied to the covariant metric
  tensor g_ij = b_i . b_j before it is simplified. For example the Pythagorean
  identity, written in the direction that lets the entries collapse.

Each system below is given the hints it needs to reach its textbook metric
tensor and volume element. Running the script prints them; under pytest it
checks them.
"""

import os
import sys
import time
from dataclasses import dataclass, field

import sympy as sp

from jaxfun.coordinates import CoordSys, get_CoordSys

cos, sin, cosh, sinh = sp.cos, sp.sin, sp.cosh, sp.sinh
pos = {"real": True, "positive": True}
r, theta, phi, z = sp.symbols("r theta phi z", **pos)
mu, tau, sigma = sp.symbols("mu tau sigma", **pos)
nu = sp.Symbol("nu", **pos)
nu_o = sp.Symbol("nu", real=True)  # oblate spheroidal nu lies in (-pi/2, pi/2)
xi, eta = sp.symbols("xi eta", real=True)
x = sp.Symbol("x", real=True)

# The Pythagorean identities, written towards cos and cosh or towards sin and sinh
towards_cos = [
    (sin(sigma) ** 2, 1 - cos(sigma) ** 2),
    (sinh(tau) ** 2, cosh(tau) ** 2 - 1),
]
towards_sin = [(cos(nu) ** 2, 1 - sin(nu) ** 2), (cosh(mu) ** 2, 1 + sinh(mu) ** 2)]

R = 2  # major radius of the tori
D = cosh(tau) - cos(sigma)  # denominator of bipolar and toroidal coordinates


@dataclass
class System:
    name: str
    coordinates: tuple[sp.Symbol, ...]
    position: tuple[sp.Expr, ...]
    g: sp.Matrix  # textbook covariant metric tensor
    sg: sp.Expr  # textbook volume element sqrt(det(g_ij))
    hints: dict = field(default_factory=dict)

    def create(self) -> CoordSys:
        mapping = sp.Lambda(self.coordinates, self.position)
        return get_CoordSys(self.name.replace(" ", "_"), mapping, **self.hints)


systems = [
    System("polar", (r, theta), (r * cos(theta), r * sin(theta)), sp.diag(1, r**2), r),
    System(
        "cylindrical",
        (r, theta, z),
        (r * cos(theta), r * sin(theta), z),
        sp.diag(1, r**2, 1),
        r,
    ),
    System(
        "spherical",
        (r, theta, phi),
        (r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)),
        sp.diag(1, r**2, r**2 * sin(theta) ** 2),
        r**2 * sin(theta),
        {"assumptions": sp.Q.positive(sin(theta))},
    ),
    System(
        "parabolic",
        (tau, sigma),
        (tau * sigma, (tau**2 - sigma**2) / 2),
        sp.diag(tau**2 + sigma**2, tau**2 + sigma**2),
        tau**2 + sigma**2,
    ),
    System(
        "elliptic",
        (mu, nu),
        (cosh(mu) * cos(nu), sinh(mu) * sin(nu)),
        sp.diag(sinh(mu) ** 2 + sin(nu) ** 2, sinh(mu) ** 2 + sin(nu) ** 2),
        sinh(mu) ** 2 + sin(nu) ** 2,
        {"replace": towards_sin},
    ),
    System(
        "bipolar",
        (tau, sigma),
        (sinh(tau) / D, sin(sigma) / D),
        sp.diag(1 / D**2, 1 / D**2),
        1 / D**2,
        {"replace": towards_cos},
    ),
    System(
        "prolate spheroidal",
        (mu, nu, phi),
        (
            sinh(mu) * sin(nu) * cos(phi),
            sinh(mu) * sin(nu) * sin(phi),
            cosh(mu) * cos(nu),
        ),
        sp.diag(
            sinh(mu) ** 2 + sin(nu) ** 2,
            sinh(mu) ** 2 + sin(nu) ** 2,
            sinh(mu) ** 2 * sin(nu) ** 2,
        ),
        (sinh(mu) ** 2 + sin(nu) ** 2) * sinh(mu) * sin(nu),
        {"assumptions": sp.Q.positive(sin(nu))},
    ),
    System(
        "oblate spheroidal",
        (mu, nu_o, phi),
        (
            cosh(mu) * cos(nu_o) * cos(phi),
            cosh(mu) * cos(nu_o) * sin(phi),
            sinh(mu) * sin(nu_o),
        ),
        sp.diag(
            sinh(mu) ** 2 + sin(nu_o) ** 2,
            sinh(mu) ** 2 + sin(nu_o) ** 2,
            cosh(mu) ** 2 * cos(nu_o) ** 2,
        ),
        (sinh(mu) ** 2 + sin(nu_o) ** 2) * cosh(mu) * cos(nu_o),
        {"assumptions": sp.Q.positive(cos(nu_o))},
    ),
    System(
        "toroidal",
        (tau, sigma, phi),
        (sinh(tau) * cos(phi) / D, sinh(tau) * sin(phi) / D, sin(sigma) / D),
        sp.diag(1 / D**2, 1 / D**2, sinh(tau) ** 2 / D**2),
        sinh(tau) / D**3,
        {"assumptions": sp.Q.positive(D), "replace": towards_cos},
    ),
    System(
        "torus",
        (r, theta, phi),
        (
            (R + r * cos(theta)) * cos(phi),
            (R + r * cos(theta)) * sin(phi),
            r * sin(theta),
        ),
        sp.diag(1, r**2, (R + r * cos(theta)) ** 2),
        r * (R + r * cos(theta)),
        {"assumptions": sp.Q.positive(R + r * cos(theta))},
    ),
    System(
        "sphere surface",
        (theta, phi),
        (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)),
        sp.diag(1, sin(theta) ** 2),
        sin(theta),
        {"assumptions": sp.Q.positive(sin(theta))},
    ),
    System(
        "torus surface",
        (theta, phi),
        ((R + cos(theta)) * cos(phi), (R + cos(theta)) * sin(phi), sin(theta)),
        sp.diag(1, (R + cos(theta)) ** 2),
        R + cos(theta),
        {"assumptions": sp.Q.positive(R + cos(theta))},
    ),
    # A one-dimensional map that clusters points around x = 0 of the periodic
    # domain [0, 2 pi). Its derivative is smooth only once simplified.
    System(
        "clustering",
        (x,),
        (2 * sp.atan(sp.tan(x / 2) / 9),),
        sp.Matrix([[81 / (41 + 40 * cos(x)) ** 2]]),
        9 / (41 + 40 * cos(x)),
        {"assumptions": sp.Q.positive(41 + 40 * cos(x))},
    ),
    # A non-orthogonal system: the bilinear map of [-1, 1]^2 onto the
    # quadrilateral with corners (0, 0), (4, 0), (3, 2) and (0, 3).
    System(
        "bilinear quadrilateral",
        (xi, eta),
        ((1 + xi) * (7 - eta) / 4, (1 + eta) * (5 - xi) / 4),
        sp.Matrix(
            [
                [(eta**2 - 6 * eta + 25) / 8, (xi * eta - 2 * eta - 3 * xi - 6) / 8],
                [(xi * eta - 2 * eta - 3 * xi - 6) / 8, (xi**2 - 4 * xi + 13) / 8],
            ]
        ),
        (17 - 4 * xi - 3 * eta) / 8,
        {
            "assumptions": sp.Q.positive(1 + xi)
            & sp.Q.positive(1 - xi)
            & sp.Q.positive(1 + eta)
            & sp.Q.positive(1 - eta)
        },
    ),
]


def as_textbook(computed: sp.Expr, textbook: sp.Expr) -> bool:
    """Whether `computed` equals `textbook` and is no longer than it.

    One operation is allowed for a sign moved between factors, as in
    -sinh(tau)/(cos(sigma) - cosh(tau))**3 for sinh(tau)/(cosh(tau) - cos(sigma))**3.
    """
    equal = sp.cancel(sp.together(computed - textbook)) == 0
    return equal and sp.count_ops(computed) <= sp.count_ops(textbook) + 1


results = []
for s in systems:
    t0 = time.perf_counter()
    C = s.create()
    G = C.get_covariant_metric_tensor()
    sg = C.sg
    elapsed = time.perf_counter() - t0
    # Compare in the plain coordinate symbols of the definition
    g = sp.Matrix([[C.expr_base_scalar_to_psi(e) for e in row] for row in G])
    sg = C.expr_base_scalar_to_psi(sg)
    n = C.dims
    ok_g = all(as_textbook(g[i, j], s.g[i, j]) for i in range(n) for j in range(n))
    ok_sg = as_textbook(sg, s.sg)
    results.append((s, g, sg, ok_g and ok_sg, elapsed))

if "PYTEST" in os.environ:
    failed = [s.name for s, *_, ok, _ in results if not ok]
    assert not failed, f"not simplified to the textbook metric: {failed}"
    sys.exit(0)

for s, g, sg, ok, elapsed in results:
    hints = ", ".join(s.hints) or "none"
    entries = list(g.diagonal()) if g.is_diagonal() else g.tolist()
    print(f"{s.name} ({elapsed:.2f} s, hints: {hints})")
    print(f"    g_ij {'(diagonal)' if g.is_diagonal() else ''}: {entries}")
    print(f"    sqrt(g): {sg}")
    print(f"    textbook form: {'yes' if ok else 'NO'}")
