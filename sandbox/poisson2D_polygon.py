# ruff: noqa: E402
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep

from jaxfun import Div, Grad
from jaxfun.pinns import LSQR, FlaxFunction, MLPSpace, Trainer
from jaxfun.pinns.mesh import Square_with_hole
from jaxfun.pinns.optimizer import adam, lbfgs

V = MLPSpace([12, 12, 12], dims=2, rank=0, name="V")
w = FlaxFunction(V, name="w")

mesh = Square_with_hole()
xi = mesh.get_points_inside_domain(6000)
xb = mesh.get_points_on_domain(400, corners=True)

x, y = V.system.base_scalars()

f = Div(Grad(w)) - 1

loss_fn = LSQR((f, xi, 0), (w, xb, 0))
trainer = Trainer(loss_fn)

t0 = time.time()

opt_adam = adam(w, learning_rate=1e-3)
trainer.train(opt_adam, 1000, epoch_print=100)
print("Time for Adam:", time.time() - t0)

t1 = time.time()
opt_lbfgs = lbfgs(w, memory_size=20)
trainer.train(opt_lbfgs, 1000, epoch_print=100)
print("Time for LBFGS:", time.time() - t1)

print("time", time.time() - t0)

X_all = jnp.vstack((xi, xb))
w_all = w(X_all)


def plot_solution_all(mesh, X, values, xb=None, levels=30):
    """
    mesh   : mesh object (for polygon)
    X      : all sample points (N, 2) = vstack((xi, xb))
    values : solution values at X (N,)
    xb     : optional boundary points to overlay as red dots
    """
    poly = mesh.make_polygon()
    pts = np.asarray(X)
    vals = np.asarray(values).reshape(-1)

    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])

    # Mask triangles whose centroid lies outside the polygon (handles holes)
    prepared = prep(poly)
    centroids = pts[tri.triangles].mean(axis=1)
    mask = np.array([not prepared.contains(Point(c[0], c[1])) for c in centroids])
    tri.set_mask(mask)

    fig, ax = plt.subplots(figsize=(6, 6))
    tpc = ax.tripcolor(tri, vals, shading="gouraud", cmap="viridis")
    ax.tricontour(tri, vals, levels=levels, colors="k", linewidths=0.5)
    if xb is not None:
        xb_np = np.asarray(xb)
        ax.plot(xb_np[:, 0], xb_np[:, 1], "r.", ms=2, label="boundary")
        ax.legend(loc="lower left")
    ax.set_aspect("equal")
    ax.set_title("Solution w(x,y)")
    fig.colorbar(tpc, ax=ax, shrink=0.8, label="w")
    plt.show()


plot_solution_all(mesh, X_all, w_all, xb=xb, levels=30)
