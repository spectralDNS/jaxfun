import matplotlib.pyplot as plt
from jaxfun.coordinates import get_CoordSys
import sympy as sp
import numpy as np
from jaxfun.utils import lambdify


def plot():
    u, v = sp.symbols("u, v", real=True, positive=True)
    P = get_CoordSys("P", sp.Lambda((u, v), (u+v, v)))
    u, v = P.base_scalars()
    rv = P.position_vector(False)
    assert isinstance(rv, sp.Tuple)

    b = P.get_covariant_basis()
    bt = P.get_contravariant_basis()

    r0 = 2.0
    t0 = 1.0
    pos = np.array([lambdify((u, v), rv[0])(r0, t0), lambdify((u, v), rv[1])(r0, t0)])

    b_num = np.array([[lambdify((u, v), b[i, j])(r0, t0) for j in range(2)] for i in range(2)], dtype=float)
    bt_num = np.array([[lambdify((u, v), bt[i, j])(r0, t0) for j in range(2)] for i in range(2)], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_title("Gridlines and basis vectors")

    u_grid = np.linspace(0.0, 2.0, 7)
    v_grid = np.linspace(0.0, 2.0, 7)
    v_line = np.linspace(0.0, 2.0, 100)
    u_line = np.linspace(0.0, 2.0, 100)

    for u0 in u_grid:
        x_line = lambdify((u, v), rv[0])(u0, v_line)
        y_line = lambdify((u, v), rv[1])(u0, v_line)
        ax.plot(x_line, y_line, color="0.85", linewidth=1)

    for v0 in v_grid:
        x_line = np.atleast_1d(lambdify((u, v), rv[0])(u_line, v0))
        y_line = np.atleast_1d(lambdify((u, v), rv[1])(u_line, v0))
        if y_line.shape[0] == 1:
            y_line = np.full_like(x_line, y_line.item())
        ax.plot(x_line, y_line, color="0.85", linewidth=1)

    ax.quiver(*pos, *b_num[0], color="tab:blue", angles="xy", scale_units="xy", scale=1, label=r"$\mathbf{b}_u$")
    ax.quiver(*pos, *b_num[1], color="tab:blue", angles="xy", scale_units="xy", scale=1, alpha=0.7, label=r"$\mathbf{b}_v$")
    ax.quiver(*pos, *bt_num[0], color="tab:orange", angles="xy", scale_units="xy", scale=1, label=r"$\mathbf{b}^u$")
    ax.quiver(*pos, *bt_num[1], color="tab:orange", angles="xy", scale_units="xy", scale=1, alpha=0.7, label=r"$\mathbf{b}^v$")

    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, 2.2)
    ax.legend(loc="upper left")
    plt.show()

plot()
