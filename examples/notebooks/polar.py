import marimo

__generated_with = "0.19.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polar coordinates with jaxfun

    Just some tests to illustrate curvilinear coordinates.
    """)
    return


@app.cell
def _():
    import numpy as np
    import sympy as sp

    from jaxfun.coordinates import get_CoordSys
    from jaxfun.galerkin import (
        FunctionSpace,
        Legendre,
        TensorProductSpace,
        TestFunction,
        TrialFunction,
    )
    from jaxfun.operators import Cross, Curl, Div, Dot, Grad
    from jaxfun.utils.common import Domain

    system = "cylindrical"
    _r, _theta, _z = sp.symbols("r,theta,z", real=True, positive=True)
    if system == "polar":
        C = get_CoordSys(
            "C", sp.Lambda((_r, _theta), (_r * sp.cos(_theta), _r * sp.sin(_theta)))
        )
        R = FunctionSpace(
            20,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=Domain(0.5, 1),
            name="R",
            fun_str="phi",
        )
        T = FunctionSpace(
            20,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=Domain(0, np.pi),
            name="T",
            fun_str="psi",
        )
        P = TensorProductSpace((R, T), system=C, name="P")
    elif system == "cylindrical":
        # system = 'polar'
        C = get_CoordSys(
            "C",
            sp.Lambda((_r, _theta, _z), (_r * sp.cos(_theta), _r * sp.sin(_theta), _z)),
        )
        R = FunctionSpace(
            20,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=Domain(0.5, 1),
            name="R",
            fun_str="phi",
        )
        T = FunctionSpace(
            20,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=Domain(0, np.pi),
            name="T",
            fun_str="psi",
        )
        Z = FunctionSpace(
            20, Legendre.Legendre, domain=Domain(0, 1), name="Z", fun_str="L"
        )
        P = TensorProductSpace((R, T, Z), system=C, name="P")
    return C, Cross, Curl, Div, Dot, Grad, P, TestFunction, TrialFunction, sp


@app.cell
def _(Div, Grad, P, TrialFunction):
    f = TrialFunction(P, name="f")
    du = Div(Grad(f))
    du
    return du, f


@app.cell
def _(du):
    du.doit()
    return


@app.cell
def _(du):
    from sympy import srepr

    srepr(du)
    return (srepr,)


@app.cell
def _(C, Div, Grad, f, sp):
    dv = sp.Add.fromiter(C._parent.base_scalars()) * Div(Grad(f))
    dv
    return (dv,)


@app.cell
def _(dv, srepr):
    srepr(dv)
    return


@app.cell
def _(dv):
    dv.doit()
    return


@app.cell
def _(Grad, f):
    Grad(f).doit()
    return


@app.cell
def _(C):
    from jaxfun.galerkin.arguments import ScalarFunction

    g = ScalarFunction("g", C)
    g
    return (g,)


@app.cell
def _(g):
    G = g.doit()
    G
    return


@app.cell
def _(Div, Grad, g):
    Div(Grad(g))
    return


@app.cell
def _(Div, Grad, g):
    Div(Grad(g)).doit()
    return


@app.cell
def _(P):
    P.tensorname
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test operators on vectors
    """)
    return


@app.cell
def _(P):
    from jaxfun.galerkin import VectorTensorProductSpace

    V = VectorTensorProductSpace(P, name="V")
    V.tensorname
    return (V,)


@app.cell
def _(V):
    V.name
    return


@app.cell
def _(TestFunction, V):
    v = TestFunction(V, name="v")
    v
    return (v,)


@app.cell
def _(v):
    v.doit()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Gradient of vector:
    """)
    return


@app.cell
def _(Grad, u):
    Grad(u)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the gradient of a vector $\boldsymbol{u}$ is written as $\nabla \boldsymbol{u}$ since $\nabla$ is interpreted as an operator and not a vector. If $\nabla$ had instead been interpreted as a vector, then the gradient of $\boldsymbol{u}$ would be written as $(\nabla \boldsymbol{u})^T$.
    """)
    return


@app.cell
def _(Grad, u):
    Grad(u).doit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Dot product:
    """)
    return


@app.cell
def _(Dot, v):
    Dot(v, v).doit()
    return


@app.cell
def _(C, Cross, v):
    from IPython.display import display

    if C.dims == 3:
        y = Cross(v, v).doit()
        display(y)
    return (display,)


@app.cell
def _(C, Cross, display, v):
    if C.dims == 3:
        z = Cross(v, C.b_r)
        display(z)
        display(z.doit())
    return


@app.cell
def _(C, Curl, display, v):
    if C.dims == 3:
        d = Curl(v)
        F = C.to_cartesian(d.doit())
        display(d)
        display(d.doit())
        display(F)
    return


@app.cell
def _(C):
    from jaxfun.galerkin.arguments import VectorFunction

    h = VectorFunction("h", C)
    h
    return (h,)


@app.cell
def _(h):
    H = h.doit()
    H
    return


@app.cell
def _(Dot, TrialFunction, V, srepr, v):
    w = TrialFunction(V, name="u")
    vw = Dot(w, v)
    srepr(vw)
    return (vw,)


@app.cell
def _(vw):
    VW = vw.doit()
    VW
    return (VW,)


@app.cell
def _(VW):
    VW.args
    return


@app.cell
def _(C, VW, sp):
    a = sp.separatevars(VW.args[1], dict=True, symbols=C._base_scalars)
    return (a,)


@app.cell
def _(a):
    a
    return


@app.cell
def _(C, a):
    import contextlib

    for _r in C._base_scalars:
        for j in a[_r].args:
            with contextlib.suppress(AttributeError):
                print(j.local_index)
    return


@app.cell
def _(C):
    from flax import nnx

    from jaxfun.operators import Identity, Outer
    from jaxfun.pinns import FlaxFunction, MLPSpace

    W = MLPSpace([8], dims=C.dims, rank=1, system=C, name="V")
    Q = MLPSpace([8], dims=C.dims, rank=0, system=C, name="Q")
    u = FlaxFunction(W, "u", rngs=nnx.Rngs(2002))  # Vector space for velocity
    p = FlaxFunction(Q, "p", rngs=nnx.Rngs(1001))  # Scalar space for pressure
    return Identity, Outer, p, u


@app.cell
def _(C, Div, u):
    C.simplify(Div(u).doit())
    return


@app.cell
def _(Div, Dot, Grad, p, u):
    R1 = Dot(Grad(u), u) - Div(Grad(u)) + Grad(p)
    R1
    return (R1,)


@app.cell
def _(C, Div, Grad, Identity, Outer, p, sp, u):
    I = Identity(C)
    R2 = Div(Outer(u, u)) - Div(
        Grad(u) + Grad(u).T - sp.Rational(2, 3) * Div(u) * I + p * I
    )
    R2.doit()
    return


@app.cell
def _(C, Cross, Curl, Div, Dot, Grad, display, p, sp, u):
    if C.dims == 3:
        R3 = Cross(Curl(u), u) + sp.S.Half * Grad(Dot(u, u)) - Div(Grad(u)) + Grad(p)
        display(R3)
        display(R3.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the divergence of a dyadic $A$ is defined as $\text{div}(A) = \nabla \cdot A$, since we choose to interpret $\nabla$ as an operator and not a vector. This operator acts over the last axis of $A$. Some authors define $\nabla$ as a vector and in that case the definition of the divergence is $\nabla \cdot A^T$. This is perfectly fine, just a difference in notation. With index notation in Cartesian coordinates the divergence may also be written as $\frac{\partial A_{ij}}{\partial x_j}$ or $\frac{\partial A_{ij}}{\partial x_j} \boldsymbol{i}_i$, where $\{\boldsymbol{i}_j\}_{j=1}^3$ are the 3 Cartesian unit vectors.

    Hence we write Div(Grad(u)) as $\nabla \cdot \nabla \boldsymbol{u}$.
    """)
    return


@app.cell
def _(C, Dot, sp, u):
    ut = Dot(u, C.get_contravariant_basis_vector(1)).doit()
    sp.srepr(ut)
    return


@app.cell
def _(Outer, u):
    uu = Outer(u, u).doit()
    return (uu,)


@app.cell
def _(uu):
    uu
    return


@app.cell
def _(C, Dot, R1):
    C.simplify(Dot(R1, C.get_contravariant_basis_vector(0)).doit())
    return


@app.cell
def _(C, R1):
    C.simplify(R1.doit())
    return


@app.cell
def _(C, Dot, R1):
    r1 = Dot(R1, C.get_contravariant_basis_vector(0)).doit()
    r1
    return (r1,)


@app.cell
def _(r1):
    r1.expand()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
