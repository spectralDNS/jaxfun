import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Helmholtz' equation in an annulus using Jaxfun with polar coordinates

    In this demo we will solve Helmholtz' equation in an annulus domain using polar coordinates and spectral accuracy. Helmholtz' equation reads

    $$
    \nabla^2 u(x, y) + \alpha u(x, y) = f(x, y),\quad x, y \in \Omega
    $$

    where $\Omega = \{x, y \, | \,  r_0 < \sqrt{x^2 + y^2} < r_1\}$ and $r_0 < r_1$ are the inner and outer radii of the annulus. The parameter $\alpha$ is a constant and $f(x, y)$ is a real function. We use homogeneous Dirichlet boundary conditions on the entire boundary.

    We start by importing necessary classes and functions
    """)
    return


@app.cell
def _():
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import sympy as sp

    from jaxfun.coordinates import get_CoordSys
    from jaxfun.galerkin import (
        Fourier,
        FunctionSpace,
        Legendre,
        TensorProductSpace,
        TestFunction,
        TrialFunction,
        inner,
    )
    from jaxfun.operators import Div, Grad
    from jaxfun.utils import Domain

    return (
        Div,
        Domain,
        Fourier,
        FunctionSpace,
        Grad,
        Legendre,
        TensorProductSpace,
        TestFunction,
        TrialFunction,
        get_CoordSys,
        inner,
        jnp,
        plt,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then create a polar coordinate system
    """)
    return


@app.cell
def _(get_CoordSys, sp):
    _r, _theta = sp.symbols("r,theta", real=True, positive=True)
    C = get_CoordSys(
        "C", sp.Lambda((_r, _theta), (_r * sp.cos(_theta), _r * sp.sin(_theta)))
    )
    return (C,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This coordinate system is further used to create a tensor product space, where the radial direction uses composite Legendre polynomials (basis functions $\psi_i=P_i-P_{i+2}$) and the angular direction uses Fourier exponentials.
    """)
    return


@app.cell
def _(C, Domain, Fourier, FunctionSpace, Legendre, TensorProductSpace, sp):
    r0, r1 = sp.S.Half, 1
    R = FunctionSpace(
        20,
        Legendre.Legendre,
        bcs={"left": {"D": 0}, "right": {"D": 0}},
        domain=Domain(r0, r1),
        name="R",
        fun_str="phi",
    )
    F = FunctionSpace(20, Fourier.Fourier, name="F", fun_str="E")
    P = TensorProductSpace((R, F), system=C, name="P")
    return (P,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To implement Helmholtz' equation we need test and trial functions for the tensor product space
    """)
    return


@app.cell
def _(P, TestFunction, TrialFunction):
    u = TrialFunction(P, name="u")
    v = TestFunction(P, name="v")
    return u, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The test and trial functions are subclasses of the Sympy [Function](https://docs.sympy.org/latest/modules/core.html#sympy.core.function.Function)
    """)
    return


@app.cell
def _(u):
    u
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we see that the trial function `u` is a function of spatial coordinates `x` and `y`, and it is a trial function of the `P` function space. In computational coordinates `u` is a tensor product function using a tensor product between the trial functions from the one-dimensional `R` and `F` spaces. `u` is evaluated in computational coordinates using the `doit()` method
    """)
    return


@app.cell
def _(u):
    u.doit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that $\phi_k(r)$ is component $k$ of the trial functions of the radial coordinate and $E_l(\theta)=\exp(\imath \underline{l} \theta)$ is component $l$ of the complex exponentials. (We defined the function names "phi" and "E" when creating the spaces.) The trial function is actually an expansion

    $$
    u(x, y) = U(r, \theta) = \sum_{k=0}^{N-3}\sum_{l=0}^{N-1} \hat{u}_{kl} \phi_{k}(r) \exp(\imath \underline{l} \theta)
    $$

    where

    $$
    \underline{l} = \begin{cases} l, \quad &\text{if} \, l < N/2 \\
    -(N-l) \quad &\text{if} \, l \ge N/2
    \end{cases}
    $$

    However, we do not print the expansion factors $\{\hat{u}_{kl}\}$, only the basis functions. The expansion factors are the unknown that we will compute in the end.

    Helmholtz' equation is now
    """)
    return


@app.cell
def _(Div, Grad, u):
    alpha = 1
    eq = Div(Grad(u)) + alpha * u
    eq
    return alpha, eq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Like the trial function we can also evaluate the equation in computational coordinates using `doit()`
    """)
    return


@app.cell
def _(eq):
    eq.doit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that we get the Helmholtz equation in polar computational coordinates, and the trial functions are expanded into tensor products. It looks a bit better if we multiply through with the radius $r^2$:
    """)
    return


@app.cell
def _(C, eq):
    r, theta = C.base_scalars()
    C.simplify((r**2 * eq).doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want to see what the Helmholtz equation looks like with a regular (non-tensor-product) function, then you can create a `ScalarFunction` and use that instead of the trial function
    """)
    return


@app.cell
def _(C):
    from jaxfun.galerkin.arguments import ScalarFunction

    h = ScalarFunction("h", C)
    h
    return (h,)


@app.cell
def _(C, Div, Grad, alpha, h):
    C.simplify((Div(Grad(h)) + alpha * h).doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the scalar function use a lowercase letter in physical space and an upper case letter in computational space

    $$
    h(x, y) = H(r, \theta).
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to solve the Helmholtz equation we need to define $f(x, y)$ and assemble the inner products

    $$
    (\nabla^2u + \alpha u, v)_{L^2(\Omega)} = (f, v)_{L^2(\Omega)}
    $$

    Since the Fourier exponentials are complex functions we need to use a complex inner product defined as

    $$
    (a, b)_{L^2(\Omega)} = \int_{\Omega} \,a \, \overline{b}\, d\Omega,
    $$

    where $\overline{b}$ is the complex conjugate of $b$.

    We can assemble both sides of Helmholtz' equation in one call. Here we actually assemble all terms in

    $$
    (\nabla^2 u + \alpha u - f, v)_{L^2(\Omega)} = 0
    $$
    """)
    return


@app.cell
def _(eq, inner, sp, v):
    A, b = inner((eq - 2) * sp.conjugate(v))
    return A, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now all matrices will be collected in `A`, whereas the right hand side vector is collected in `b`
    """)
    return


@app.cell
def _(A):
    A
    return


@app.cell
def _(b):
    b.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tensor product matrices in A are tensors with four indices, and each item is the outer product of two (1D) matrices. We can see which matrices there are by inspecting the equation in computational space
    """)
    return


@app.cell
def _(C, eq, sp, v):
    C.simplify((eq * sp.conjugate(v)).doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here the test function uses indices $i$ and $j$, whereas the trial function uses $k$ and $l$. Each term with four indices is one `TPMatrix` in `A`. As seen above, there are 4 tensor product matrices (4 items with 4 indices). Note that the complex conjugate of the test function is evaluated automatically inside the `inner` function and it is not necessary to call it explicitly as above. Hence `eq*v` would produce the same result, but it would not print as nicely (and correctly) as above. Also, when evaluating the inner product, the equation is multiplied by an additional `r`, since the domain measure $d\Omega = rdrd\theta$

    In the end we assemble a Kronecker product matrix from all the items in `A` and then solve the linear system of equations:
    """)
    return


@app.cell
def _(A, b, jnp):
    M = jnp.sum(jnp.array([jnp.kron(*a.mats) for a in A]), axis=0)
    uh = jnp.linalg.solve(M, b.flatten()).reshape(b.shape)
    return (uh,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot the solution in Cartesian coordinates on a uniform grid with 100 by 100 points:
    """)
    return


@app.cell
def _(P, plt, uh):
    xc, yc = P.cartesian_mesh(kind="uniform", N=(100, 100))
    uj = P.backward(uh, kind="uniform", N=(100, 100))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    ax.contourf(xc, yc, uj.real, 50)
    plt.axis("equal")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
