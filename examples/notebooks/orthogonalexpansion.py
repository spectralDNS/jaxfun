import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Orthogonal expansions

    An `OrthogonalExpansion` is a function

    \begin{equation}
    u(x) = \sum_{k=0}^{N-1} \hat{u}_k \psi_k(x) \tag{1}
    \end{equation}

    where $\psi_k$ is an orthogonal basis (Chebyshev, Legendre, Jacobi, Ultraspherical or Fourier) and $\hat{u}_k$ are known coefficients. It works much like the series classes of `numpy.polynomial`: you add, subtract, multiply, differentiate and evaluate series, and the result is again a series in the same basis. The key word is *exact*. The product of two series of lengths $M$ and $N$ is a series of length $M+N-1$ that holds the product without any truncation or aliasing, and $x u(x)$ is a series of length $N+1$.

    The `OrthogonalExpansion` is mainly a convenience class for such exact manipulations. It is *not* a part of the form language. To solve PDEs or ODEs through inner products you still use test, trial and `JAXFunction`s, as described in the notebook on the form language. This notebook explains the difference between the two, and how to obtain an `OrthogonalExpansion` from the solution of an ODE.
    """)
    return


@app.cell
def _():
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp

    from jaxfun.galerkin import (
        Chebyshev,
        FunctionSpace,
        JAXFunction,
        Legendre,
        OrthogonalExpansion,
        TestFunction,
        TrialFunction,
        inner,
    )
    from jaxfun.operators import Div, Grad

    return (
        Chebyshev,
        Div,
        FunctionSpace,
        Grad,
        JAXFunction,
        Legendre,
        OrthogonalExpansion,
        TestFunction,
        TrialFunction,
        inner,
        jax,
        jnp,
        np,
        plt,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating expansions

    An expansion is created from its coefficients and an orthogonal function space. The space is resized to the number of coefficients, so the size of the space you pass in does not matter. Below are $a(x) = 1 + 2T_1(x) + 3T_2(x)$ and $b(x) = T_1(x) = x$, where $T_k$ is the Chebyshev polynomial of degree $k$:
    """)
    return


@app.cell
def _(Chebyshev, OrthogonalExpansion, jnp):
    T = Chebyshev.Chebyshev(3)
    x = T.system.x
    a = OrthogonalExpansion(jnp.array([1.0, 2.0, 3.0]), T)
    b = OrthogonalExpansion(jnp.array([0.0, 1.0]), T)
    print("N of a and b:", a.N, b.N)
    print("a(x) at x = -1, 0, 1:", a(jnp.array([-1.0, 0.0, 1.0])))
    return T, a, b, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact arithmetic

    Sums, differences and products of expansions, numbers and polynomials in the coordinate $x$ are all computed exactly, and the number of basis functions grows as needed. We compare with `numpy.polynomial.chebyshev`, which does the same for Chebyshev series:
    """)
    return


@app.cell
def _(a, b, np):
    ab = a * b
    print("a*b   ", ab.coef, "N =", ab.N)
    print("numpy ", np.polynomial.chebyshev.chebmul([1, 2, 3], [0, 1]))
    a3 = a**3
    print("a**3  ", a3.coef, "N =", a3.N)
    print("numpy ", np.polynomial.chebyshev.chebpow([1, 2, 3], 3))
    print("2*a - b + 1", (2 * a - b + 1).coef)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Multiplication with a polynomial in $x$ uses the three-term recurrence of the basis, and raises the length by the degree of the polynomial. Since $b(x) = x$, the two products below are the same series. Differentiation shrinks the length by one per derivative:
    """)
    return


@app.cell
def _(a, b, np, x):
    print("x*a   ", (x * a).coef)
    print("b*a   ", (b * a).coef)
    print("(x**2 - 1)*a", ((x**2 - 1) * a).coef)
    print("a'    ", a.deriv().coef)
    print("numpy ", np.polynomial.chebyshev.chebder([1, 2, 3]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expansions from expressions

    `OrthogonalExpansion.from_expr` projects a SymPy expression onto a basis. Given a function space, the expansion has the size of that space. Given the *class* of a basis, like `Chebyshev.Chebyshev`, the size is chosen to resolve the expression. A polynomial gets exactly its degree + 1 coefficients, and other functions are projected onto 16, 32, 64, ... basis functions until the trailing coefficients are at roundoff level. These are then trimmed away. Keyword arguments, like `domain` or `alpha`, go to the basis class:
    """)
    return


@app.cell
def _(Chebyshev, Legendre, OrthogonalExpansion, T, sp, x):
    from jaxfun import Domain

    print(
        "exp(x) on 3 basis functions:", OrthogonalExpansion.from_expr(sp.exp(x), T).coef
    )
    e = OrthogonalExpansion.from_expr(sp.exp(x), Chebyshev.Chebyshev)
    print("exp(x), resolved: N =", e.N)
    p3 = OrthogonalExpansion.from_expr(x**3 - x, Legendre.Legendre, domain=Domain(0, 1))
    print("x**3 - x in Legendre on [0, 1]: N =", p3.N, p3.coef)
    return (e,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The coefficients of $e^x$ fall off fast, and the expansion stops when they reach roundoff:
    """)
    return


@app.cell
def _(e, jnp, plt):
    plt.figure(figsize=(5, 3))
    plt.semilogy(jnp.abs(e.coef), "o")
    plt.xlabel("k")
    plt.ylabel(r"$|\hat{u}_k|$")
    plt.title("Chebyshev coefficients of $u(x) = e^x$")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An expansion is a JAX pytree, with the coefficients as its only leaf. It can therefore pass through `jax.jit`, `jax.grad` and `jax.vmap`, like any array:
    """)
    return


@app.cell
def _(a, jax, jnp):
    X = jnp.linspace(-1, 1, 5)
    square = jax.jit(lambda u: (u * u)(X))
    print(square(a) - a(X) ** 2)
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `jax.grad` differentiates with respect to the coefficients, through all the operations of the expansion. As an example, take the sum of squares of $a(x) = \sum_{k} \hat{a}_k T_k(x)$ at the five points $x_j$ in `X` as a function of the coefficients $\hat{a}_k$

    $$
    F(\hat{a}) = \sum_{j} a(x_j)^2,
    $$

    with gradient

    $$
    \frac{\partial F}{\partial \hat{a}_k} = 2 \sum_{j} a(x_j) T_k(x_j).
    $$

    Below, `jax.grad` computes this gradient through the product `u**2` and the evaluation of `u`, and we compare with the formula, using the matrix $P_{jk} = T_k(x_j)$ of basis functions evaluated at the points:
    """)
    return


@app.cell
def _(OrthogonalExpansion, X, a, jax):
    def F(c):
        u = OrthogonalExpansion(c, a.space)
        return (u**2)(X).sum()

    print("jax.grad:", jax.grad(F)(a.coef))
    P = a.space.evaluate_basis_derivative(X)
    print("formula: ", 2 * a(X) @ P)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## JAXFunction or OrthogonalExpansion?

    Both a `JAXFunction` and an `OrthogonalExpansion` hold a function as coefficients times basis functions. They are, however, made for different jobs.

    A `JAXFunction` is a part of the form language. It is a SymPy function that lives in a fixed function space, possibly with boundary conditions, and it can appear in the weak forms given to `inner`. There it is expanded into its coefficients and basis functions. Its space must keep its size, since the matrices and vectors assembled by `inner` are sized by the spaces. Therefore:

    * `u + w` and `u - w` of two `JAXFunction`s are computed on the coefficients and return a new `JAXFunction`. With inhomogeneous boundary conditions the result is moved to the orthogonal space of `u`, since the boundary values add up too.
    * Products like `u * w`, `u**2` or `x * u` stay symbolic SymPy expressions. `inner` integrates them by numerical quadrature on the points of the test space, and that quadrature is only exact if there are enough points.

    An `OrthogonalExpansion` is not a SymPy expression and cannot appear in `inner`. It has no boundary conditions, only a basis, and it grows as needed to keep every operation exact. It is the tool for post-processing, for example computing exact products, derivatives and residuals of a solution, and for building known functions that are later given to a solver as a `JAXFunction`.

    | | `JAXFunction` | `OrthogonalExpansion` |
    |---|---|---|
    | Used in `inner` | yes | no |
    | Function space | any, with boundary conditions | 1D orthogonal basis |
    | `u + w`, `c*u` | same size | exact, padded to the longer |
    | `u * w`, `x * u` | symbolic, integrated by quadrature | exact, grows |
    | Dimensions | 1D, 2D, 3D, vectors | 1D |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From the solution of an ODE

    Consider the ODE

    \begin{equation}
    u''(x) - 4u(x) = f(x), \quad x \in (-1, 1), \quad u(-1) = 0, \, u(1) = 1, \tag{2}
    \end{equation}

    with $f$ chosen such that the exact solution is $u(x) = \sin(\pi x) + (1+x)/2$. We solve it with a Chebyshev Galerkin method, using the sparse `"GR"` test space, as described in the other notebooks:
    """)
    return


@app.cell
def _(
    Chebyshev,
    Div,
    FunctionSpace,
    Grad,
    JAXFunction,
    TestFunction,
    TrialFunction,
    inner,
    sp,
    x,
):
    N = 20
    V = FunctionSpace(
        N, Chebyshev.Chebyshev, bcs={"left": {"D": 0}, "right": {"D": 1}}, name="V"
    )
    v = TestFunction(V.get_testspace("GR"), name="v")
    u = TrialFunction(V, name="u")
    ue = sp.sin(sp.pi * x) + (1 + x) / 2
    f = ue.diff(x, 2) - 4 * ue
    A, rhs = inner(v * (Div(Grad(u)) - 4 * u - f), kind="system", sparse=True)
    w = JAXFunction(A.solve(rhs), V, name="w")
    print("Number of unknowns:", w.array.shape)
    return N, V, f, ue, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The solution `w` is a `JAXFunction` in the space $V$. Since the boundary conditions are inhomogeneous, `FunctionSpace` returns $V$ as a direct sum

    \begin{equation}
    V = V_0 \oplus V_b, \tag{3}
    \end{equation}

    where $V_0$ is the space of Chebyshev polynomials with homogeneous boundary conditions, $u(\pm 1) = 0$, and $V_b$ is a space of lifting functions that take on the boundary values. A function in $V$ is the sum $u = u_0 + u_b$ of $u_0 \in V_0$ and $u_b \in V_b$. Only the $N-2$ coefficients of $u_0$ are unknowns, whereas the coefficients of $u_b$ are fixed by the boundary values. The direct sum shows in the name of the space, and so in the printed `w`:
    """)
    return


@app.cell
def _(V, w):
    V0, Vb = V.basespaces
    print("V =", V.name, " dim(V_0) =", V0.dim, " dim(V_b) =", Vb.dim)
    print("Boundary values:", V.bcs)
    w
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The method `to_expansion` (or `OrthogonalExpansion.from_jaxfunction(w)`) returns the solution as an `OrthogonalExpansion` in the orthogonal Chebyshev basis, with $u_0$ and the lifting $u_b$ added together. It has all $N$ Chebyshev coefficients, and it satisfies the boundary conditions:
    """)
    return


@app.cell
def _(jnp, w):
    p = w.to_expansion()
    print("Basis:", type(p.space).__name__, " N =", p.N)
    print("p(-1), p(1):", p(jnp.array([-1.0, 1.0])))
    return (p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the expansion we can compute the residual of Eq. (2) exactly in coefficient space. We expand $f$ in the same basis and form $p'' - 4p - f$. The residual is not zero, because the solution is only an approximation with $N$ basis functions, but it is small:
    """)
    return


@app.cell
def _(Chebyshev, OrthogonalExpansion, f, jnp, p):
    fe = OrthogonalExpansion.from_expr(f, Chebyshev.Chebyshev)
    residual = p.deriv(2) - 4 * p - fe
    print("Length of f:", fe.N, " residual:", residual.N)
    print("Max |residual coefficient|:", jnp.abs(residual.coef).max())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exact products and aliasing

    Nonlinear terms are where the two classes differ the most. The product `w**2` of the `JAXFunction` is a symbolic expression, still of the function in $V = V_0 \oplus V_b$:
    """)
    return


@app.cell
def _(w):
    w**2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    whereas `p**2` is a new expansion of length $2N-1$ with the exact Chebyshev coefficients of $u^2$:
    """)
    return


@app.cell
def _(p):
    q = p**2
    print("N of p**2:", q.N)
    return (q,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    An expansion can be turned back into a `JAXFunction` with `to_jaxfunction`, for any space with the same orthogonal basis that is large enough to hold it. The exact square is turned into a `JAXFunction` in a Chebyshev space of $2N-1$ basis functions, and is then used as a known function in an inner product. We compare with the inner product of the symbolic `w**2`. By default `inner` uses the $N$ quadrature points of the test space. That is not enough to integrate $u^2 v$ exactly, which is a polynomial of degree $3N-3$. The aliasing error is small here, since $u$ is well resolved. With the 3/2-rule (`num_quad_points=3N/2`) the quadrature is exact, and the two agree to roundoff:
    """)
    return


@app.cell
def _(Chebyshev, N, TestFunction, inner, jnp, q, w):
    O = Chebyshev.Chebyshev(N)
    phi = TestFunction(O)
    q2 = q.to_jaxfunction(O.resized(q.N))
    exact = inner(phi * q2)
    print("Default quadrature: ", jnp.abs(inner(phi * w**2) - exact).max())
    print(
        "3/2-rule quadrature:",
        jnp.abs(inner(phi * w**2, num_quad_points=3 * N // 2) - exact).max(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `to_jaxfunction` refuses an expansion that does not fit in the target space, unless `truncate=True` is given. It also assumes that the expansion satisfies the boundary conditions of the target space. Here the solution itself goes back into $V$:
    """)
    return


@app.cell
def _(V, jnp, p, w):
    u2 = p.to_jaxfunction(V)
    print("Same coefficients as w:", jnp.allclose(u2.array, w.array))
    try:
        (p * p).to_jaxfunction(V)
    except ValueError as err:
        print("ValueError:", err)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we plot the solution and its square, both evaluated from their expansions, together with the exact solution:
    """)
    return


@app.cell
def _(jnp, p, plt, q, ue, x):
    from jaxfun.utils.common import lambdify

    xj = jnp.linspace(-1, 1, 200)
    plt.figure(figsize=(5, 3))
    plt.plot(xj, p(xj), "b", label="$u$")
    plt.plot(xj, q(xj), "g", label="$u^2$")
    plt.plot(xj[::10], lambdify(x, ue)(xj[::10]), "ro", label="exact $u$")
    plt.legend()
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    * Solve PDEs and ODEs with test, trial and `JAXFunction`s in `inner`. Products of `JAXFunction`s are integrated by quadrature, so use enough quadrature points for nonlinear terms.
    * Use `w.to_expansion()` to get a 1D solution as an `OrthogonalExpansion` in its orthogonal basis, with the boundary lifting included.
    * Use the expansion for exact arithmetic: sums, products, powers, multiplication by polynomials in $x$, derivatives and evaluation. The number of basis functions grows as needed.
    * Use `to_jaxfunction(V)` to give an expansion back to a solver as a known function.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
