import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # The form language in Jaxfun (WIP)

    Jaxfun consists of a rich form-language that is used to formulate problems. This language is built on top of Sympy's Function and Vector classes. The fundamental idea is that you choose a functionspace and then you create functions on that space. For the global Galerkin method, this means that you work with functions like

    \begin{equation}
    v(x) = \sum_{i=0}^N \hat{v}_i \phi_i(x) \tag{1}
    \end{equation}

    where $v(x)$ is your function, $\phi_i(x)$ are $N+1$ basis functions and $\hat{v}_i$ are $N+1$ unknown expansion coefficients. The functionspace corresponding to this function is $V=\text{span}\{\phi_i\}_{i=0}^N$. When we say that $v(x) \in V$, this means that $v$ can be expanded as shown in Eq. (1), but we do not necessarily know the coefficients.

    A tensorproductspace in 2D Cartesian space can be defined as $T = V \otimes V$. A scalar function $f \in T$ is then defined as

    \begin{equation}
    f(x, y) = \sum_{i=0}^N \sum_{j=0}^N \hat{f}_{ij} \phi_i(x) \phi_j(y) \tag{2}
    \end{equation}

    with the unknown matrix $F=\{f_{ij}\}_{i,j=0}^N \in \mathbb{R}^{(N+1) \times (N+1)}$.

    A vector tensorproductspace in 2D can be defined as $W = T \times T$. A function $\boldsymbol{w} \in W$ is then defined as

    \begin{equation}
    \boldsymbol{w}(x, y) = w_x(x, y) \boldsymbol{i} + w_y(x, y) \boldsymbol{j} \tag{3}
    \end{equation}

    where $\boldsymbol{i}$ and $\boldsymbol{j}$ are Cartesian unit vectors in $x$ and $y$ direction, respectively. In Eq. (3) both $w_x$ and $w_y$ can be expanded as shown in Eq (2).

    In Jaxfun you choose the functionspace and then you create functions. Let's create samples of all the three Galerkin functions defined above ($v(x), f(x, y), \boldsymbol{w}(x, y)$):
    """)
    return


@app.cell
def _():
    from IPython.display import display

    from jaxfun.galerkin import (
        Legendre,
        TensorProduct,
        TestFunction,
        TrialFunction,
        VectorTensorProductSpace,
        inner,
    )

    N = 8
    V = Legendre.Legendre(N, name="V", fun_str="phi")
    T = TensorProduct(V, V, name="T")
    W = VectorTensorProductSpace(T, name="W")
    v = TestFunction(V, name="v")
    f = TestFunction(T, name="f")
    w = TestFunction(W, name="w")
    display(v)
    display(f)
    display(w)
    return N, T, TrialFunction, V, W, display, f, inner, v, w


@app.cell
def _(mo):
    mo.md(r"""
    Note that the three displayed functions are all in unevaluated state. They appear as the left hand side of their definition. We see $v(x; V)$, which is the left hand side of Eq. (1), with an additional $V$ to show that this function has been created from the space named "V".

    We can get the basisfunctions from the right hand side by evaluating the expressions:
    """)
    return


@app.cell
def _(display, f, v, w):
    display(v.doit())
    display(f.doit())
    display(w.doit())
    return


@app.cell
def _(mo):
    mo.md(r"""
    Note that since TestFunctions do not contain any expansion coefficients, only the basis functions are shown.

    JAXFunctions are complete functions, containing also the expansion coefficients. When evaluated, the JAXFunction returns the product of the expansion coefficients and a basis function.
    """)
    return


@app.cell
def _(N, V, display):
    import jax.numpy as jnp
    from jaxfun.galerkin import JAXFunction

    h = JAXFunction(jnp.ones(N), V, name="h")
    display(h)
    display(h.doit())
    return JAXFunction, h, jnp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here there is summation implied by repeating indices. The same goes for the other two function spaces.
    """)
    return


@app.cell
def _(JAXFunction, N, T, display, jnp):
    b = JAXFunction(jnp.ones((N, N)), T, name="b")
    display(b)
    display(b.doit())
    return


@app.cell
def _(JAXFunction, N, W, display, jnp):
    c = JAXFunction(jnp.ones((2, N, N)), W, name="c")
    display(c)
    display(c.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider the L2 inner product

    $$
    (u, v)_{L^2(\Omega)} = \int_{\Omega} \phi_j(x)\phi_i(x)dx, (i, j) \in (0, \ldots, N) \times (0, \ldots, N)
    $$

    This inner product results in a matrix $A \in \mathbb{R}^{N+1 \times N+1}$ with components $a_{ij} = \int_{\Omega} \phi_j(x)\phi_i(x)dx$. We can create the trial function `u`, and when printing the evaluated trial function, it appears with the index j.
    """)
    return


@app.cell
def _(TrialFunction, V, display, inner, v):
    u = TrialFunction(V, name="u")
    A = inner(u * v)
    print(A.shape)
    display(u * v)
    display((u * v).doit())
    return (A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the JAXFunction `h` we get instead
    """)
    return


@app.cell
def _(h, v):
    (h * v).doit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the repeated $j$ index, which implies summation. There is only one free index, $i$, and a such `inner(h*v)` returns a vector.
    """)
    return


@app.cell
def _(h, inner, v):
    hv = inner(h * v)
    print(hv.shape)
    return (hv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The vector `hv` equals the matrix vector product$A h$
    """)
    return


@app.cell
def _(A, h, hv, jnp):
    assert jnp.all(hv == A @ h)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the tensorproductspace we get 4 free indices, $\phi_i(x)\phi_j(y)$ from the test function and $\phi_k(x)\phi_l(y)$ from the trial.
    """)
    return


@app.cell
def _(T, TrialFunction, display, f):
    g = TrialFunction(T, "g")
    display((g * f).doit())
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hence the inner product leads to a tensor with 4 indices.
    """)
    return


@app.cell
def _(display, f, g, inner):
    B = inner(f * g)
    display(B)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `TPMatrix` B is a tensor product matrix equal to the outer product of two regular matrices $b_{ijkl}=b^{(0)}_{ik}b^{(1)}_{jl}$, where $b^{(0)}_{ik}=\int_{\Omega_x}\phi_i(x)\phi_k(x)dx$ and $b^{(1)}_{jl}=\int_{\Omega_y}\phi_j(y)\phi_l(y)dy$, with $\Omega = \Omega_x \times \Omega_y$. The two smaller matrices are stored under `B.mats`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A multilayer perceptron neural network is defined as the function

    $$
    M(\boldsymbol{z}; \theta) =  W^{L} \sigma( W^{L-1} \ldots \sigma( W^1 \boldsymbol{z} + \boldsymbol{b}^1) \ldots + \boldsymbol{b}^{L-1})  + \boldsymbol{b}^L
    $$

    where $\theta = \{W^l, \boldsymbol{b}^l\}_{l=1}^L$ represents all the unknowns in the model and $W^l, \boldsymbol{b}^l$ represents the weights and biases on level $l$. The model input $\boldsymbol{z}$ can be anything. For example, $\boldsymbol{z}$ can represent $(x), (x, y), (x, y, z), (x, t), (x, y, t), ...$. For $Z$ number of inputs the scalar function $M_{\theta}: \mathbb{R}^Z \rightarrow \mathbb{R}$. Similarly, a vector function in 2D space is simply $M_{\theta}: \mathbb{R}^Z \rightarrow \mathbb{R}^2$.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
