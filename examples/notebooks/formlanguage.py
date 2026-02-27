import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # The form language in Jaxfun (WIP)

    Jaxfun consists of a rich form-language that is used to formulate problems. This language is built on top of Sympy's Function and Vector classes. The fundamental idea is that you choose a functionspace and then you create functions on that space. For the global Galerkin method, this means that you work with functions like

    \begin{equation}
    v(x) = \sum_{i=0}^{N-1} \hat{v}_i \phi_i(x) \tag{1}
    \end{equation}

    where $v(x)$ is your function, $\phi_i(x)$ are $N$ basis functions and $\hat{v}_i$ are $N$ unknown expansion coefficients. The functionspace corresponding to this function is $V=\text{span}\{\phi_i\}_{i=0}^{N-1}$. When we say that $v(x) \in V$, this means that $v$ can be expanded as shown in Eq. (1), but we do not necessarily know the coefficients.

    A tensorproductspace in 2D Cartesian space can be defined as $T = V \otimes V$. A scalar function $f \in T$ is then defined as

    \begin{equation}
    f(x, y) = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} \hat{f}_{ij} \phi_i(x) \phi_j(y) \tag{2}
    \end{equation}

    with the unknown matrix $F=\{\hat{f}_{ij}\}_{i,j=0}^{N-1} \in \mathbb{R}^{N \times N}$.

    A vector tensorproductspace in 2D can be defined as $W = T \times T$. A function $\boldsymbol{w} \in W$ is then defined as

    \begin{equation}
    \boldsymbol{w}(x, y) = w^{(0)}(x, y) \boldsymbol{i} + w^{(1)}(x, y) \boldsymbol{j} \tag{3}
    \end{equation}

    where $\boldsymbol{i}$ and $\boldsymbol{j}$ are Cartesian unit vectors in $x$ and $y$ direction, respectively, and $w^{(i)}$ represents component $i$ of the vector. That is, $w^{(0)} = \boldsymbol{w} \cdot \boldsymbol{i}$ and $w^{(1)} = \boldsymbol{w} \cdot \boldsymbol{j}$. In Eq. (3) both $w^{(0)}$ and $w^{(1)}$ can be expanded as shown in Eq (2).

    In Jaxfun you choose the functionspace and then you create functions. Let's create samples of all the three Galerkin functions defined above ($v(x), f(x, y), \boldsymbol{w}(x, y)$):
    """)
    return


@app.cell
def _():
    from IPython.display import display

    import jax.numpy as jnp

    from jaxfun.galerkin import (
        JAXFunction,
        Legendre,
        TensorProduct,
        TestFunction,
        TrialFunction,
        VectorTensorProductSpace,
        inner,
    )

    N = 20
    V = Legendre.Legendre(N, name="V", fun_str="phi")
    T = TensorProduct(V, V, name="T")
    W = VectorTensorProductSpace(T, name="W")
    v = TestFunction(V, name="v")
    f = TestFunction(T, name="f")
    w = TestFunction(W, name="w")
    display(v)
    display(f)
    display(w)
    return JAXFunction, N, T, TrialFunction, V, W, display, f, inner, jnp, v, w


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
    Note that since test functions do not contain any expansion coefficients, only the basis functions are shown. The superscript in parenthesis on the vector components shows the index into the vector of basis functions. Hence, $\phi_i^{(0)}(x)$ is the $i$'th basis function in the first component of the vector space $W$, and $\phi_j^{(1)}(y)$ is the $j$'th basis function in the second component of $W$.

    A trial function is similar to the test function, but represents the unknown solution that we are trying to find. The trial function is displayed exactly like the test function, but with a different index. Remember the test function on $V$ was displayed as $\phi_i(x)$. The trial function is displayed as $\phi_j(x)$:
    """)
    return


@app.cell
def _(TrialFunction, V, display):
    u = TrialFunction(V, name="u")
    display(u)
    display(u.doit())
    return (u,)


@app.cell
def _(mo):
    mo.md(r"""
    Trial and test functions are used together to assemble matrices in variational forms. Consider the L2 inner product

    $$
    (u, v)_{L^2(\Omega)} = \int_{\Omega} \phi_j(x)\phi_i(x)dx, (i, j) \in (0, \ldots, N-1) \times (0, \ldots, N-1)
    $$

    where $u$ is a trial function and $v$ is a test function. The left hand side is in unevaluated form and the right hand side with the integral is in evaluated form using the basis functions retrieved from `doit`. This inner product results in a matrix $A \in \mathbb{R}^{N \times N}$ with components $a_{ij} = \int_{\Omega} \phi_j(x)\phi_i(x)dx$.
    """)
    return


@app.cell
def _(display, inner, u, v):
    A = inner(u * v)
    display(u * v)
    display((u * v).doit())
    print("Shape of matrix A =", A.shape)
    return (A,)


@app.cell
def _(mo):
    mo.md(r"""
    The integral is computed using numerical quadrature on the quadrature points of the Legendre basis functions. For multidimensional problems we get tensor product matrices, that we will get back to in a little while. First, there is a third function that deserves some attentions. The `JAXFunction`s are complete Galerkin functions, containing also the expansion coefficients of the Galerkin function.
    """)
    return


@app.cell
def _(JAXFunction, N, V, display, jnp):
    h = JAXFunction(jnp.ones(N), V, name="h")
    display(h)
    display(h.doit(linear=True))
    return (h,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The JAXFunction displayed above is shown first in unevaluated state. It is shown as a function of $x$, created on the function space `V`. When evaluated using `doit`, the JAXFunction returns the product of its expansion coefficients $\hat{h}_j$ and trial basis functions (using the same index as the trial function). Note that there is summation implied by repeating indices. Unlike the test and trial functions, the JAXFunction may appear in expressions in non-linear form. In that case, we do not expand the function with indices, but treat it like any other computable function. Note the missing keyword argument `linear=True` below:
    """)
    return


@app.cell
def _(display, h):
    display(h.doit())
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can create JAXFunctions for any of the three spaces created initially in this notebook:
    """)
    return


@app.cell
def _(JAXFunction, N, T, display, jnp):
    b = JAXFunction(jnp.ones((N, N)), T, name="b")
    display(b)
    display(b.doit(linear=True))
    display(b.doit())
    return


@app.cell
def _(JAXFunction, N, W, display, jnp):
    c = JAXFunction(jnp.ones((2, N, N)), W, name="c")
    display(c)
    display(c.doit(linear=True))
    display(c.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Consider the $L^2$ inner product $(h, v)_{L^2(\Omega)}$. The integrand is now
    """)
    return


@app.cell
def _(h, v):
    (h * v).doit(linear=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the repeated $j$ index, which implies summation (Einstein summation convention). There is only one free index, $i$, and a such `inner(h*v)` returns a vector.
    """)
    return


@app.cell
def _(h, v):
    from jaxfun.galerkin.forms import split

    split(v * h)
    return


@app.cell
def _(h, inner, v):
    hv = inner(h * v)
    print(hv.shape)
    return (hv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The vector `hv` equals the matrix vector product $A h$
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

    where $\theta = \{W^l, \boldsymbol{b}^l\}_{l=1}^L$ represents all the unknowns in the model and $W^l, \boldsymbol{b}^l$ represents the weights and biases on level $l$. The model input $\boldsymbol{z}$ can be anything. For example, $\boldsymbol{z}$ can represent $(x), (x, y), (x, y, z), (x, t), (x, y, t), ...$. For $Z$ number of inputs the scalar function $M_{\theta}: \mathbb{R}^Z \rightarrow \mathbb{R}$. Similarly, a vector function in 2D space is simply $M_{\theta}: \mathbb{R}^Z \rightarrow \mathbb{R}^2$. The neural networks are represented in Jaxfun through a `FlaxFunction`, and we can use them in any numer of dimensions and also of vector rank.
    """)
    return


@app.cell
def _(display):
    from jaxfun.pinns import FlaxFunction, MLPSpace

    M1 = MLPSpace([20], dims=1, rank=0, name="M1")
    M2 = MLPSpace([20], dims=2, rank=0, name="M2")
    M3 = MLPSpace([20], dims=2, rank=1, name="M3")
    f1 = FlaxFunction(M1, name="a")
    f2 = FlaxFunction(M2, name="b")
    f3 = FlaxFunction(M3, name="c")
    display(f1)
    display(f2)
    display(f3)
    display(f1.doit())
    display(f2.doit())
    display(f3.doit())
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
