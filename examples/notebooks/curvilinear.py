import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curvilinear coordinates

    Jaxfun is developed to work with any curvilinear coordinate system and not just the regular Cartesian. A user should describe equations for a generic coordinate system using operators like `div`, `grad`, `dot`, `outer`, `cross` and `curl`. The correct equations in curvilinear coordinates should then be automatically derived by Jaxfun under the hood. What goes on under the hood is described in more detail in this article.

    In the following we will consider two coordinate systems, the Cartesian with coordinates

    \begin{equation}
        \mathbf{x} = \{x^{i}\}_{i\in\mathcal{I}^n},
    \end{equation}

    and the computational, with coordinates

    \begin{equation}
        \mathbf{X} = \{X^{i}\}_{i\in\mathcal{I}^m}.
    \end{equation}

    Here $\mathcal{I}^m$ and $\mathcal{I}^n$ are two index sets for the number of computational coordinates, and the dimension of the physical space, respectively, and $m\leq n$. Throughout this article an index set will be denoted as $\mathcal{I}^{N}=\{0, 1, \ldots, N-1\}$, where $N\in \mathbb{Z}^+$.

    The physical domain will be denoted as $\Omega^m \subseteq \mathbb{R}^n$, where $m=1,2$ and 3 represents curves, surfaces and volumes, respectively. Likewise, the computational domain will be denoted as $\mathbb{I}^m \subseteq \mathbb{R}^m$. The computational domain is a hypercube, with $\mathbb{I}^2 = \text{I}^0 \times \text{I}^1$ and $\mathbb{I}^3 = \text{I}^0 \times \text{I}^1 \times \text{I}^2$, where $\text{I}^m$ is the interval used for computational dimension $m$. Hence, computations are always performed on hypercubes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The coordinate curves $X^{i}$ are functions of the Cartesian coordinates

    \begin{equation}
     X^{i} = X^{i}(\mathbf{x}), \quad \text{for } i \in \mathcal{I}^m, \tag{3}
    \end{equation}

    and we get the Cartesian coordinates through the inverse maps

    \begin{equation}
    x^i=x^i(\mathbf{X}), \quad \text{for } i \in \mathcal{I}^n.\tag{4}
    \end{equation}

    Using these generic maps a function $u : \Omega^m \rightarrow \mathbb{K}$, where $\mathbb{K}$ may be either $\mathbb{R}$ or $\mathbb{C}$, may now be transformed to the computational space as

    \begin{equation}
    u(\mathbf{x}) = u(\mathbf{x}(\mathbf{X})) = \tilde{u}(\mathbf{X}),\tag{5}
    \end{equation}

    where $\tilde{u} : \mathbb{I}^m \rightarrow \mathbb{K}$. A tilde is used throughout this article to denote a function transformed to the computational domain, but we will not use a tilde for functions that are defined directly and used exclusively in computational space. Also, we will skip the middle step above from now on and simply use either $u(\mathbf{x})$ or $\tilde{u}(\mathbf{X})$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A position vector can now be written as a function of the new coordinates

    \begin{equation}
        \mathbf{r}(\mathbf{X}) = x^i(\mathbf{X}) \,\mathbf{i}_i,\tag{6}
    \end{equation}

    where there is here, and throughout this article, summation implied by repeating indices. The Cartesian unit basis vectors are denoted as $\mathbf{i}_i\in\mathbb{R}^n$ for $i\in \mathcal{I}^n$.

    The Jacobian matrix for the transformation between coordinate systems is described by

    \begin{equation}
        J_{ij}(\mathbf{X}) = \frac{\partial x^{i}}{\partial X^{j}}, \quad \text{for } i,j \in \mathcal{I}^{n \times m}, \tag{7}
    \end{equation}

    where the index set  $\mathcal{I}^{n \times m} = \mathcal{I}^m \times \mathcal{I}^n$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The covariant basis vectors are defined as

    \begin{equation}
        \mathbf{b}_i(\mathbf{X}) = \frac{\partial \mathbf{r}}{\partial X^{i}}, \quad \text{for } i \, \in \mathcal{I}^m,\tag{8}
    \end{equation}

    and contravariant basis vectors are defined as

    \begin{equation}
        \mathbf{b}^{i}(\mathbf{X}) =  \mathbf{i}_j\frac{\partial X^{i}}{\partial x^j}, \quad \text{for } i \, \in \mathcal{I}^m.\tag{9}
    \end{equation}

    The co- and contravariant basis vectors satisfy

    \begin{equation}
        \mathbf{b}^{i} \cdot \mathbf{b}_{j} = \delta_{j}^{i}, \quad \text{for } i, j \, \in \mathcal{I}^{m \times m}.\tag{10}
    \end{equation}

    A vector $\mathbf{v}(\mathbf{x}) = \tilde{\mathbf{v}}(\mathbf{X})$ can be given in either basis as

    \begin{equation}
        \mathbf{v} = \tilde{v}^{i} \mathbf{b}_i = \tilde{v}_i \mathbf{b}^{i},\tag{11}
    \end{equation}

    where $\tilde{v}_{i}(\mathbf{X})$ and $\tilde{v}^{i}(\mathbf{X})$ are co- and contravariant vector components, respectively. The vector can also be given in a Cartesian coordinate system as $\mathbf{v}(\mathbf{x}) = v^{j}\mathbf{i}_j$.

    The components $\tilde{v}^{i}$ are called contravariant vector components and the components $\tilde{v}_i$ are called covariant vector components. When a vector is written with contravariant vector components $\mathbf{v} = \tilde{v}^{i} \mathbf{b}_i$, the vector is called a contravariant vector. This is the default used by Jaxfun. When the vector is written with covariant components $\mathbf{v} = \tilde{v}_i \mathbf{b}^{i}$ it is called a covariant vector. This is a bit confusing since a contravariant vector is using covariant basis vectors and vice versa!

    In what follows we will drop the tilde on the vector components and work with simply

    \begin{equation}
        \mathbf{v} = {v}^{i} \mathbf{b}_i = {v}_i \mathbf{b}^{i}.
    \end{equation}

    It should be obvious from the context that we are working with curvilinear vectors in curvilinear coordinates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In Jaxfun we work by default in the covariant basis with contravariant vector components. However, we can obtain the covariant components ${v}_{i}$ as

    $$
    {v}_{i} = {v}^j \mathbf{b}_j \cdot \mathbf{b}_i = {v}^j g_{ji}
    $$

    The co- and contravariant metric tensors are defined as

    \begin{align}
    g_{ij}(\mathbf{X}) &= \mathbf{b}_i \cdot \mathbf{b}_j \quad \text{for } i, j \in \mathcal{I}^{m \times m}, \tag{12} \\
    g^{ij}(\mathbf{X}) &= \mathbf{b}^{i} \cdot \mathbf{b}^{j} \quad \text{for } i, j \in \mathcal{I}^{m \times m}, \tag{13}
    \end{align}

    respectively. Finally, the determinant of the covariant metric tensor is given as

    \begin{equation}
        g(\mathbf{X}) = \det([g_{ij}]),\tag{14}
    \end{equation}

    where $[g_{ij}]$ represents the tensor with components $g_{ij}$.
    Note that if $m=n$, then $\sqrt{g}$ is equal to the determinant of the Jacobian matrix. Also note that the basis vectors are not necessarily orthogonal. If they are, then $[g_{ij}]$ and $[g^{ij}]$ are diagonal matrices.

    We can get the contravariant basis vectors from the covariant and vice versa using

    $$
    \begin{align*}
    \mathbf{b}^{i} &= \mathbf{b}_{i} g^{ij} \\
    \mathbf{b}_{i} &= \mathbf{b}^{i} g_{ij}
    \end{align*}
    $$

    The metric tensors are available in Jaxfun using methods `get_covariant_metric_tensor` and `get_contravariant_metric_tensor` on the given curvilinear coordinate system.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The vector dot product can be defined for curvilinear coordinates as

    $$
    \mathbf{u} \cdot \mathbf{v} = {u}^{i}\mathbf{b}_i \cdot {v}_j \mathbf{b}^{j} = {u}^{i} {v}_i
    $$

    because of Eq. (10).

    The dot product between two vectors both expressed in a covariant basis can be computed as

    $$
    {u}^{i}\mathbf{b}_i \cdot {v}^{j} \mathbf{b}_{j} = {u}^{i} {v}^{j} g_{ij}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will illustrate by creating a cylindrical coordinate system, defined from the position vector

    $$
    \mathbf{r}(r, \theta, z) = r \cos \theta \mathbf{i} + r \sin \theta \mathbf{j} + z \mathbf{k}
    $$
    """)
    return


@app.cell
def _():
    from IPython.display import display
    from jaxfun.operators import Dot, dot
    from jaxfun.coordinates import get_CoordSys
    import sympy as sp

    r, theta, z = sp.symbols("r,theta,z", real=True, positive=True)
    C = get_CoordSys(
        "C", sp.Lambda((r, theta, z), (r * sp.cos(theta), r * sp.sin(theta), z))
    )
    r, theta, z = C.base_scalars()
    return C, Dot, display, dot, r, theta, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that we here on the last line overwrite the sympy symbols `r`, `theta`, `z` with objects of type `BaseScalar`.

    Create two contravariant vectors and take the dot product:
    """)
    return


@app.cell
def _(C, Dot, display):
    u = 4 * C.b_r + 3 * C.b_theta + 2 * C.b_z
    v = C.b_r
    d = Dot(u, v)
    display(d)
    display(d.doit())
    return u, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `Dot` represents an unevaluated dot product. It is evaluated by calling `doit()`. Under the hood this invokes the `dot` function. We could alternatively just use the `dot`function:
    """)
    return


@app.cell
def _(dot, u, v):
    dot(u, v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The covariant metric tensor is given for cylindrical coordinates as:
    """)
    return


@app.cell
def _(C):
    print(C.get_covariant_metric_tensor())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The metric tensor is diagonal since the cylindrical coordinates are orthogonal.

    We can also create vectors in Cartesian Coordinates and transform them into the curvilinear system. For example
    """)
    return


@app.cell
def _(C, display):
    N = C._parent  # The Cartesian coordinate system
    p = N.x * N.i + N.y * N.j + N.z * N.k
    display(p)
    return (p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running `doit` on this vector transforms the components into computational coordinates, but does not change basis
    """)
    return


@app.cell
def _(display, p):
    display(p.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can get this vector $\mathbf{p}$ in cylinder coordinates by computing

    $$
    \boldsymbol{p} = (\boldsymbol{p} \cdot \boldsymbol{b}_r) \boldsymbol{b}_r + (\boldsymbol{p} \cdot \boldsymbol{b}_{\theta}) \boldsymbol{b}_{\theta} + (\boldsymbol{p} \cdot \boldsymbol{b}_{z}) \boldsymbol{b}_{z} = r \boldsymbol{b}_r + z \mathbf{b}_z
    $$

    This is called a projection of the vector $\boldsymbol{p}$ to the cylinder basis. The projection can be computed with method `from_cartesian`:
    """)
    return


@app.cell
def _(C, display, p):
    display(C.from_cartesian(p))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross product

    The cross product between two vectors can be obtained as

    $$
    \mathbf{u} \times \mathbf{v} = \varepsilon_{ijk} \sqrt{g} {u}^j {v}^k \mathbf{b}^{i} = \varepsilon_{ijk} \sqrt{g} {u}^j {v}^k g^{il}\mathbf{b}_l
    $$
    """)
    return


@app.cell
def _(C, display):
    from jaxfun.operators import Cross

    cr = Cross(C.b_r, C.b_theta)
    display(cr)
    display(cr.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dyadics

    The tensor product (or outer product) of two vectors is written

    $$
    \mathbf{A} = \mathbf{u} \otimes \mathbf{v}
    $$

    The second order tensor $\mathbf{A}$ is called a dyad. There are four possible basis configurations for dyads

    $$
    \begin{align*}
    \mathbf{b}^{i} &\otimes \mathbf{b}^{j} \\
    \mathbf{b}^{i} &\otimes \mathbf{b}_{j} \\
    \mathbf{b}_{i} &\otimes \mathbf{b}^{j} \\
    \mathbf{b}_{i} &\otimes \mathbf{b}_{j} \\
    \end{align*}
    $$

    In Jaxfun we use the last one, with two covariant base vectors $\mathbf{b}_{i} \otimes \mathbf{b}_{j}$. In this case

    $$
    \mathbf{A} = A^{ij} \mathbf{b}_i \otimes \mathbf{b}_j.
    $$

    where $A^{ij}$ are called the contravariant components.

    The outer product of two vectors is computed as

    $$
    \mathbf{u} \otimes \mathbf{v} = {u}^{i} \mathbf{b}_{i} \otimes {v}^{j} \mathbf{b}_{j} = {u}^{i} {v}^{j} \mathbf{b}_{i} \otimes \mathbf{b}_{j}
    $$

    Dyads are actually defined by how they contract with vectors

    $$
    \begin{align*}
    (\mathbf{u} \otimes \mathbf{v}) \cdot \mathbf{w} &= (\mathbf{v} \cdot \mathbf{w}) \mathbf{u}\\
    \mathbf{w} \cdot (\mathbf{u} \otimes \mathbf{v}) &= (\mathbf{w} \cdot \mathbf{u}) \mathbf{v}
    \end{align*}
    $$

    Note that some authors do not use a dot for the contraction of a dyad with a vector and would write instead $(\mathbf{u} \otimes \mathbf{v}) \mathbf{w}$. Since we use the `Dot/dot` in Jaxfun to represent the contraction of both dyadics and vectors, we find it more appropriate to include the dot also for contraction of higher order tensors.

    A linear combination of dyads is called a dyadic. For example $\mathbf{A} = 2(\mathbf{u} \otimes \mathbf{v}) + 3 (\mathbf{w} \otimes \mathbf{z})$. Note that even though a dyad is a second order tensor, not all second order tensors are dyads. That is, we cannot always write the tensor $\mathbf{A}$ as the tensor product of two vectors $\mathbf{u} \otimes \mathbf{v}$

    The contraction of two dyads leads to a dyad

    $$
    (\mathbf{u} \otimes \mathbf{v}) \cdot (\mathbf{w} \otimes \mathbf{z}) = (\mathbf{v} \cdot \mathbf{w}) (\mathbf{u} \otimes \mathbf{z})
    $$

    Let create some dyadics in Jaxfun and illustrate. First, the 9 dyad base vectors can be obtained as
    """)
    return


@app.cell
def _(C, display):
    bb = C.base_dyadics()
    display(bb)
    return (bb,)


@app.cell
def _(bb, display, r, theta):
    import numpy as np

    bv = np.array(bb, dtype=object).reshape((3, 3))
    A = 2 * bv[0, 1] + 3 * bv[1, 2]
    B = r * bv[1, 1] + theta * bv[2, 2]
    display(A)
    display(B)
    return A, B, bv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Contract `A` with `B`
    """)
    return


@app.cell
def _(A, B, Dot, display):
    H = Dot(A, B)
    display(H)
    display(H.doit())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Operators in curvilinear coordinates

    ### Vector operations

    The gradient of a scalar field $f(\mathbf{x})$ is termed $\text{grad}(f)$ and equals

    \begin{equation}
    \text{grad}(f) = \nabla f = \frac{\partial {f}}{\partial X^{i}}\,\mathbf{b}^{i} = \frac{\partial {f}}{\partial X^{i}} g^{ij} \,\mathbf{b}_{j}. \tag{15}
    \end{equation}

    The divergence of a vector $\mathbf{v}(\mathbf{x})$, in terms of its contravariant components, is given as

    \begin{equation}
    \text{div}(\mathbf{v}) = \nabla \cdot \mathbf{v} = \frac{\partial \mathbf{v}}{\partial X^j} \cdot \mathbf{b}^j = \frac{1}{\sqrt{g}} \frac{\partial {v}^{i} \sqrt{g}}{\partial X^{i}},\tag{16}
    \end{equation}

    and the Laplacian of the scalar field $f$ can be written as

    \begin{equation}
    \text{div}(\text{grad}(f)) = \nabla^2 f  = \frac{1}{\sqrt{g}}\frac{\partial}{\partial X^{i}}\left( g^{ij} \sqrt{g} \frac{\partial {f}}{\partial X^{j}}\right). \tag{17}
    \end{equation}

    Note that $g^{ij}$ and $\sqrt{g}$ will act as variable coefficients when the operators are used in differential equations.

    The curl of a vector is defined as

    $$
    \text{curl}(\mathbf{v}) = \nabla \times \mathbf{v} = \mathbf{b}^{j} \times \frac{\partial \mathbf{v}}{\partial X^j} = \frac{\varepsilon_{ijk}}{\sqrt{g}} \frac{\partial {v}_k}{\partial X^j} \mathbf{b}_i
    $$

    The partial derivative $\frac{\partial \mathbf{v}}{\partial X^j}$ is implemented in Jaxfun as the `diff` operator working on vectors. The partial derivative can be computed as

    $$
    \frac{\partial \mathbf{v}}{\partial X^j} = \left(\frac{\partial {v}^{i}}{\partial X^j} + \Gamma^{i}_{kj} {v}^k\right) \mathbf{b}_i
    $$

    where $\Gamma^{i}_{kj}$ is the Christoffel symbol of the second kind

    $$
    \Gamma^{k}_{ji} = \frac{\partial \mathbf{b}_{i}}{\partial X^j} \cdot \mathbf{b}^k
    $$

    available in Jaxfun as method `get_christoffel_second`.

    The term $\frac{\partial {v}^{i}}{\partial X^j} + \Gamma^{i}_{kj} {v}^k$ is called the covariant derivative of vector components.

    The Christoffel symbol falls out of the last identity for the curl due to the symmetry $\Gamma^{k}_{ji} = \Gamma^k_{ij}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tensor operations

    The gradient of a vector is a second order tensor

    $$
    \text{grad}(\mathbf{v}) = \frac{\partial \mathbf{v}}{\partial X^j} \otimes \mathbf{b}^j
    $$

    Note that if $\nabla$ is interpreted as a vector, then this corresponds to $(\nabla \otimes \mathbf{v})^T$. However, in Jaxfun printing we interpret $\nabla$ as an operator and as such $\text{grad}(\mathbf{v})$ is printed as $\nabla \mathbf{v}$.

    The divergence of a second order tensor $\mathbf{A}$ is

    $$
    \text{div}(\mathbf{A}) = \frac{\partial \mathbf{A}}{\partial X^j} \cdot \mathbf{b}^j
    $$

    Again, if $\nabla$ is interpreted as a vector, then $\text{div}(\mathbf{A}) = \nabla \cdot \mathbf{A}^T$. But in Jaxfun's printing $\nabla$ is an operator acting over the last axis of its input tensor. Hence the divergence of a tensor is printed as $\nabla \cdot \mathbf{A}$.
    """)
    return


@app.cell
def _(C, display, r, theta):
    from jaxfun.operators import Grad, Div

    w = r * C.b_r + theta * C.b_z
    g = Grad(w)
    display(g)
    display(g.doit())
    return Div, Grad


@app.cell
def _(Div, bv, display, r, z):
    D = r * bv[0, 0] + z * bv[1, 2]
    q = Div(D)
    display(q)
    display(q.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curvilinear equations

    Lets consider the Laplace operator in curvilinear coordinates. For this we use a `ScalarFunction` and Jaxfun operators:
    """)
    return


@app.cell
def _(C, Div, Grad, display):
    from jaxfun.galerkin.arguments import ScalarFunction

    u_ = ScalarFunction(name="u", system=C)
    Lu = Div(Grad(u_))
    display(Lu)
    return Lu, u_


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Laplace operator is displayed in Cartesian coordinates. When evaluated, we get the equation in cylinder coordinates:
    """)
    return


@app.cell
def _(Lu, display):
    display(Lu.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the map from Cartesian function $u(x, y, z)$ for the generic (unevaluated) operator to computational $U(r, \theta, z)$ in the evaluated equation.

    We can also look at the gradient
    """)
    return


@app.cell
def _(Grad, display, u_):
    display(Grad(u_))
    return


@app.cell
def _(Grad, display, u_):
    display(Grad(u_).doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the small difference here from most textbooks that present the gradient as (see, e.g., [wikipedia](https://en.wikipedia.org/wiki/Gradient))

    $$
    \nabla u = \frac{\partial U}{\partial r} \hat{\mathbf{b}}_r + \frac{1}{r}\frac{\partial U}{\partial \theta} \hat{\mathbf{b}}_{\theta} + \frac{\partial U}{\partial z}\hat{\mathbf{b}}_z
    $$

    using normalized (unit) basis vectors $\hat{\mathbf{b}}_r = {\mathbf{b}}_r, \hat{\mathbf{b}}_{\theta} = \mathbf{b}_{\theta} / r$ and $\hat{\mathbf{b}}_z = \mathbf{b}_z$. Replacing the hat vectors above with the covariant basis vectors leads to the same result.
    """)
    return


app._unparsable_cell(
    r"""
    The curl of a gradient should be zero. Lets verify
    """,
    name="_",
)


@app.cell
def _(Grad, display, u_):
    from jaxfun.operators import Curl

    display(Curl(Grad(u_)))
    return (Curl,)


@app.cell
def _(C, Curl, Grad, display, u_):
    display(C.simplify(Curl(Grad(u_)).doit()))
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
