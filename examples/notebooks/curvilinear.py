import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Curvilinear coordinates

    Jaxfun is developed to work with any curvilinear coordinate system and not just the regular Cartesian. A user should describe equations for a generic coordinate system using operators like `div`, `grad`, `dot`, `outer`, `cross` and `curl`. The correct equations in curvilinear coordinates should then be automatically derived by Jaxfun under the hood. What goes on under the hood is described in more detail in this article.

    There is not really enough time or space to go through all details regarding curvilinear coordinates here, so the interested reader is referred to the excellent online textbook by Kelly, PA. Mechanics Lecture Notes: An introduction to Solid Mechanics.  Available from [here](http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/index.html)

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

    Using these generic maps a function $u : \Omega^m \rightarrow \mathbb{K}$, where $\mathbb{K}$ may be either $\mathbb{R}$ or $\mathbb{C}$, may now be transformed from physical to computational space through a change of variables as

    \begin{equation}
    u(\mathbf{x}) = u(\mathbf{x}(\mathbf{X})) = U(\mathbf{X}),\tag{5}
    \end{equation}

    A simple and well known example is for polar coordinates, where the Cartesian coordinates $x$ and $y$ are mapped to the polar coordinates $r$ and $\theta$ as

    $$
    x(r, \theta) = r \cos \theta \quad \text{and} \quad  y(r, \theta) = r \sin \theta
    $$

    A Cartesian function for a circle of radius one is $u(x, y) = \sqrt{x^2 + y^2} -1$. Inserting for the polar maps we get $u(x(r, \theta), y(r, \theta)) = \sqrt{(r \cos \theta)^2 + (r \sin \theta)^2} - 1 = r - 1 = U(r, \theta)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The transformation for any curvilinear coordinate system will be defined through the use of a position vector, which can be written as a function of the new coordinates

    \begin{equation}
        \mathbf{r}(\mathbf{X}) = x^i(\mathbf{X}) \,\mathbf{i}_i,\tag{6}
    \end{equation}

    where there is here, and throughout this article, summation implied by repeating indices. The Cartesian unit basis vectors are denoted as $\mathbf{i}_i\in\mathbb{R}^n$ for $i\in \mathcal{I}^n$. The position vector can describe any point in the physical domain $\Omega^m$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A curvilinear coordinate system can be described using two different sets of basis vectors, the co- or contravariant basis vectors.

    The covariant basis vectors are defined as

    \begin{equation}
        \mathbf{b}_i(\mathbf{X}) = \frac{\partial \mathbf{r}}{\partial X^{i}}, \quad \text{for } i \, \in \mathcal{I}^m,\tag{8}
    \end{equation}

    The covariant basis vectors $\mathbf{b}_i$ are tangent to the $m$ coordinate curves $X^i(\mathbf{x})$.

    The contravariant basis vectors are defined as

    \begin{equation}
        \mathbf{b}^{i}(\mathbf{X}) =  \mathbf{i}_j\frac{\partial X^{i}}{\partial x^j}, \quad \text{for } i \, \in \mathcal{I}^m.\tag{9}
    \end{equation}

    The contravariant basis vectors are orthogonal to the covariant basis vectors

    \begin{equation}
        \mathbf{b}^{i} \cdot \mathbf{b}_{j} = \delta_{j}^{i}, \quad \text{for } i, j \, \in \mathcal{I}^{m \times m}.\tag{10}
    \end{equation}

    Any vector $\mathbf{v}(\mathbf{X})$ (boldface lower case letter) can be given with either co- or contravariant basis functions in the computational domain and we write them as

    \begin{equation}
        \mathbf{v} = {v}^{i} \mathbf{b}_i = {v}_i \mathbf{b}^{i},\tag{11}
    \end{equation}

    where ${v}_{i}(\mathbf{X})$ and ${v}^{i}(\mathbf{X})$ are co- and contravariant vector components, respectively. The sub- or superscript of the coefficient are used to recognise which basis functions are used. The vector $\mathbf{v}$ can also be given in Cartesian basis vectors, but since both sub- and superscripts are taken, we must make use of an alternative notation that includes the Cartesian variables $\mathbf{v}(\mathbf{x}) = v^{i}(\mathbf{x}) \mathbf{i}_i$.

    When a vector is written with contravariant vector components $\mathbf{v} = {v}^{i} \mathbf{b}_i$, the vector is called a contravariant vector. This is the default used by Jaxfun. When the vector is written with covariant components $\mathbf{v} = {v}_i \mathbf{b}^{i}$ it is called a covariant vector. This is a bit confusing since a contravariant vector is using covariant basis vectors and vice versa!
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
    Note that if $m=n$, then $\sqrt{g}$ is equal to the determinant of the Jacobian matrix. Also note that if the basis vectors are orthogonal, then $[g_{ij}]$ and $[g^{ij}]$ are diagonal matrices and the co- and contravariant basis vectors  $\mathbf{b}_{i}$ and $\mathbf{b}^{i}$ are parallel.

    We can get the contravariant basis vectors from the covariant and vice versa using

    $$
    \begin{align*}
    \mathbf{b}^{i} &= g^{ij} \mathbf{b}_{j} \\
    \mathbf{b}_{i} &= g_{ij} \mathbf{b}^{j}
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
    We illustrate first with a simple example. Consider a simple skewed coordinate system defined by $x(u, v) = u + v$ and $y(u, v) = v$, with curvilinear coordinates $u, v \in [0, 2]\times [0, 2]$, such that the position vector is

    $$
    \mathbf{r}(u, v) = (u+v) \mathbf{i} + v \mathbf{j},
    $$

    where $\mathbf{i}$ and $\mathbf{j}$ are the Cartesian unit basis vectors. The coordinate curves are straight lines, but they are not orthogonal to each other. The covariant basis vectors are $\mathbf{b}_u = \frac{\partial \mathbf{r}}{\partial u} = \mathbf{i}$ and $\mathbf{b}_v = \frac{\partial \mathbf{r}}{ \partial v} = \mathbf{i} + \mathbf{j}$, and the contravariant basis vectors are $\mathbf{b}^u = \mathbf{i} - \mathbf{j}$ and $\mathbf{b}^v = \mathbf{j}$. The basis vectors are illustrated in the figure below. Note that the covariant base vectors align with the constant gridlines (the coordinate curves of $u$ and $v$) and the contravariant base vectors are orthogonal to the gridlines. The contravariant vectors are as such the vectors that point orthogonally out from the physical domain $\Omega$. That is, $\mathbf{b}^{u}$ points out of the domain below defined by the curveline drawn at constant $v=2$, i.e., $u = x - 2$.
    """)
    return


@app.cell(hide_code=True)
def _(get_CoordSys, np, sp):
    from jaxfun.utils import lambdify
    import matplotlib.pyplot as plt

    def plot():
        u, v = sp.symbols("u, v", real=True, positive=True)
        P = get_CoordSys("P", sp.Lambda((u, v), (u + v, v)))
        u, v = P.base_scalars()
        rv = P.position_vector(False)
        assert isinstance(rv, sp.Tuple)

        b = P.get_covariant_basis()
        bt = P.get_contravariant_basis()

        r0 = 2.0
        t0 = 1.0
        pos = np.array(
            [lambdify((u, v), rv[0])(r0, t0), lambdify((u, v), rv[1])(r0, t0)]
        )

        b_num = np.array(
            [[lambdify((u, v), b[i, j])(r0, t0) for j in range(2)] for i in range(2)],
            dtype=float,
        )
        bt_num = np.array(
            [[lambdify((u, v), bt[i, j])(r0, t0) for j in range(2)] for i in range(2)],
            dtype=float,
        )

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

        ax.quiver(
            *pos,
            *b_num[0],
            color="tab:blue",
            angles="xy",
            scale_units="xy",
            scale=1,
            label=r"$\mathbf{b}_u$",
        )
        ax.quiver(
            *pos,
            *b_num[1],
            color="tab:blue",
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.5,
            label=r"$\mathbf{b}_v$",
        )
        ax.quiver(
            *pos,
            *bt_num[0],
            color="tab:orange",
            angles="xy",
            scale_units="xy",
            scale=1,
            label=r"$\mathbf{b}^u$",
        )
        ax.quiver(
            *pos,
            *bt_num[1],
            color="tab:orange",
            angles="xy",
            scale_units="xy",
            scale=1,
            alpha=0.5,
            label=r"$\mathbf{b}^v$",
        )

        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-0.2, 2.2)
        ax.legend(loc="upper left")
        plt.show()

    plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cylindrical coordinate system

    We will illustrate further by creating an orthogonal cylindrical coordinate system, defined from the position vector

    $$
    \mathbf{r}(r, \theta, z) = r \cos \theta \mathbf{i} + r \sin \theta \mathbf{j} + z \mathbf{k}
    $$

    where $r \in [0, 1]$ and $\theta \in (0, 2 \pi]$
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
    return C, Dot, display, dot, get_CoordSys, r, sp, theta, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the curvilinear coordinate system is created from only the forward map $\mathbf{x}(\mathbf{X})$ and we do not need to provide the reverse map $\mathbf{X}(\mathbf{x})$, which for the cylinder coordinates would be

    $$
    r = \sqrt{x^2+y^2}, \quad \theta = \tan^{-1}(y/x), \quad z = z.
    $$

    This is because Jaxfun is using the covariant basis vectors computed from the forward map, and the contravariant basis vector may then be computed simply from the $\mathbf{b}^i = g^{ij} \mathbf{b}_j$, where $[g^{ij}] = [g_{ij}]^{-1}$. So the reverse map is not explicitly used and as such it does not have to be provided by the user.

    Note that on the last line in the code section above we overwrite the sympy symbols `r`, `theta`, `z` with objects of type `BaseScalar`.

    As a first example create two contravariant vectors and take the dot product:
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

    We can also create vectors in Cartesian Coordinates and transform them into the curvilinear system. For example, the position vector in Cartesian coordinates is
    """)
    return


@app.cell
def _(C, display):
    R = C._parent  # The Cartesian coordinate system
    p = R.x * R.i + R.y * R.j + R.z * R.k
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
    We can get this vector $\mathbf{p}$ as fully contravariant by computing

    $$
    \boldsymbol{p} = (\boldsymbol{p} \cdot \boldsymbol{b}_r) \boldsymbol{b}_r + (\boldsymbol{p} \cdot \boldsymbol{b}_{\theta}) \boldsymbol{b}_{\theta} + (\boldsymbol{p} \cdot \boldsymbol{b}_{z}) \boldsymbol{b}_{z} = r \boldsymbol{b}_r + z \mathbf{b}_z
    $$

    This is called a projection of the vector $\boldsymbol{p}$ onto the covariant cylinder basis. The projection can be computed with method `from_cartesian`:
    """)
    return


@app.cell
def _(C, display, p):
    display(C.from_cartesian(p))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also go the other way and compute a Cartesian vector from a contravariant:
    """)
    return


@app.cell
def _(C):
    C.to_cartesian(C.b_theta)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can look at the contravariant basis vectors in terms of Cartesian coordinates
    """)
    return


@app.cell
def _(C):
    C.get_contravariant_basis(True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The point is that a vector $\mathbf{v}$ can be formulated using several different basis vectors and there may be advantages or disadvantages to all of them.
    """)
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

    A linear combination of dyads is called a dyadic. For example $\mathbf{A} = 2(\mathbf{u} \otimes \mathbf{v}) + 3 (\mathbf{w} \otimes \mathbf{z})$. Note that even though a dyad is a second order tensor, not all second order tensors are dyads. That is, we cannot always write the tensor $\mathbf{A}$ as the tensor product of two vectors $\mathbf{u} \otimes \mathbf{v}$.

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
    return A, B, bv, np


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

    The gradient of a scalar field $f(\mathbf{x}) (=F(\mathbf{X}))$ is termed $\text{grad}(f)$ and equals

    \begin{equation}
    \text{grad}(f) = \nabla f = \frac{\partial {F}}{\partial X^{i}}\,\mathbf{b}^{i} = \frac{\partial {F}}{\partial X^{i}} g^{ij} \,\mathbf{b}_{j}. \tag{15}
    \end{equation}

    Note that the computation in curvilinear coordinates is simply using a change of variables and the chain rule

    $$
    \nabla f = \frac{\partial f(\mathbf{x})}{\partial x_i} \mathbf{i}_{i} = \frac{\partial F(\mathbf{X})}{\partial X^j} \frac{\partial X^j}{\partial x_i} \mathbf{i}_{i} = \frac{\partial F}{\partial X^j} \mathbf{b}^j.
    $$

    The divergence of a vector $\mathbf{v}$, in terms of its contravariant components, is given as

    \begin{equation}
    \text{div}(\mathbf{v}) = \nabla \cdot \mathbf{v} = \frac{\partial \mathbf{v}}{\partial X^j} \cdot \mathbf{b}^j = \frac{1}{\sqrt{g}} \frac{\partial {v}^{i} \sqrt{g}}{\partial X^{i}},\tag{16}
    \end{equation}

    and the Laplacian of the scalar field $f$ can be written as

    \begin{equation}
    \text{div}(\text{grad}(f)) = \nabla^2 f  = \frac{1}{\sqrt{g}}\frac{\partial}{\partial X^{i}}\left( g^{ij} \sqrt{g} \frac{\partial {F}}{\partial X^{j}}\right). \tag{17}
    \end{equation}

    Note that $g^{ij}$ and $\sqrt{g}$ will act as variable coefficients when the operators are used in differential equations.

    The curl of a vector is defined as

    $$
    \text{curl}(\mathbf{v}) = \nabla \times \mathbf{v} = \mathbf{b}^{j} \times \frac{\partial \mathbf{v}}{\partial X^j} = \frac{\varepsilon_{ijk}}{\sqrt{g}} \frac{\partial {v}_k}{\partial X^j} \mathbf{b}_i \tag{18}
    $$

    The partial derivative $\frac{\partial \mathbf{v}}{\partial X^j}$ is implemented in Jaxfun as the `diff` operator working on vectors. The partial derivative can be computed as

    $$
    \frac{\partial \mathbf{v}}{\partial X^j} = \left(\frac{\partial {v}^{i}}{\partial X^j} + \Gamma^{i}_{kj} {v}^k\right) \mathbf{b}_i \tag{19}
    $$

    where $\Gamma^{i}_{kj}$ is the Christoffel symbol of the second kind

    $$
    \Gamma^{k}_{ji} = \frac{\partial \mathbf{b}_{i}}{\partial X^j} \cdot \mathbf{b}^k \tag{20}
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
    \text{grad}(\mathbf{v}) = \frac{\partial \mathbf{v}}{\partial X^j} \otimes \mathbf{b}^j \tag{21}
    $$

    Note that if $\nabla$ is interpreted as a vector, then this corresponds to $(\nabla \otimes \mathbf{v})^T$. However, in Jaxfun printing we interpret $\nabla$ as an operator and as such $\text{grad}(\mathbf{v})$ is printed as $\nabla \mathbf{v}$.

    The divergence of a second order tensor $\mathbf{A}$ is

    $$
    \text{div}(\mathbf{A}) = \frac{\partial \mathbf{A}}{\partial X^j} \cdot \mathbf{b}^j \tag{22}
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The curl of a gradient should be zero. Lets verify:
    """)
    return


@app.cell
def _(Grad, display, u_):
    from jaxfun.operators import Curl

    display(Curl(Grad(u_)))
    return (Curl,)


@app.cell
def _(Curl, Grad, display, u_):
    display(Curl(Grad(u_)).doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inner products

    The weighted $L_{\omega}^2(\Omega^m)$ inner product is defined as

    $$
    (u(\mathbf{x}), v(\mathbf{x}))_{L_{\omega}^2(\Omega^m)} = \int_{\Omega^m}u(\mathbf{x}) \overline{v}(\mathbf{x}) \omega(\mathbf{x}) d\lambda^m \tag{23}
    $$

    where as before $m$= 1, 2 or 3 represents a curve, surface or volume, and  $d\lambda^m$ represents an infinitesimal line, surface or volume element. The overline on $\overline{v}$ represents a complex conjugate.

    The computational domain is a hypercube, with $\mathbb{I}^2 = \text{I}^0 \times \text{I}^1$ and $\mathbb{I}^3 = \text{I}^0 \times \text{I}^1 \times \text{I}^2$, where $\text{I}^m$ is the interval used for computational dimension $m$. When computing the inner product, we first transform the integral to computational coordinates. We use the following mapped  function approximations

    $$
    u(\mathbf{x}) = U(X^0) = \sum_{k=0}^{N-1} \hat{u}_k \phi_k(X^0),
    $$

    $$
    u(\mathbf{x}) = U(X^0, X^1) = \sum_{k=0}^{N_0-1}\sum_{j=0}^{N_1-1} \hat{u}_{kj} \phi_k(X^0) \psi_j(X^1),
    $$

    $$
    u(\mathbf{x}) = U(X^0, X^1, X^2) = \sum_{k=0}^{N_0-1}\sum_{j=0}^{N_1-1} \sum_{i=0}^{N_2-1}\hat{u}_{kji} \phi_k(X^0) \psi_j(X^1) \gamma_i(X^2),
    $$

    for curves, surfaces or volumes.

    Consider a differential equation of the form

    \begin{equation}
        L u(\mathbf{x}) = f(\mathbf{x}), \quad \text{for } \mathbf{x} \in \Omega^m, \tag{24}
    \end{equation}

    where $L$ is a linear operator, $f(\mathbf{x}) \in L^2_{\omega}(\Omega^m)$, and $u(\mathbf{x}) \in \text{V}(\Omega^m)$. The weighted variational forms obtained by multiplying the differential equation by a test function and integrating over the domain are

    \begin{align}
        a(u, v)_{{L_{\omega}^2(\Omega^m)}} &= \int_{\Omega^m} L u(\mathbf{x}) \,v^*(\mathbf{x}) \, \omega(\mathbf{x}) \, d\lambda^m, \tag{25} \\
        (f, v)_{L_{\omega}^2(\Omega^m)} &= \int_{\Omega^m} f(\mathbf{x}) \,v^*(\mathbf{x}) \, \omega(\mathbf{x}) \, d\lambda^m, \tag{26}
    \end{align}

    and the weighted Galerkin method is to find $u \in \text{V}(\Omega^m)$ such that

    \begin{equation}
        a(u, v)_{L_{\omega}^2(\Omega^m)} = (f, v)_{L_{\omega}^2(\Omega^m)}, \quad \forall \, v \in \text{V}(\Omega^m). \tag{27}
    \end{equation}

    For either curves, surfaces or volumes, the integrals are transformed to

    \begin{align}
        a(u, v)_{L_{\omega}^2(\Omega^m)} &=  \int_{\mathbb{I}^m} {L} U \, \overline{V} \, W \, \sqrt{g} d{X}, \tag{28}\\
        (f, v)_{L_{\omega}^2(\Omega^m)} &= \int_{\mathbb{I}^m} F \,\overline{V} \, W \, \sqrt{g} d{X}, \tag{29}
    \end{align}

    where $d{X}=\prod_{i\in \mathcal{I}^m} dX^{i}$ and ${L}U$ is an operator transformed to computational space, like Eqs. (17). The computational weights $W(\mathbf{X})$ are determined by the choice of basis functions. Note that the transformation makes use of $d\lambda^m = \sqrt{g}d{X}$ for all $m \le n$.

    An example solving Helmholtz equation on an annulus is given in the following  [notebook](https://spectraldns.github.io/jaxfun/annulus.html).
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
