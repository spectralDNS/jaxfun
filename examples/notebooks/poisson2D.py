import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Poisson's equation in 2D

    Poisson's equation on the unit square with homogeneous Dirichlet boundary conditions (the solution is zero on the entire domain) and a constant right hand side is given as

    $$
    \begin{align}
    \nabla^2 u(x, y) &= 2, \quad (x, y) \in \Omega = (0, 1)^2 \tag{1}\\
    u(x, y) &= 0, \quad (x, y) \in \partial \Omega \tag{2}
    \end{align}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In order to solve this problem with the Galerkin method we choose a functionspace for our solution $T$ and attempt to find $u_{N} \in T$ such that

    $$
    (\nabla^2 u_N, v)_{L^2(\Omega)} = (2, v)_{L^2(\Omega)}, \quad \forall \, v \in T, \tag{3}
    $$

    where the real multidimensional $L^2(\Omega)$ inner product

    $$
    (u, v)_{L^2(\Omega)} = \int_{\Omega} u {v} d\Omega. \tag{4}
    $$

    That's the full extent of the Galerkin method! Now all we have to do is be specific about $T$ and everything will fall out from the variational equation. If we want to we can also use Green's first identity to arrive at another formulation that is very popular with the finite element method because the regularity of the solution $u_N$ can be lower ($u_N$ only need to be differentiable once, not twice)

    $$
    -(\nabla u_N, \nabla v)_{L^2(\Omega)} = (2, v)_{L^2(\Omega)}, \quad \forall \, v \in T. \tag{5}
    $$

    For the spectral Galerkin method using basis functions of very high order and regularity, there is practically no difference between the two approaches.

    Lets solve this problem using a composition of Legendre polynomials $P_j$, such that each basis function satisfies the homogeneous boundary conditions

    $$
    \phi_j = P_j - P_{j+2},
    $$

    and $V = \text{span}\{\phi_j\}_{j=0}^{N-1}$ and $T = V \otimes V$. Hence we have test functions $v \in T$ such that each basis function is

    $$
    v(x, y) = \phi_i(x) \phi_j(y).
    $$

    We also create a trial function $u_N$ with unknown expansion coefficients $\hat{u}_{kl}$

    $$
    u_N(x, y) = \sum_{k}\sum_l \hat{u}_{kl} \phi_k(x) \phi_l(y).
    $$

    The unknown $U=\{\hat{u}_{kl}\}_{k,l=0}^{N-1}$ is what we need to find through the Galerkin method.

    Inserting for test and trial functions in Eq. (3) we get

    $$
    \sum_{k=0}^{N-1}\sum_{l=0}^{N-1} \int_{\Omega} \left( \phi''_k(x) \phi_i(x) \phi_l(y)\phi_j(y) + \phi_k(x) \phi_i(x) \phi''_l(y)\phi_j(y) \right)\, d \Omega \, \hat{u}_{kl} =\int_{\Omega} 2 \phi_i(x) \phi_j(y) d\Omega,
    $$

    where the prime represents a derivative with respect to the given functions variable. That is, $\phi'_j(x) = \frac{\partial \phi_j(x)}{\partial x}$ and $\phi'_j(y) = \frac{\partial \phi_j(y)}{\partial y}$.

    The tensor on the left hand side contains no less than 4 indices and is a tensor of rank 4. However, since the basis functions are separable (they are all functions of one variable), we can split the integral into

    $$
    \sum_{k=0}^{N-1}\sum_{l=0}^{N-1} \left( \int_{0}^1 \phi''_k(x) \phi_i(x) dx \int_{0}^1 \phi_l(y)\phi_j(y) dy + \int_{0}^1 \phi_k(x) \phi_i(x) dx \int_{0}^1 \phi''_l(y)\phi_j(y)\, dy \right) \, \hat{u}_{kl} =\int_{\Omega} 2 \phi_i(x) \phi_j(y) d\Omega.
    $$

    Now introduce $s_{ik} = \int_0^1\phi''_k(x) \phi_i(x) dx = \int_0^1\phi''_k(y) \phi_i(y) dy$ and $a_{jl} = \int_0^1 \phi_j(x) \phi_l(x) dx = \int_0^1 \phi_j(y) \phi_l(y) dy$, where we get the same matrix for $x$ and $y$ directions since we use the same basis functions in both $x$ and $y$. The equation becomes

    $$
    \sum_{k=0}^{N-1}\sum_{l=0}^{N-1} \left( s_{ik} a_{jl} + a_{ik} s_{jl}\right) \hat{u}_{kl} = f_{ij},
    $$

    where $f_{ij} = \int_{\Omega} 2 \phi_i(x) \phi_j(y) d\Omega$. In matrix form, with $S=\{s_{ik}\}_{i,k=0}^{N-1}$, $A=\{a_{ik}\}_{i,k=0}^{N-1}$ and $F=\{f_{ij}\}_{i,j=0}^{N-1}$ we get the linear algebra problem

    $$
    SUA + AUS = F
    $$

    In order to solve for $U$ we can use the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) rule $\text{vec}(SUA) = (S \otimes A^T) \text{vec}(U)$, where $\otimes$ represent a Kronecker product and $\text{vec}$ represents a vectorization (flattening) (see also [lecture notes in MAT-MEK4270](https://matmek-4270.github.io/matmek4270-book/lecture6.html#vectorization-the-vec-trick)). The linear algebra problem becomes

    $$
    (S \otimes A^T + A \otimes S^T) \text{vec}(U) = \text{vec}(F)
    $$

    which can be solved for the flattened $\text{vec}(U)$.

    We are now ready to solve the Poisson problem using Jaxfun. With Jaxfun we choose functionspaces first and the assemble the matrices $S, A$ and $F$ from above. The entire problem is solved in the function `poisson()` below and should be pretty self-explanatory.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from jax import Array
    from jaxfun.operators import Div, Grad
    from jaxfun.galerkin import (
        inner,
        FunctionSpace,
        JAXFunction,
        Legendre,
        TestFunction,
        TensorProduct,
        TrialFunction,
    )
    from jaxfun.galerkin.tensorproductspace import vec
    from scipy import sparse as scipy_sparse

    def poisson(N: int) -> Array:
        V = FunctionSpace(
            N + 2,
            Legendre.Legendre,
            bcs={"left": {"D": 0}, "right": {"D": 0}},
            domain=(0, 1),
            name="V",
            fun_str="phi",
        )
        T = TensorProduct(V, V, name="T")
        u = TrialFunction(T, name="u")
        v = TestFunction(T, name="v")
        A = inner(Div(Grad(u)) * v)
        F = inner(2 * v)
        A0 = vec(A)
        h = jnp.array(scipy_sparse.linalg.spsolve(A0, F.flatten()).reshape(F.shape))
        xj = T.mesh(kind="uniform", N=(50, 50), broadcast=False)
        plt.contourf(xj[0], xj[1], T.backward(h, kind="uniform", N=(50, 50)))
        plt.colorbar()
        plt.show()
        return JAXFunction(h, T, name="h")

    h = poisson(20)
    return h, jnp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The returned `JAXFunction` `h` represents

    $$
    h(x, y) = \sum_{i=0}^{N-1}\sum_{j=0}^{N-1} \hat{h}_{ij} \phi_i(x) \phi_j(y)
    $$

    and can be evaluated for any point in the domain $[0, 1]^2$. We can display `h` both in unevaluated and evaluated state:
    """)
    return


@app.cell
def _(h):
    from IPython.display import display

    display(h)
    display(h.doit(linear=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note the repeated indices on the last expression, which imply summation.

    Evaluate `h` at a single point $h(x=0.5, y=0.5)$ in space:
    """)
    return


@app.cell
def _(h, jnp):
    h(jnp.array([0.5, 0.5]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Evaluate `h` on all quadrature points in the mesh:
    """)
    return


@app.cell
def _(h):
    xj = h.functionspace.mesh()
    hj = h.evaluate_mesh(xj)
    print(hj.shape)
    return (hj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This operation is normally called a backward transform, and it can also be computed as
    """)
    return


@app.cell
def _(h, hj, jnp):
    hj2 = h.backward()
    assert jnp.allclose(hj, hj2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that each functionspace V contains $N$ unknowns, but $N+2$ quadrature points since there are two restrictions on the boundary conditions.

    Alternatively, use a flattened mesh of shape $((N+2)^2, 2)$:
    """)
    return


@app.cell
def _(h):
    xf = h.functionspace.flatmesh()
    hf = h(xf)
    print(hf.shape, xf.shape)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
