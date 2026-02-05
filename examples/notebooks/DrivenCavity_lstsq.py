import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lid driven cavity

    In this notebook we use Physics Informed Neural Networks (PINNS) to solve the Navier-Stokes equations in a rectangular domain. The Navier-Stokes problem to solve is

    \begin{align*}
    (\boldsymbol{u} \cdot \nabla) \boldsymbol{u} - \nu \nabla^2 \boldsymbol{u} + \nabla p &= 0, \quad \boldsymbol{x} \in \Omega = (-1, 1)^2\\
    \nabla \cdot \boldsymbol{u} &= 0, \quad \boldsymbol{x} \in \Omega = (-1, 1)^2 \\
    \end{align*}

    where $\nu$ is a constant kinematic viscosity, $p(\boldsymbol{x})$ is pressure, $\boldsymbol{u}(\boldsymbol{x}) = u_x(\boldsymbol{x}) \boldsymbol{i} + u_y(\boldsymbol{x}) \boldsymbol{j}$ is the velocity vector and the position vector $\boldsymbol{x} = x \boldsymbol{i} + y \boldsymbol{j}$, with unit vectors $\boldsymbol{i}$ and $\boldsymbol{j}$. The Dirichlet boundary condition for the velocity vector is zero everywhere, except for the top lid, where $\boldsymbol{u}(x, y=1) = (1-x)^2(1+x)^2 \boldsymbol{i}$. There is no boundary condition on pressure, but since the pressure is only present inside a gradient, it can only be found up to a constant. Hence, we specify that $p(x=0, y=0) = 0$.

    We start the implementation by importing necessary modules such as [jax](https://docs.jax.dev/en/latest/index.html), [flax](https://flax.readthedocs.io/), where the latter is a module that provides neural networks for JAX.
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from flax import nnx

    jax.config.update("jax_enable_x64", True)
    return jnp, nnx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will solve the Navier-Stokes equations using a multilevel perceptron neural network, where the solution will be approximated as

    $$
    F_{\theta}(\boldsymbol{x}; \theta) =  W^{L} \sigma( W^{L-1} \ldots \sigma( W^1 \boldsymbol{x} + \boldsymbol{b}^1) \ldots + \boldsymbol{b}^{L-1})  + \boldsymbol{b}^L
    $$

    where $\theta = \{W^l, \boldsymbol{b}^l\}_{l=1}^L$ represents all the unknowns in the model and $W^l, \boldsymbol{b}^l$ represents the weights and biases on level $l$. The model contains both velocity components and the pressure, for a total of three scalar outputs: $u_{x}(\boldsymbol{x}), u_y(\boldsymbol{x})$ and $p(\boldsymbol{x})$. Hence $F_{\theta}: \mathbb{R}^2 \rightarrow \mathbb{R}^3$.

    We split the coupled $F_{\theta}$ into velocity and pressure

    $$
    F_{\theta} = \boldsymbol{u}_{\theta} \times p_{\theta}
    $$

    where $\boldsymbol{u}_{\theta}: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ is a vector function and $p_{\theta}: \mathbb{R}^2 \rightarrow \mathbb{R}$ is a scalar function.

    We start the implementation by creating multilayer perceptron functionspaces for $\boldsymbol{u}_{\theta}$ and $p_{\theta}$, and then combining these to a space for $F_{\theta}$ using a Cartesian product:
    """)
    return


@app.cell
def _(nnx):
    from jaxfun.pinns import FlaxFunction, MLPSpace, Comp

    V = MLPSpace([16], dims=2, rank=1, name="V")  # Vector space for velocity
    Q = MLPSpace([12], dims=2, rank=0, name="Q")  # Scalar space for pressure

    u = FlaxFunction(V, "u", rngs=nnx.Rngs(2002))
    p = FlaxFunction(Q, "p", rngs=nnx.Rngs(2002))
    module = Comp(u, p)
    return V, module, p, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `u` and `p` both are `FlaxFunction`s, that are subclasses of the Sympy [Function](https://docs.sympy.org/latest/modules/functions/index.html). However, in Jaxfun these functions have some additional properties, that makes it easy to describe equations.

    We can inspect `u` and `p`
    """)
    return


@app.cell
def _(p, u):
    from IPython.display import display

    display(u)
    display(u.doit())
    display(p)
    display(p.doit())
    return (display,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that `u` and `p` are in unevaluated state, whereas `u.doit()` and `p.doit()` returns sympy functions for the computational space. If we check the type of `u.doit()`, we get that it is a `VectorAdd`, because the vector is an addition of the two vector components in $(u_x(x, y))\boldsymbol{i} + (u_y(x, y)) \boldsymbol{j}$. The three terms $u_x(x, y), u_y(x, y)$ and $p(x, y)$ are all sympy functions and of type `AppliedUndef`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now describe the Navier-Stokes equations using `Div`, `Grad` and `Dot` from Jaxfun. First specify the Reynolds number, the kinematic viscosity and then the equations
    """)
    return


@app.cell
def _(p, u):
    import sympy as sp
    from jaxfun.operators import Div, Dot, Grad, Constant

    Re = 10.0  # Define Reynolds number
    nu = Constant(
        "nu", 2.0 / Re
    )  # Define kinematic viscosity. A number works as well, but the Constant prints better.
    R1 = Dot(Grad(u), u) - nu * Div(Grad(u)) + Grad(p)
    R2 = Div(u)
    return R1, R2, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here `R1` represents the residual of the momentum vector equation $\mathcal{R}^1_{\theta}$, whereas `R2` represents the residual of the scalar divergence constraint $\mathcal{R}^2_{\theta}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can inspect the residuals:
    """)
    return


@app.cell
def _(R1, display):
    display(R1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `R1` represents a vector equation and we can expand it using `doit`:
    """)
    return


@app.cell
def _(R1):
    R1.doit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The divergence constraint is a scalar equation
    """)
    return


@app.cell
def _(R2, display):
    display(R2)
    display(R2.doit())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To solve the equations we will use a least squares method and for this we need to create collocation points both inside and on the domain. There are some simple helper functions in `jaxfun.pinns.mesh` that can help use create such points. Below we create a total of $N^2$ points in a rectangular mesh. We separate these points into $N_i=N^2-4N$ points `xyi` inside the domain $\boldsymbol{x}^{\Omega} = \{(x_i, y_i)\}_{i=0}^{N_i-1}$, and $N_b=4N$ points `xyb` on the boundary of the domain $\boldsymbol{x}^{\partial \Omega} = \{(x_i, y_i)\}_{i=0}^{4N-1}$, including the four corners. The last point `xyp` is simply origo and used to fix the pressure in a single point.
    """)
    return


@app.cell
def _(jnp):
    from jaxfun.pinns.mesh import Rectangle

    N = 20
    mesh = Rectangle(-1, 1, -1, 1)
    xyi = mesh.get_points(N * N, 4 * N, domain="inside", kind="random")
    xyb = mesh.get_points(N * N, 4 * N, domain="boundary", kind="random")
    xyp = jnp.array([[0.0, 0.0]])
    return xyb, xyi, xyp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The boundary conditions on the velocity vector needs to be specified and to this end we can use a boundary function $ue(x, y) = (1-x)^2(1+x)^2 \boldsymbol{i}$ created as a Sympy function
    """)
    return


@app.cell
def _(V, sp):
    x, y = V.system.base_scalars()
    ub = sp.Mul(
        sp.Piecewise((0, y < 1), ((1 - x) ** 2 * (1 + x) ** 2, True)), V.system.i
    )
    return (ub,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The unknowns will now be found using the least squares method, which is to minimize

    \begin{equation*}
    \underset{\theta \in \mathbb{R}^M}{\text{minimize}}\, L(\theta):=\frac{1}{N_i}\sum_{k=0}^1\sum_{i=0}^{N_i-1} \mathcal{R}^{1}_{\theta} (\boldsymbol{x}^{\Omega}_i)^2 \cdot \boldsymbol{i}_k + \frac{1}{N_i}\sum_{i=0}^{N_i-1} \mathcal{R}^{2}_{\theta}(\boldsymbol{x}^{\Omega}_i)^2+ \frac{1}{N_b} \sum_{k=0}^1 \sum_{i=0}^{N_b-1} (\boldsymbol{u}_{\theta}(\boldsymbol{x}^{\partial \Omega}_i) - \boldsymbol{u}_b(\boldsymbol{x}^{\partial \Omega}_i))^2 \cdot \boldsymbol{i}_k + p_{\theta}(0, 0)
    \end{equation*}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    where $\boldsymbol{i}_0 = \boldsymbol{i}$ and $\boldsymbol{i}_1 = \boldsymbol{j}$.
    We define the minimization problem using the jaxfun class `Loss`, which takes as arguments tuples containing the residual, the collocation points to use for that residual, the target (defaults to zero) and optionally some weights. The weights may be constants or 1D arrays of length the number of collocation points. If weights are provided, then these are applied to the squared residuals in each term above. For example, the divergence loss becomes

    $$
    \frac{1}{N_i}\sum_{i=0}^{N_i-1} \omega_i \mathcal{R}^{2}_{\theta}(\boldsymbol{x}^{\Omega}_i)^2
    $$

    using weights $\{\omega_i\}_{i=0}^{N_i-1}$. Below the pressure anchor is simply weighted with a constant factor 5.
    """)
    return


@app.cell
def _(R1, R2, p, u, ub, xyb, xyi, xyp):
    from jaxfun.pinns import Loss

    loss_fn = Loss((R1, xyi), (R2, xyi), (u - ub, xyb), (p, xyp, 0, 5))
    return (loss_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are six scalar losses computed with `loss_fn`
    """)
    return


@app.cell
def _(loss_fn):
    print(loss_fn.residuals)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each loss may be computed individually, or we can compute the whole sum $L(\theta)$. Since we have not started the least squares solver yet, the module for now only contains randomly initialized weights and the loss is significant
    """)
    return


@app.cell
def _(loss_fn, module):
    loss_fn(module)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To create a solver, we use the [optax](https://optax.readthedocs.io/en/latest/) module, and start with the Adam optimizer
    """)
    return


@app.cell
def _(loss_fn, module):
    from jaxfun.pinns.optimizer import Trainer, adam

    trainer = Trainer(loss_fn)
    opt_adam = adam(module)
    trainer.train(opt_adam, 5000, epoch_print=1000)
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The loss has now been reduced, but is still far from zero.
    """)
    return


@app.cell
def _(loss_fn, module):
    loss_fn(module)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To refine the solution, we use a low-memory BFGS solver:
    """)
    return


@app.cell
def _(module, trainer):
    from jaxfun.pinns.optimizer import lbfgs

    opt_lbfgs = lbfgs(module, memory_size=100)
    trainer.train(opt_lbfgs, 5000, epoch_print=1000)
    return


@app.cell
def _(loss_fn, module):
    loss_fn(module)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally an incomplete Newton method, which is still quite slow because we have not yet implemented a preconditioner.
    """)
    return


@app.cell
def _(module, trainer):
    from jaxfun.pinns.optimizer import GaussNewton

    opt_hess = GaussNewton(module, use_lstsq=False, cg_max_iter=100)
    trainer.train(opt_hess, 100, epoch_print=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now visualize the solution using [matplotlib](https://matplotlib.org/)
    """)
    return


@app.cell
def _(jnp, module):
    import matplotlib.pyplot as plt

    yj = jnp.linspace(-1, 1, 50)
    xx, yy = jnp.meshgrid(yj, yj, sparse=False, indexing="ij")
    z = jnp.column_stack((xx.ravel(), yy.ravel()))
    uvp = module(z)
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6, 2))
    axs[0].contourf(xx, yy, uvp[:, 0].reshape(xx.shape), 100)
    axs[0].set_title(r"$u_x$")
    axs[1].contourf(xx, yy, uvp[:, 1].reshape(xx.shape), 100)
    axs[1].set_title(r"$u_y$")
    axs[2].contourf(xx, yy, uvp[:, 2].reshape(xx.shape), 100)
    axs[2].set_title(r"$p$")
    fig.set_tight_layout("tight")
    plt.show()
    return plt, uvp, xx, yy


@app.cell
def _(plt, uvp, xx, yy):
    plt.figure(figsize=(4, 4))
    plt.quiver(xx, yy, uvp[:, 0].reshape(xx.shape), uvp[:, 1].reshape(xx.shape))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also compute the losses of each equation. Here we get the loss of each collocation point in $\boldsymbol{x}^{\Omega}$ for the momentum equation in the x-direction:
    """)
    return


@app.cell
def _(loss_fn, module):
    loss_fn.compute_residual_i(module, 0)[:10]  # plot only 10 numbers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The loss can be plotted using for example scatter. Below is the scatter plot of the error in the divergence constraint.
    """)
    return


@app.cell
def _(loss_fn, module, plt, xyi):
    plt.figure(figsize=(4, 3))
    plt.scatter(*xyi.T, c=loss_fn.compute_residual_i(module, 2), s=20)
    plt.colorbar()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally a vector plot of the velocity vectors:
    """)
    return


@app.cell
def _(jnp, plt, uvp, xx, yy):
    plt.figure(figsize=(4, 4))
    plt.quiver(xx, yy, uvp[:, 0], uvp[:, 1], jnp.linalg.norm(uvp[:, :2], axis=1))
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
