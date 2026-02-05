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
    # Physics informed neural networks

    In this notebook we will use [Jaxfun](https://github.com/spectralDNS/jaxfun) together with

    * [Jax](https://docs.jax.dev/en/latest/index.html)
    * [Optax](https://optax.readthedocs.io/en/latest/)
    * [Flax](https://github.com/google/flax)

    in order to solve a differential equation using the least squares minimization formulation of the problem. We will use basis functions based on regular multi layer perceptrons as well as spectral expansions in orthogonal polynomials.

    The ubiquitous linear Helmholtz equation is defined as

    $$
    u^{''}(x) + \alpha u(x) = f(x), \quad x \in (-1, 1), \, \alpha \in \mathbb{R^+}
    $$

    and in this notebook it will be used with boundary conditions $u(-1)=u(1)=0$. The function $u(x)$ represents the unknown solution and the right hand side function $f(x)$ is continuous and known. We can define a residual $\mathcal{R}u(x)$ for the Helmholtz equation as

    $$
    \mathcal{R}u(x) = u^{''}(x) + \alpha u(x) - f(x),
    $$

    which should be zero for any point in the domain. In this notebook we will approximate $u(x)$ with a neural network $u_{\theta}(x)$, where $\theta \in \mathbb{R}^M$ represents the unknown weights of the network. In order to find $u_{\theta}(x)$ we will attempt to force the residual $\mathcal{R}u_{\theta}(x_i)$ to zero in a least squares sense for some $N$ chosen training points $\{x_i\}_{i=0}^{N-1}$. To this end the least squares problem reads

    \begin{equation*}
    \underset{\theta \in \mathbb{R}^M}{\text{minimize}}\, L(\theta):=\frac{1}{N}\sum_{i=0}^{N-1} \mathcal{R}u_{\theta}(x_i; \theta)^2 + \frac{1}{2} \left(u_{\theta}(-1; \theta)^2 + u_{\theta}(1; \theta)^2 \right)
    \end{equation*}

    We start by importing necessary functionality from both [jax](https://docs.jax.dev/en/latest/index.html), [optax](https://optax.readthedocs.io/en/latest/), [flax](https://github.com/google/flax) and Jaxfun. We also make use of Sympy in order to describe the equations.
    """)
    return


@app.cell
def _():
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import sympy as sp
    from flax import nnx

    from jaxfun.operators import Div, Grad
    from jaxfun.pinns import FlaxFunction, Loss, MLPSpace
    from jaxfun.pinns.optimizer import GaussNewton, Trainer, adam, lbfgs
    from jaxfun.utils.common import lambdify, ulp

    return (
        Div,
        FlaxFunction,
        GaussNewton,
        Grad,
        Loss,
        MLPSpace,
        Trainer,
        adam,
        jax,
        jnp,
        lambdify,
        lbfgs,
        nnx,
        plt,
        sp,
        ulp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the `MLPSpace` class, which represents a functionspace for a regular multilayer perceptron. The space will make use of a subclass of the [flax](https://flax.readthedocs.io/en/latest) [nnx.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#module-flax.nnx).

    We create this space holds information about the input, output and hidden layers in the neural network. Here we create an MLP for a one-dimensional problem (one input variable) and 16 neurons for a single hidden layer.
    """)
    return


@app.cell
def _(MLPSpace):
    V = MLPSpace([16], dims=1, name="V")
    return (V,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is possible to use several hidden layers, for example by choosing `V = MLPSpace([8, 8, 8], dims=1)`.

    The MLP function space is subsequently used to create a trial function for the neural network. The function `v` below holds all the unknown weights in the MLP.
    """)
    return


@app.cell
def _(FlaxFunction, V, nnx):
    v = FlaxFunction(
        V,
        rngs=nnx.Rngs(1001),
        name="v",
        fun_str="phi",
        kernel_init=nnx.initializers.xavier_uniform(),
    )
    return (v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inside `v`, the regular flax nnx module is accessible through `v.module`.

    We will test the solver using a known manufactured solution. We can use any solution, but it should be continuous and the solution needs to use the same symbols as Jaxfun. Below we choose a mixture of a second order polynomial (to get the correct boundary condition) an exponential and a cosine function. This function is continuous, but it requires quite a few unknowns in order to get a decent solution.
    """)
    return


@app.cell
def _(V, sp):
    x = V.system.x
    ue = (1 - x**2) * sp.exp(sp.cos(2 * sp.pi * x))
    return ue, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The equation to solve is now described in strong form using the residual $\mathcal{R}u_{\theta}$. Note that we create the right hand side function $f(x)$ from the manufactured solution.
    """)
    return


@app.cell
def _(Div, Grad, ue, v):
    alpha = 1
    fe = Div(Grad(ue)) + alpha * ue
    residual = Div(Grad(v)) + alpha * v - fe
    return alpha, residual


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two operators [Div](https://github.com/spectralDNS/jaxfun/blob/main/jaxfun/operators.py) and [Grad](https://github.com/spectralDNS/jaxfun/blob/main/jaxfun/operators.py) are defined in Jaxfun. For a 1D problem on the straight line there is no difference from writing the residual simply as

    residual = sp.diff(v, x, 2) + alpha * v - (sp.diff(ue, x, 2) + alpha*ue)

    We can look at the residual in code:
    """)
    return


@app.cell
def _(residual):
    residual
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that $\nabla \cdot$ represents divergence and $\nabla v$ represents the gradient of the scalar field $v$. The neural network function $v$ is written as $v(x; V)$ since $v$ is a function of $x$ and it is a function on the space $V$. The residual above is in unevaluated form. We can evaluate it using the Sympy function `doit`:
    """)
    return


@app.cell
def _(residual):
    residual.doit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that Sympy here evaluates derivatives to the best of its abilities, and the neural network function $v$ has been replaced by the expression $\phi(x)$. This is because `phi` is set as `fun_str` for the created `FlaxFunction` `v`.

    We need training points inside the domain in order to solve the least squares problem. Create random points `xj` using the helper class `Line` and an array `xb` that holds the coordinates of the two boundaries. Note that the argument to `mesh.get_points` is the total number of points in the mesh, including boundary points. Since it's a 1D mesh, the number of boundary points is 2.
    """)
    return


@app.cell
def _(jax):
    from jaxfun.pinns.mesh import Line

    mesh = Line(-1, 1, key=jax.random.PRNGKey(2002))
    xj = mesh.get_points(1200, domain="inside", kind="random")
    xb = mesh.get_points(1200, domain="boundary")
    return xb, xj


@app.cell
def _(xb):
    print(xb)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have two coupled problems to solve: The equation defined by `residual` and the boundary conditions. In order to solve these problems we now make use of the `Loss` class and a feed the two problems to it. We also need to feed the correct collocation points to each problem:
    """)
    return


@app.cell
def _(Loss, residual, v, xb, xj):
    loss_fn = Loss((residual, xj), (v, xb))
    loss_fn.residuals
    return (loss_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first residual represents the equation to solve. It contains two subequations representing $v''$ and $\alpha v$, wheras the constant part of the equation is placed in the target (an array of shape $N-2$).
    """)
    return


@app.cell
def _(loss_fn):
    loss_fn.residuals[0].eqs
    return


@app.cell
def _(loss_fn):
    print(loss_fn.residuals[0].target.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As such each `Residual` is a class holding the required functions in order to compute the residuals. The first one computes $\mathcal{R}u_{\theta}(x)$ and the other the boundary terms $u_{\theta}(-1)+u_{\theta}(1)$. Calling `loss_fn(v.module)` returns

    $$
    \frac{1}{N-2}\sum_{i=0}^{N-3} \mathcal{R}u_{\theta}(x_i; \theta)^2 + \frac{1}{2} \left(u_{\theta}(-1; \theta)^2 + u_{\theta}(1; \theta)^2 \right)
    $$

    In order to solve the least squares problem we need an optimizer. Any `optax` optimizer may be used, but we will start with Adam, and then switch to a more accurate optimizer after a while. We first run 5000 epochs with Adam
    """)
    return


@app.cell
def _(Trainer, adam, loss_fn, v):
    trainer = Trainer(loss_fn)
    opt_adam = adam(v.module, learning_rate=0.001)
    trainer.train(opt_adam, 5000, epoch_print=1000)
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Adam optimizer is good at eliminating local minima and as such it is good at finding a solution that is close to the global minimum. However, Adam is only first order and not able to find a very accurate solution. For this we need either a quasi-Newton or a Newton optimizer. We start with the limited-memory BFGS optimizer and take 10000 more epochs.
    """)
    return


@app.cell
def _(lbfgs, trainer, v):
    opt_lbfgs = lbfgs(v.module, memory_size=20)
    trainer.train(opt_lbfgs, 10000, epoch_print=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is possible to run even more BFGS epoch to further polish this root. However, we will in the end switch to an even more accurate Newton optimizer. Since the Newton optimizer is costly, we run only 10 epochs. The Newton optimizer should only be used when the residual is already close to zero.
    """)
    return


@app.cell
def _(GaussNewton, trainer, v):
    opt_hess = GaussNewton(v.module, use_lstsq=False, cg_max_iter=500)
    trainer.train(opt_hess, 10, epoch_print=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running even more epochs the solution will become even more accurate.

    We can now compute the $L^2$ error norm by comparing to the exact solution
    """)
    return


@app.cell
def _(jnp, lambdify, ue, v, x, xj):
    uej = lambdify(x, ue)(xj)  # Exact
    print("Error", jnp.linalg.norm(v.module(xj) - uej) / jnp.sqrt(len(xj)))
    return (uej,)


@app.cell
def _(jnp, lambdify, plt, ue, v, x):
    xa = jnp.linspace(-1, 1, 100)[:, None]
    uea = lambdify(x, ue)(xa)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(xa, v.module(xa), "b", label="PINNs")
    ax1.plot(xa, uea, "ro", label="Exact")
    ax1.legend()
    ax2.plot(xa, v.module(xa) - uea, "b*")
    ax1.set_title("Solutions")
    ax2.set_title("Error")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The residuals of each problem may also be computed using the `loss_fn` class. The residuals for the two boundary conditions are
    """)
    return


@app.cell
def _(loss_fn, v):
    loss_fn.compute_residual_i(v.module, 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spectral least squares solver

    The neural network is capturing the solution quite well, but the convergence is quite slow. We know that a problem like the Helmholtz equation with a continuous solution should be very well captured using Legendre or Chebyshev basis functions, that have much better approximation properties than the neural network. Using Jaxfun we can solve this problem with the Galerkin method, but we can also use the least squares formulation similar to as above.

    The solver below is using simply

    $$
    u_{\theta}(x) = \sum_{i=0}^{N-1} \hat{u}_i L_i(x)
    $$

    where $L_i(x)$ is the i'th Legendre polynomial and $\hat{u}_i$ are the unknowns.

    The least squares implementation goes as follows:
    """)
    return


@app.cell
def _(
    Div,
    FlaxFunction,
    GaussNewton,
    Grad,
    Loss,
    Trainer,
    adam,
    alpha,
    lbfgs,
    nnx,
    ue,
    ulp,
    xb,
    xj,
):
    from jaxfun.galerkin.Legendre import Legendre

    VN = Legendre(60)
    w = FlaxFunction(
        VN, rngs=nnx.Rngs(1001), kernel_init=nnx.initializers.xavier_uniform(), name="v"
    )
    res = Div(Grad(w)) + alpha * w - (Div(Grad(ue)) + alpha * ue)
    loss_fn_1 = Loss((res, xj), (w, xb))
    trainer_1 = Trainer(loss_fn_1)
    opt_adam_1 = adam(w.module)
    trainer_1.train(opt_adam_1, 5000, epoch_print=1000)
    opt_lbfgs_1 = lbfgs(w.module)
    trainer_1.train(opt_lbfgs_1, 1000, epoch_print=100)
    opt_hess_1 = GaussNewton(w.module, use_lstsq=True, cg_max_iter=100)
    trainer_1.train(opt_hess_1, 4, epoch_print=1, abs_limit_loss=ulp(1))
    return Legendre, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that with only 2 Newton iterations the error plunges to zero. This is because this method is an exact solver for linear equations and expansions like the Legendre polynomials. Comparing now with the exact solution we get a very accurate $L^2$ error
    """)
    return


@app.cell
def _(jnp, uej, w, xj):
    print("Error", jnp.linalg.norm(w.module(xj) - uej) / jnp.sqrt(len(xj)))
    return


@app.cell
def _(plt, uej, w, xj):
    plt.plot(xj, w.module(xj) - uej, "b*")
    plt.title("Error")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implicit spectral least squares
    """)
    return


@app.cell
def _(Div, Grad, Legendre, alpha, jnp, lambdify, ue, x, xj):
    from jaxfun.galerkin import FunctionSpace, TestFunction, TrialFunction, inner

    D = FunctionSpace(
        60, Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}}, name="D", fun_str="psi"
    )
    q = TestFunction(D, name="v")
    u = TrialFunction(D, name="u")
    A, L = inner(
        (Div(Grad(q)) + alpha * q) * (Div(Grad(u)) + alpha * u)
        - (Div(Grad(q)) + alpha * q) * (Div(Grad(ue)) + alpha * ue),
        sparse=True,
        sparse_tol=1000,
        return_all_items=False,
    )
    uh = jnp.linalg.solve(A.todense(), L)
    uj = D.evaluate(xj, uh)
    uej_1 = lambdify(x, ue)(xj)
    error = jnp.linalg.norm(uj - uej_1) / jnp.sqrt(len(xj))
    print(
        error
    )  # v * (Div(Grad(u)) + alpha * u) - v * (Div(Grad(ue)) + alpha * ue), # Galerkin  # LSQ
    return uej_1, uj


@app.cell
def _(plt, uej_1, uj, xj):
    plt.plot(xj, uj - uej_1, "b*")
    plt.title("Error")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The solution with implicit Galerkin or least squares is about as accurate as the regular least squares.
    """)
    return


if __name__ == "__main__":
    app.run()
