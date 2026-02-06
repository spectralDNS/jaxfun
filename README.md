# jaxfun

Spectral / Galerkin / PINNs experimentation toolkit built on top of JAX for fast differentiable ODE / PDE prototyping, variational forms, and mixed spectral bases.

## Status & Badges

[![Tests](https://github.com/spectralDNS/jaxfun/actions/workflows/pytest.yml/badge.svg)](https://github.com/spectralDNS/jaxfun/actions/workflows/pytest.yml)
[![Linting](https://github.com/spectralDNS/jaxfun/actions/workflows/lint.yml/badge.svg)](https://github.com/spectralDNS/jaxfun/actions/workflows/lint.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue)](https://spectraldns.github.io/jaxfun/)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-brightgreen.svg)](pyproject.toml)

> The coverage badge will update once Codecov is fully configured.

## Features

- Orthogonal polynomial and Fourier bases (Chebyshev, Legendre, Jacobi, etc.)
- Tensor product and direct sum spaces with boundary conditions
- Assembly of bilinear / linear forms with symbolic (SymPy) coefficients
- A SymPy-based form-language for describing PDEs
- Curvilinear coordinates
- JAX-backed forward/backward transforms and differentiation
- Utilities for sparse conversion, preconditioning, and projection
- A friendly interface for experimenting with PINNs

## Installation

Using uv (recommended):

```bash
pip install uv  # if not already installed
uv add jaxfun   # when published
```

From source:

```bash
git clone https://github.com/spectralDNS/jaxfun.git
cd jaxfun
uv sync
```

## Quickstart

### Galerkin method

```python
from jaxfun.galerkin import Chebyshev, TensorProduct, TestFunction, TrialFunction
from jaxfun.galerkin.inner import inner
from jaxfun.operators import Div, Grad

C = Chebyshev.Chebyshev(16)
T = TensorProduct((C, C))
v = TestFunction(T)
u = TrialFunction(T)
A = inner(Div(Grad(u)) * v)
```

### Multilayer Perceptron

Use a simple multilayer perceptron neural network and solve Poisson's equation on the unit square

```python
from jaxfun.pinns import FlaxFunction, Loss, MLPSpace, Trainer, UnitSquare, adam, lbfgs
from jaxfun.operators import Div, Grad

# Create an MLP neural network space with two hidden layers
V = MLPSpace([12, 12], dims=2, rank=0, name="V")
u = FlaxFunction(V, name="u") # The trial function, which here is a neural network

# Get mesh points on and inside the unit square
N = 50
mesh = UnitSquare()
xyi = mesh.get_points_inside_domain(N, N, "uniform")
xyb = mesh.get_points_on_domain(N, N, "uniform")

# Define Poisson's equation: residual = △u - 2
residual = Div(Grad(u)) - 2

# Define loss function based on Poisson's equation, including
# homogeneous Dirichlet boundary conditions, and train model
loss_fn = Loss((residual, xyi), (u, xyb))
trainer = Trainer(loss_fn)
trainer.train(adam(u), 5000)
trainer.train(lbfgs(u), 5000)
```

See the [`examples`](examples/) directory for more patterns.

## Development

Run tests (excluding slow):

```bash
uv run pytest
```

Run full (including slow demos):

```bash
uv run pytest -m "slow or not slow"
```

Lint & format:

```bash
uv run pre-commit run --all-files
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

BSD 2-Clause – see [LICENSE](LICENSE).

## Authors

- Mikael Mortensen: mikaem@math.uio.no
- August Femtehjell: august.femtehjell@uio.no
