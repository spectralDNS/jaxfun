# jaxfun

Spectral / Galerkin experimentation toolkit built on top of JAX for fast differentiable PDE prototyping, variational forms, and mixed spectral bases.

## Status & Badges

[![Tests](https://github.com/spectralDNS/jaxfun/actions/workflows/pytest.yml/badge.svg)](https://github.com/spectralDNS/jaxfun/actions/workflows/pytest.yml)
[![Linting](https://github.com/spectralDNS/jaxfun/actions/workflows/lint.yml/badge.svg)](https://github.com/spectralDNS/jaxfun/actions/workflows/lint.yml)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Python 3.12](https://img.shields.io/badge/python-3.12+-brightgreen.svg)](pyproject.toml)

> The coverage badge will update once Codecov is fully configured.

## Features

- Orthogonal polynomial and Fourier bases (Chebyshev, Legendre, Jacobi, etc.)
- Tensor product and direct sum spaces with boundary conditions
- Assembly of bilinear / linear forms with symbolic (SymPy) coefficients
- Curvilinear coordinates
- A Sympy-based form-language for describing PDEs 
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

```python
from jaxfun.galerkin import Chebyshev, TensorProduct, TestFunction, TrialFunction, Div, Grad
from jaxfun.galerkin.inner import inner

C = Chebyshev.Chebyshev(16)
T = TensorProduct((C, C))
v = TestFunction(T)
u = TrialFunction(T)
A = inner(Div(Grad(u)) * v)
```

See the [`examples`](examples/) for more patterns.

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
