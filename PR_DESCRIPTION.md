## Summary

This PR adds comprehensive docstrings and documentation improvements across the entire jaxfun codebase, covering core modules for differential operators, coordinate systems, Galerkin methods, and PINNs functionality. The changes improve API documentation, add parameter descriptions, return type specifications, and practical examples to make the library more accessible to users and contributors.

## Types of Changes

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [x] Documentation update
- [ ] Refactor / cleanup
- [ ] Build / CI
- [ ] Other (describe):

## Related Issues

Refs # (if applicable)

## Description of Changes

This PR enhances documentation throughout the codebase with:

### Core Operators (`src/jaxfun/operators.py`)
- Reformatted docstrings for `outer`, `cross`, `dot`, `grad`, `div`, `curl` operators
- Added detailed Args/Returns/Raises sections following Google style
- Included mathematical notation and formulas for curvilinear coordinates
- Improved examples with consistent formatting
- Added implementation notes explaining tensor calculus approach

### Coordinate Systems (`src/jaxfun/coordinates.py`)
- Enhanced `CoordSys` class documentation
- Added detailed parameter descriptions for coordinate transformations
- Documented basis vector and metric tensor methods
- Included examples for common coordinate systems (cylindrical, spherical, etc.)

### Galerkin Method Modules
Enhanced documentation for all polynomial basis families:
- **Chebyshev** (`src/jaxfun/galerkin/Chebyshev.py`): Added detailed docstrings for evaluation methods (`evaluate2`, `evaluate3`), quadrature points, basis function evaluation, and norms
- **Legendre** (`src/jaxfun/galerkin/Legendre.py`): Documented Gauss-Legendre quadrature and evaluation kernels
- **Fourier** (`src/jaxfun/galerkin/Fourier.py`): Added FFT-based transform documentation
- **Jacobi** (`src/jaxfun/galerkin/Jacobi.py`): Documented general Jacobi polynomial framework
- **Function spaces** (`functionspace.py`, `composite.py`, `tensorproductspace.py`): Added comprehensive class and method docstrings
- **Inner products** (`inner.py`): Documented bilinear form assembly
- **Forms** (`forms.py`): Added detailed documentation for weak form utilities
- **Arguments** (`arguments.py`): Documented trial/test function classes

### PINNs Module (`src/jaxfun/pinns/`)
- **Mesh** (`mesh.py`): Added detailed docstrings for domain classes (UnitLine, Line, UnitSquare, Rectangle, Annulus) with sampling methods and weight computations
- **Boundary conditions** (`bcs.py`): Documented DirichletBC class with usage examples
- **Loss functions** (`loss.py`): Added documentation for LSQR class and residual computation
- **Neural network spaces** (`nnspaces.py`): Enhanced MLPSpace and PirateSpace documentation
- **Modules** (`module.py`): Documented FlaxFunction and Comp classes with parameter details
- **Embeddings** (`embeddings.py`): Added docstrings for coordinate embedding layers
- **Optimizers** (`optimizer.py`): Documented adam, lbfgs, GaussNewton optimizers with usage notes

### Utility Modules (`src/jaxfun/basespace.py`)
- Added docstrings for base space classes and utilities

### Test Coverage (`tests/operators/test_basic_tensors.py`)
- Added new test cases for tensor operations

## Screenshots / Logs (if applicable)

N/A - Documentation only changes

## Breaking Changes

- [x] No
- [ ] Yes (describe below)

No breaking changes. This is purely additive documentation.

## Checklist

- [x] Tests added or updated
- [x] Documentation updated (README, examples, docstrings)
- [x] Added type hints
- [ ] Linting & formatting pass locally (`pre-commit run --all-files`)
- [ ] All new and existing tests pass (`uv run pytest`)
- [ ] Coverage does not decrease
- [ ] Git history is clean (squash/fixup before merge)

## Additional Notes

**Documentation Style**: The docstrings follow a hybrid Google/NumPy style with clear sections for Args, Returns, Raises, and Examples. Mathematical notation is included where relevant for clarity.

**Scope**: This PR touches 22 files with approximately 2,689 insertions and 842 deletions, primarily adding docstrings without changing functionality.

**Testing**: Added test cases in `tests/operators/test_basic_tensors.py` to ensure operators work correctly with enhanced documentation.

**Next Steps**: Contributors should follow this documentation style for future additions. Consider running a documentation linter or doc coverage tool in CI to maintain quality.
