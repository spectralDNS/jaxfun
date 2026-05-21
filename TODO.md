# Temporal Integration Cleanup

Persistent checklist for simplifying temporal integration after the matrix interface refactor. Keep the original task text in place; append implementation notes under the relevant sections as decisions are made.

## 1. Normalize Assembled Operators

- [x] Add a canonical operator wrapper in `src/jaxfun/utils/operator_tools.py`, tentatively named `LinearTerm`, returned by `assemble_linear_term`.
- [x] Replace parallel `operator` / `diagonal` handling with term methods: `apply(u)`, `solve(rhs)`, `todense(state_size)`, `diagonal_or_none()`, and `is_zero`.
- [x] Keep old attributes temporarily as compatibility aliases where tests currently inspect `mass_operator`, `mass_diag`, `linear_operator`, and `linear_diag`.

Design notes:

- Started by preserving the compatibility surface while moving the normalization point into `operator_tools.py`; this lets integrator changes stay incremental.
- Chose wrapper-level diagonal application and solve helpers that accept either coefficient-shaped diagonals or flattened global diagonals, so the integrators do not need to reason about tensor-product shape details.
- `BaseIntegrator` now rebuilds compatibility aliases from the normalized terms, which keeps current tests readable while making the term wrapper the source of behavior.

## 2. Use Matrix Interface For Solves

- [x] Update `solve_operator` to call matrix-native `.solve(rhs)` for `Matrix`, `DiaMatrix`, `TPMatrix`, `TPMatrices`, and `TensorMatrix`.
- [x] Use dense fallback only for raw arrays or unsupported list combinations.
- [x] Preserve cached LU and tensor-product solver paths where the matrix interface already provides them.

Design notes:

- Matrix-native solves now dispatch before dense conversion; dense fallback remains for raw arrays and multi-operator lists where no combined structured solver exists yet.
- `BackwardEuler` now warms a `Matrix` LU factorization in setup for the dense fallback system, so repeated implicit steps no longer call `jnp.linalg.solve` directly inside the timestep loop.
- Focused tests showed `DiaMatrix.solve` must not create its LU cache for the first time inside the jitted RK4 RHS path, because the singularity check requires a concrete value. The fix is to warm solve caches for reusable mass operators during integrator construction.
- Added mass-operator solve-cache warming in `BaseIntegrator` for non-diagonal mass terms; focused integrator tests now pass with native solves enabled.

## 3. Simplify BaseIntegrator

- [x] Store `self.mass_term` and `self.linear_term` as normalized terms.
- [x] Update `apply_mass`, `apply_mass_inverse`, `linear_rhs`, `mass_matrix_dense`, and `linear_matrix_dense` to delegate to the normalized terms.
- [x] Keep current public attributes as aliases during this cleanup pass.

Design notes:

- Kept the historical identity fallback for a missing mass operator in `apply_mass` and `apply_mass_inverse`, while all non-empty mass and linear operations now route through `LinearTerm`.

## 4. Simplify BackwardEuler

- [x] Build and cache one implicit system term for `M - dt L`.
- [x] Use diagonal solve only when both mass and linear terms are diagonal.
- [x] Otherwise solve through the wrapper rather than direct `jnp.linalg.solve`.

Design notes:

- Represented the implicit system as a `LinearTerm`, using a diagonal-only term for the fast path and a `Matrix` term with a warmed LU cache for the dense fallback path.

## 5. Split ETDRK4 Setup Paths

- [x] Extract `_setup_diagonal_etd(dt)` from `ETDRK4.setup`.
- [x] Extract `_setup_dense_etd(dt)` from `ETDRK4.setup`.
- [x] Preserve stored fields `E`, `E2`, `Q`, `f1`, `f2`, `f3`, and `is_diag`.
- [x] Preserve the current diagonal FFT path and no-`dot_general` behavior.

Design notes:

- Split setup without changing the stage application code; the diagonal backend still stores elementwise arrays and the dense backend still stores global matrices.
- Treated a missing mass operator as identity inside ETD setup, matching the existing runtime mass-apply fallback while keeping explicit mass operators unchanged.

## 6. Tests And Performance Checks

- [x] Run focused integrator tests: `uv run pytest tests/integrators/test_integrator_skeleton.py tests/integrators/test_backward_euler.py tests/integrators/test_etdrk4.py tests/integrators/test_nls1d_etdrk4.py`.
- [x] Run matrix solver regression tests: `uv run pytest tests/la/test_diamatrix.py tests/la/test_kron.py tests/la/test_tpmatrices_solvers.py`.
- [x] Confirm ETDRK4 diagonal FFT tests still assert no `dot_general`.
- [x] If failures occur, append the cause and design adjustment under the relevant TODO section before patching.

Design notes:

- Focused integrator suite passed after cache warming: 25 tests passed.
- Matrix solver regression suite passed: 380 tests passed and 2 skipped.
- The focused ETDRK4 suite includes the diagonal FFT path checks that assert no `dot_general`, and those remained green.
- Ruff passed on the touched source files after simplifying the ETDRK4 setup branch expression.
- Re-ran the focused integrator suite after the final source edit: 25 tests passed.

## 7. Move Operator Semantics Into Matrix Classes

- [x] Add a matrix-level pure-diagonal API, starting with `DiaMatrix.diagonal_or_none()` based on whether the matrix stores only the main diagonal.
- [x] Add the same pure-diagonal API to `Matrix` so dense matrices do not expose unsafe diagonal shortcuts for non-diagonal data.
- [x] Update `operator_tools.operator_diagonal()` to ask matrix objects for their own diagonal representation before falling back to external inspection.
- [x] Extend tensor-product operator classes with compatible diagonal helpers so tensor-product diagonal detection does not live in temporal integration helpers.
- [ ] Add explicit zero and identity operator types, or equivalent matrix-level representations, so `None` is no longer overloaded as both zero and identity depending on integrator context.
- [ ] Replace `LinearTerm` with plain `(operator, forcing)` once operators consistently implement apply, solve, dense conversion, diagonal detection, zero/identity semantics, and structured linear combinations.

Design notes:

- The intended direction is to make every assembled operator a uniform matrix-like object. `LinearTerm` should become temporary scaffolding, not the final abstraction.
- Diagonal fast paths belong in the matrix layer: `DiaMatrix` can decide pure diagonal status from its stored offsets, while dense `Matrix` must check whether off-diagonal entries are actually zero.
- Implemented `is_diagonal` and `diagonal_or_none()` on `DiaMatrix` and `Matrix`. `DiaMatrix` only reports diagonal for square matrices with offsets exactly `(0,)`; a single shifted diagonal is intentionally not treated as elementwise.
- `operator_tools.operator_diagonal()` now delegates to the matrix-level API for `DiaMatrix` and `Matrix`, reducing duplicated diagonal policy in the temporal helper layer.
- Added direct matrix-layer regression coverage for `DiaMatrix.diagonal_or_none()` and `Matrix.diagonal_or_none()`, including the shifted single-diagonal case.
- Verified this first matrix-layer migration with DIA, tensor-product, and focused integrator tests.
- Moved tensor-product diagonal construction into `TPMatrix.diagonal_or_none()` and `TPMatrices.diagonal_or_none()`, including the TPMatrix scale factor. `operator_tools` now delegates to matrix-level diagonal APIs for dense, DIA, and tensor-product operators.
- Added tensor-product diagonal regression coverage, including scale propagation and rejection of shifted diagonal factors.
- Re-ran style, full matrix regressions, and focused integrator tests after moving tensor-product diagonal logic: all passed.
