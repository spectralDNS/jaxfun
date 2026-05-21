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
- Superseded by the matrix-owned operator cleanup in section 7: `BaseIntegrator` now stores normalized matrix-like `mass_operator` and `linear_operator` values directly, while retaining `mass_diag`, `linear_diag`, and `linear_forcing` compatibility attributes.

## 4. Simplify BackwardEuler

- [x] Build and cache one implicit system term for `M - dt L`.
- [x] Use diagonal solve only when both mass and linear terms are diagonal.
- [x] Otherwise solve through the wrapper rather than direct `jnp.linalg.solve`.

Design notes:

- Represented the implicit system as a `LinearTerm`, using a diagonal-only term for the fast path and a `Matrix` term with a warmed LU cache for the dense fallback path.
- Superseded by the matrix-owned operator cleanup in section 7: `BackwardEuler` now stores the structured `M - dt L` operator directly and solves it through the shared operator helper, with diagonal fast paths owned by the matrix classes.

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
- [x] Investigate high per-process memory usage in the focused integrator and matrix test files before continuing larger implementation work.

Design notes:

- Focused integrator suite passed after cache warming: 25 tests passed.
- Matrix solver regression suite passed: 393 tests passed and 2 skipped.
- The focused ETDRK4 suite includes the diagonal FFT path checks that assert no `dot_general`, and those remained green.
- Ruff passed on the touched source files after simplifying the ETDRK4 setup branch expression.
- Re-ran the focused integrator suite after the final source edit: 25 tests passed.
- After removing `LinearTerm`, the focused integrator suite initially failed because diagonal mass solves reached first-time LU factorization inside jitted RK4 paths. Moved pure-diagonal solve shortcuts into `DiaMatrix`, cached-diagonal `Matrix`, and tensor-product operator `solve()` methods so diagonal systems avoid LU without restoring the wrapper.
- Re-ran focused operator tests after moving diagonal solves into matrix classes: 8 tests passed.
- Re-ran the focused integrator suite after removing `LinearTerm`: 25 tests passed.
- Re-ran the full matrix solver regression suite after removing `LinearTerm` and adding matrix-level diagonal solves: 393 tests passed and 2 skipped.
- Final focused integrator pass after removing the stale `_state_size` field: 25 tests passed.
- Sequential `-n0` memory probing showed the largest single-process peaks are `tests/la/test_diamatrix.py` at about 1.59 GB, `tests/la/test_tpmatrices_solvers.py` at about 1.55 GB, and `tests/la/test_kron.py` at about 1.30 GB. The focused integrator files are smaller, between about 0.43 GB and 0.86 GB.
- Class/section-level probing showed the heavy regions are repeated JAX solve/matvec/LU coverage rather than one large allocation: `DiaMatrix.TestLU` peaked around 0.99 GB, `DiaMatrix.TestSolve` around 0.82 GB, tensor-product dense LU around 1.06 GB, and tensor-product dispatch/L2/3D coverage around 1.19 GB.
- JAX compile logging confirmed the pressure is largely many shape-specialized XLA traces/executables retained in each worker process. `DiaMatrix.TestLU` emitted 545 compile log entries, while the tensor-product dispatch/L2/3D section emitted 747. Running with JIT disabled increased memory for representative heavy groups, so the issue is not caused by one explicit jitted kernel alone; eager JAX still dispatches many compiled primitive programs.
- The global pytest default `-n logical` multiplies this per-worker JAX/XLA footprint. On the local run that created eight workers, several workers can each accumulate around 1.5-2 GB depending on which matrix/integrator sections they receive.

## 7. Move Operator Semantics Into Matrix Classes

- [x] Add a matrix-level pure-diagonal API, starting with `DiaMatrix.diagonal_or_none()` based on whether the matrix stores only the main diagonal.
- [x] Add the same pure-diagonal API to `Matrix` so dense matrices do not expose unsafe diagonal shortcuts for non-diagonal data.
- [x] Update `operator_tools.operator_diagonal()` to ask matrix objects for their own diagonal representation before falling back to external inspection.
- [x] Extend tensor-product operator classes with compatible diagonal helpers so tensor-product diagonal detection does not live in temporal integration helpers.
- [x] Add explicit zero and identity operator types, or equivalent matrix-level representations, so `None` is no longer overloaded as both zero and identity depending on integrator context.
- [x] Add structured linear-combination support so systems like `M - dt * L` can preserve DIA, dense, or tensor-product structure before falling back to generic sums.
- [x] Replace `LinearTerm` with plain `(operator, forcing)` once operators consistently implement apply, solve, dense conversion, diagonal detection, zero/identity semantics, and structured linear combinations.

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
- Added `IdentityMatrix`, `ZeroMatrix`, `OperatorSum`, and `linear_combination` in `jaxfun.la`. `BaseIntegrator` now normalizes missing mass to identity and missing linear operators to zero, leaving `None` for absence from assembly rather than runtime operator semantics.
- `BackwardEuler` now builds its implicit system through `linear_combination((M, -dt * L))`, which preserves DIA, dense, and tensor-product structure where possible and uses `OperatorSum` as the generic fallback.
- Removed `LinearTerm` from the integrator runtime path. `assemble_linear_term()` now returns plain `(operator, forcing)`, and `BaseIntegrator`, `BackwardEuler`, and `ETDRK4` delegate directly to operator helpers backed by matrix methods.
- Added matrix-owned diagonal solve paths for `DiaMatrix`, cached-diagonal `Matrix`, `TPMatrix`, and `TPMatrices`. This keeps the performance-sensitive diagonal behavior with the matrix types and avoids first-time LU work inside jitted temporal integration paths.
- Superseded the `OperatorSum` and `linear_combination()` approach in section 8. Concrete matrix dunder methods now own the expected algebra and promotion behavior.

## 8. Prefer Matrix Dunder Algebra Over Combination Factories

- [x] Add cross-type matrix addition/subtraction promotion so `DiaMatrix + Matrix` returns `Matrix`, identity plus DIA returns `DiaMatrix`, identity plus dense returns `Matrix`, and zero plus any compatible operator returns the other operator.
- [x] Add tensor-product addition/subtraction dunders so `TPMatrix + TPMatrix`, `TPMatrices + TPMatrix`, and `TPMatrices + TPMatrices` return `TPMatrices`.
- [x] Replace temporal integration use of `linear_combination()` with direct algebra, e.g. `self.mass_operator - dt * self.linear_operator`.
- [x] Remove `OperatorSum`, `linear_combination()`, and related tests once concrete matrix promotion covers the temporal integration cases.
- [x] Keep unsupported mixed dimensional families explicit: raise or return `NotImplemented` instead of silently densifying tensor-product or tensor operators.

Design notes:

- `OperatorSum` was broader than the temporal integration problem requires. The assembly layer already returns one concrete dimensional family per operator, so `M - dt * L` should be expressible through matrix dunder methods and should promote only to the broadest concrete matrix class needed.
- Removed `OperatorSum`, `linear_combination()`, and `scale_operator()` from the public LA package. `BackwardEuler.setup()` now builds the implicit system as `self.mass_operator - dt * self.linear_operator`.
- Added concrete promotions: `IdentityMatrix` with DIA stays DIA, identity with dense promotes to dense `Matrix`, mixed DIA/dense promotes to `Matrix`, and zero operators defer to the non-zero operand. Tensor-product additions stay in `TPMatrices`, while non-separable tensor additions stay in `TensorMatrix`.
- Unsupported mixed dimensional families are not silently densified; the relevant dunder methods return `NotImplemented`, leaving Python to raise a clear unsupported operand error.
- Direct `Matrix.solve()` and `DiaMatrix.solve()` now handle coefficient-shaped RHS values against flat diagonal operators, matching the global-state solve behavior previously handled in temporal helpers.
- Verification after this pass: operator tests passed with 10 tests, focused integrator tests passed with 25 tests, and matrix regression tests passed with 395 tests and 2 skipped.

## 9. Introduce DiagonalMatrix For Pure Main-Diagonal Operators

- [x] Add `DiagonalMatrix` as a `DiaMatrix` subclass that represents only square main-diagonal operators.
- [x] Route pure main-diagonal construction through `DiagonalMatrix`, including `diags(..., offsets=(0,))` and identity scaling.
- [x] Move coefficient-shaped elementwise `matvec` and `solve` behavior into `DiagonalMatrix` so ETDRK4 can eventually store diagonal ETD coefficients as matrix objects.
- [x] Preserve `DiaMatrix` for genuinely banded/shifted DIA matrices; a single shifted diagonal must not be treated as a diagonal operator.
- [x] Add tests showing `DiagonalMatrix` is still a `DiaMatrix`, applies and solves coefficient-shaped arrays without LU, and preserves diagonal structure under diagonal arithmetic.

Design notes:

- `DiagonalMatrix` is the concrete matrix-layer abstraction needed before removing ETDRK4's separate diagonal coefficient path. It should own the pure elementwise behavior while general `DiaMatrix` continues to represent arbitrary DIA sparsity.
- Implemented `DiagonalMatrix` as a `DiaMatrix` subclass with pure main-diagonal storage, coefficient-shaped `matvec`/`solve`, and diagonal-preserving arithmetic for diagonal operands.
- `diags(..., offsets=(0,))` now constructs `DiagonalMatrix` for square pure-main-diagonal inputs, while shifted single-diagonal matrices remain plain `DiaMatrix` and do not report diagonal fast-path eligibility.
- Kept `lu_factor()` compatible with the existing `LUFactors` API by returning `L` and `U` factors rather than introducing a separate LU factor type.
- Verified the diagonal work with targeted diagonal tests, `tests/la/test_operators.py`, `tests/la/test_diamatrix.py`, the focused integrator suite, and the full matrix regression suite.
- Extended pure-diagonal construction through `DiaMatrix.from_dense()` so `Matrix.tosparse()` and dense diagonal extraction preserve the `DiagonalMatrix` abstraction.
- Added explicit `IdentityMatrix` arithmetic handling in `DiagonalMatrix` to avoid reflected-operation recursion and keep `D + I`, `D - I`, and `I - D` in the diagonal subclass.
- Added `DiagonalMatrix.astype()` so dtype conversion does not widen back to a generic `DiaMatrix`.
- Re-verified this pass with `uv run pytest -n0 tests/la/test_operators.py tests/la/test_diamatrix.py -q`, `uv run pytest -n0 tests/la/test_kron.py tests/la/test_tpmatrices_solvers.py tests/integrators/test_integrator_skeleton.py tests/integrators/test_backward_euler.py tests/integrators/test_etdrk4.py tests/integrators/test_nls1d_etdrk4.py -q`, and `uv run ruff check src/jaxfun/la/diamatrix.py tests/la/test_diamatrix.py tests/la/test_operators.py`.

## 10. Represent ETDRK4 Coefficients As Matrix Objects

- [x] Wrap diagonal ETDRK4 coefficients in `DiagonalMatrix` instead of storing the timestep coefficients as raw arrays.
- [x] Wrap dense ETDRK4 coefficients in `Matrix` so the timestep applies a uniform matrix-like coefficient interface.
- [x] Remove the raw-array `_apply_etd_operator(op, u, is_diag)` branch and apply coefficient matrices directly, flattening only for dense global `Matrix` coefficients.
- [x] Preserve compatibility fields `E`, `E2`, `Q`, `f1`, `f2`, and `f3`, but allow them to be matrix objects.
- [x] Re-run ETDRK4 diagonal FFT tests to confirm the diagonal coefficient path still has no `dot_general`.

Design notes:

- This is the next step after `DiagonalMatrix`: the diagonal ETD path should use the same matrix abstraction as mass and linear operators, while dense coefficients remain global `Matrix` objects that explicitly flatten and restore coefficient-shaped states.
- `ETDRK4.setup()` now stores diagonal coefficients as flat `DiagonalMatrix` objects and dense coefficients as global `Matrix` objects. The public coefficient field names are unchanged.
- `ETDRK4.step()` no longer branches on raw coefficient arrays. It applies coefficient matrix objects directly; only dense global `Matrix` coefficients flatten and reshape the coefficient state.
- The diagonal path still applies coefficient-shaped tensor-product states elementwise through `DiagonalMatrix`, preserving the FFT nonlinear path and avoiding dense basis matmuls.
- Verification after this pass: `tests/integrators/test_etdrk4.py` passed with 10 tests, including the no-`dot_general` assertions; the broader focused matrix/integrator run passed with 425 tests and 2 skipped; ruff passed on touched files.
- Removed the local `apply()` closure from `ETDRK4.step()` after teaching `Matrix.__matmul__` how to apply a square global matrix to coefficient-shaped state arrays by flattening and restoring shape internally.
- The flattening belongs in `Matrix.__matmul__`, not ETDRK4, because dense global operators now share the same ergonomic application surface as `DiagonalMatrix`: timestep code can write `op @ u` regardless of coefficient shape.
- Re-verified after this simplification: focused operator plus ETDRK4 tests passed with 22 tests; the broader focused matrix/integrator run passed with 426 tests and 2 skipped; ruff passed on touched files.
- `_etdrk4_diag_coeffs()` now returns `DiagonalMatrix` coefficient objects directly. The formula tests unwrap `.diagonal()` for numerical comparisons, while the integrator stores the same objects without an additional wrapping helper.
- Fixed `DiagonalMatrix` scaling so `(dt * self.Q) @ n` works inside the jitted ETDRK4 step. The underlying issue was the scaled diagonal object being passed into a nested jitted `DiagonalMatrix.matvec`; `DiagonalMatrix.matvec` is now a plain elementwise method, and scalar dunders are explicit on the subclass.
- Added a focused regression for applying a scaled `DiagonalMatrix` inside `jax.jit`, restored `dtQ = dt * self.Q` in ETDRK4, and re-verified the broader focused matrix/integrator run with 427 tests passed and 2 skipped.
- Simplified ETDRK4 setup so `setup()` queries `mass_operator.diagonal_or_none()` and `linear_operator.diagonal_or_none()` once, passes the precomputed `Ldiag` to the diagonal coefficient path, and passes the optional mass diagonal to the dense path.
- Inlined `_setup_diagonal_etd()` after it became a one-line wrapper around `_etdrk4_diag_coeffs()`. Kept `_setup_dense_etd()` because it still owns the dense mass handling and matrix-function construction.
- Corrected the diagonal setup ownership: `_setup_diagonal_etd()` stays as the named ETDRK4 setup path, and the former module-level `_etdrk4_diag_coeffs()` body is inlined into that method.

## 11. Remove Runtime Operator Dispatch Helpers

- [x] Confirm `inner(..., return_all_items=False)` collapses bilinear lists before temporal integration sees them.
- [x] Remove `list[GalerkinOperator]` from the integrator-facing assembled operator type.
- [x] Replace `apply_operator(op, u)` call sites with `op @ u`.
- [x] Replace `solve_operator(op, rhs)` call sites with `op.solve(rhs)`.
- [x] Replace `operator_to_dense(op)` call sites with `op.todense()`.
- [x] Keep `operator_tools.py` focused on assembly normalization rather than runtime operator behavior.

Design notes:

- Checked `inner.process_results()`: raw lists are returned only when `return_all_items=True`. `assemble_linear_term()` calls `inner()` with the default collapsed path, so temporal integration does not need to support list-shaped operators.
- `operator_tools.py` now only normalizes assembled output into `(operator, forcing)`, converting raw 2-D arrays to `Matrix` and rejecting raw operator lists with a message pointing to `return_all_items=True`.
- `BaseIntegrator`, `BackwardEuler`, and `ETDRK4` now use the matrix interface directly: `op @ u`, `op.solve(rhs)`, `op.todense()`, and `op.diagonal_or_none()`.
- Moved the coefficient-shaped RHS fallback into `Matrix.solve()` and `DiaMatrix.solve()`, which was the one meaningful behavior previously hidden in `solve_operator()`.
- Added `is_zero`/`diagonal_or_none()` coverage on the concrete operator classes needed by the direct integrator calls, leaving zero/diagonal semantics on the operator objects themselves.
- Verification after this pass: `uv run ty check`, ruff on touched files, and the focused matrix/integrator suite all passed.
