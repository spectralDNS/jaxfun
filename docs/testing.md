# Testing

`jaxfun` uses pytest markers to keep local feedback fast while preserving
slower numerical and end-to-end confidence checks.

## Common Commands

```bash
# Fast local tests
uv run pytest

# Fast tests in x64 mode
uv run pytest --float64

# Smoke tests
uv run pytest -m smoke

# Slow tests
uv run pytest -m slow

# Integration tests
uv run pytest -m integration

# PINN tests
uv run pytest -m pinn

# SPMD tests
uv run pytest --num-devices=2 -m spmd

# Full local suite except GPU-only tests
uv run pytest -m "not gpu"
```

## Marker Guide

Use no marker for small unit tests that should run by default.

Use `smoke` for a small end-to-end test that exercises a realistic user-facing
flow and is cheap enough for every pull request.

Use `integration` for tests that solve a complete problem or combine several
subsystems. Add `slow` as well when the test is long-running.

Use `examples` with `slow` for full example-script execution.

Use `pinn` for neural-network and PINN tests. Add `slow` to longer training
loops or expensive model checks.

Use `spmd` for tests that require more than one JAX device. Run these with
`--num-devices=2` or a larger device count.

Use `gpu` for tests that require a GPU backend.

Use `serial` for tests that must not run under pytest-xdist. Run serial tests
with `-n0` unless the test has its own serialization mechanism.

## CI Tiers

Pull requests and branch pushes run fast default tests, fast x64 tests, and the
smoke suite.

Pushes to `main`, manual workflow runs, and the nightly schedule also run
integration, slow, example, PINN, SPMD, and SPMD x64 jobs. The jobs are split
so failures point at the relevant test tier and slow tests are not duplicated
across unrelated jobs.
