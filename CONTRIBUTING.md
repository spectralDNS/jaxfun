# Contributing to jaxfun

Thanks for your interest in contributing! This guide helps you get started.

## Ways to Contribute

- Report bugs via GitHub Issues (use the Bug Report template)
- Propose features or enhancements (Feature Request template)
- Improve documentation or examples
- Add tests or improve coverage
- Optimize performance or fix bugs

## Development Environment

We use [uv](https://github.com/astral-sh/uv) for dependency management.

Clone and install:

```bash
git clone https://github.com/spectralDNS/jaxfun.git
cd jaxfun
uv sync
uv run pre-commit install
```

Run tests:

```bash
uv run pytest -q
```

Run pre-commit checks locally before pushing:

```bash
uv run pre-commit run --all-files
```

## Branching & Workflow

- Create a descriptive feature branch: `feature/<short-description>` or `fix/<short-description>`
- Keep commits focused; rebase/squash before opening a PR if needed
- Open a Pull Request against `main` using the provided template
- Ensure CI passes (lint, format, tests, coverage)

## Coding Standards

- Follow Ruff for linting & formatting (configured via pre-commit)
- Add type hints
- Keep functions small and focused
- Prefer explicit over implicit

## Tests & Coverage

- Add tests for new features or bug fixes
- Aim for coverage >= existing module average; avoid regressions
- Use parametrization for concise test cases

## Documentation

- Update README or examples for user-facing changes
- Add docstrings for public classes/functions

## Reporting Bugs

Include:

- Steps to reproduce
- Expected vs actual behaviour
- Environment details (OS, Python, JAX, jaxfun commit)
- Minimal code sample if possible

## Release Notes

If your change affects users, add a bullet to the Unreleased section in `CHANGELOG.md` (create it if absent).

## Getting Help

Use GitHub Discussions for questions or design proposals.
