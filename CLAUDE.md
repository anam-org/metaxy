# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Always use the following agents:

Start each task with @agent-planner.

Use @agent-python-dev to write code.
Use @agent-python-test-engineer to write tests for it.

Iterate until @agent-qa is happy.

## Project Status

**⚠️ Early Development - No Stable API**

This project is in active development with no users yet. Breaking changes are expected and encouraged:

- **No backward compatibility required** - **we have no users yet**, refactor freely, prioritize changes that improve design and functionality
- **API changes allowed** - improve interfaces without deprecation warnings
- **Breaking changes welcome** - prioritize better design over stability
- **Move fast** - optimize for the best long-term architecture, not short-term compatibility

## Project Overview

Metaxy is a feature metadata management system for multimodal ML pipelines that tracks feature versions, dependencies, and data lineage. It enables:

- **Declarative feature definitions** with explicit dependencies and versioning
- **Automatic change detection** to identify affected downstream features
- **Smart migrations** to reconcile metadata when code refactors don't change computation
- **Dependency-aware updates** with automatic recomputation when upstream dependencies change
- **Immutable metadata** with copy-on-write storage preserving historical versions
- **Graph snapshots** recording complete feature graph states in deployment pipelines

## Development

Never create git commits unless asked for explicitly by the user.

You can check GitHub Actions for your PR to see if tests pass.

### Environment Setup

```bash
# Install dependencies (uv is required)
uv sync

# Install with optional dependencies
uv sync --extra ibis
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_migrations.py

# Run specific test
uv run pytest tests/test_migrations.py::test_migration_generation

# Run tests with specific backend
uv run pytest tests/metadata_stores/test_duckdb.py
```

Always keep tests up-to-date and maintainable. Add or update tests as features are added or modified.

### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking

```bash
# Run pyrefly type checker
uv run basedpyright --level error
```

## Docs

See the @docs directory to learn more.
