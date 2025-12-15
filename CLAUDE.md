# CLAUDE.md

## Agent Workflow

Start each task with the planner agent. Use python-dev to write code, python-test-engineer to write tests. Iterate until qa is satisfied.

## Project Status

**Early Development - No Stable API**

No users yet. Breaking changes are encouraged. Prioritize better design over backward compatibility. Refactor freely.

## Project Overview

Metaxy is a feature metadata management system for multimodal ML pipelines. It tracks feature versions, dependencies, and data lineage with declarative definitions, automatic change detection, and smart migrations.

## Commands

```bash
just sync        # Install all dependencies
just ruff        # Lint and format
just typecheck   # Type check with ty
uv run pytest    # Run tests (add -k "pattern" to filter)
```

## Guardrails

**Git commits**: Do not create commits unless explicitly requested. When asked to commit, follow standard git workflow.

**Tests**:

- Always add or update tests when modifying features. Run `uv run pytest tests/path/to/test.py` to verify changes before considering work complete.
- Never run the full test suite unless instructed. Only run relevant tests by filtering with `-k "pattern"`, specific files or specific tests `uv run pytest tests/test_migrations.py::test_migration_generation`

**Type safety**: Code must pass `just typecheck`. Fix type errors before submitting.

**Snapshot tests**: If tests fail due to snapshot mismatches after intentional changes, update with `uv run pytest --snapshot-update path/to/test.py`.

## When to Consult Docs

- **Architecture decisions or unfamiliar patterns**: See `docs/guide/` for concepts like feature graphs, migrations, and metadata stores
- **Backend-specific issues** (DuckDB, ClickHouse, BigQuery): Check `docs/integrations/metadata-stores/` and corresponding test files in `tests/metadata_stores/`
