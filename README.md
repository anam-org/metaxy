# Metaxy

## Overview

**Metaxy** is a declarative metadata management system for multi-modal machine learning pipelines. It provides:

- **Sample-level data versioning**: Track data lineage with hashes computed from upstream dependencies
- **Feature graphs**: Declare features and dependencies between their fields
- **Incremental computation**: Automatically detect which samples need recomputation when upstream fields change
- **Migration system**: When feature code changes without changing outputs (refactoring, graph restructuring), Metaxy can reconcile metadata versions without recomputing expensive features
- **Storage flexibility**: Pluggable backends (DuckDB, ClickHouse, PostgreSQL, SQLite, in-memory) with native SQL optimization where possible
- **Big Metadata**: Metaxy is designed with large-scale distributed systems in mind and can handle large amounts of metadata efficiently.

Metaxy is designed for production data and ML systems where data and features evolve over time, and you need to track what changed, why, and whether expensive recomputation is actually necessary.

## Development

Setting up the environment:

```shell
uv sync --all-extras
uv run prek install
```

## Examples

See [examples](examples/README.md).
