# Metaxy

## Overview

**Metaxy** is a declarative metadata management system for multi-modal data and machine learning pipelines. Metaxy allows statically defining graphs of features with versioned **fields** -- logical components like `audio`, `frames` for `.mp4` files and **columns** for feature metadata stored in Metaxy's metadata store. With this in place, Metaxy provides:

- **Sample-level data versioning**: Track field and column lineage, compute versions as hashes of upstream versions for each sample
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
