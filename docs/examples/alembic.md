---
title: "Alembic Migrations Example"
description: "Alembic-managed DDL for Metaxy feature tables on DuckLake."
---

# Alembic Migrations

## Overview

::: metaxy-example source-link
    example: alembic

Alembic-managed DDL for Metaxy feature tables on a [DuckLake](https://ducklake.select/)-attached DuckDB with a local SQLite catalog.

## Alembic vs auto_create_tables

| | `auto_create_tables` | Alembic |
|---|---|---|
| **Schema control** | Inferred from DataFrame on first write | Explicit SQLModel definitions, reviewed migration scripts |
| **Schema evolution** | Drop and recreate | `ALTER TABLE` via versioned scripts |
| **Rollback** | Not possible | `alembic downgrade` |
| **Best for** | Development | Production with DDL governance |

## Provenance columns: dynamic types

Metaxy's `metaxy_provenance_by_field` and `metaxy_data_version_by_field` columns are dynamic maps whose keys change when feature fields are added or removed. `STRUCT` with fixed keys would require a migration on every field change and silently drop historical provenance when a field is removed.

The [SQLModel integration](../integrations/plugins/sqlmodel.md) declares these columns as `JSON` for cross-backend portability. This example overrides them with DuckDB 1.5's [`VARIANT`](https://duckdb.org/2025/06/13/announcing-duckdb-150.html) type in the generated migration. `VARIANT` is a native semi-structured type that stores any DuckDB value without schema constraints — the preferred choice for DuckDB-only deployments.

This example pins `duckdb>=1.5.0`. The `env.py` monkey-patches `duckdb-engine`'s column reflection to work around a missing `pg_collation` table ([Mause/duckdb_engine#1391](https://github.com/Mause/duckdb_engine/issues/1391)), so `alembic revision --autogenerate` works on 1.5.

## DuckLake adaptation

Like many OLAP systems, DuckLake does not support [`PRIMARY KEY`, `UNIQUE`](https://ducklake.select/docs/stable/duckdb/advanced_features/constraints), or `RETURNING`. The `env.py` handles all three through Alembic's extension points:

- **Constraints** are stripped from the target metadata before autogenerate runs.
- **`version_table_pk=False`** in `context.configure()` creates `alembic_version` without a primary key.
- **`insert_returning = False`** on the dialect is set in a custom `AlembicDuckDBImpl`.

DuckLake tables live in an attached catalog. A SQLAlchemy `connect` event hook runs the DuckLake initialization SQL (extension load, secret creation, `ATTACH`, `USE`) on every new connection so that Alembic's introspection and DDL target the correct catalog. DuckLake internal tables (`ducklake_*`) are excluded from reflection via `include_name`.

## Partitioning and sorting

DuckLake does not support `PARTITION BY` or `ORDER BY` clauses in `CREATE TABLE`. These are applied via `ALTER TABLE` after table creation. The `AlembicDuckDBImpl.create_table()` hook in `env.py` emits these statements automatically after each table creation:

- [Partitioning](https://ducklake.select/docs/stable/duckdb/advanced_features/partitioning) by `metaxy_feature_version` — queries for a specific version only scan relevant files.
- [Sorted/clustered tables](https://ducklake.select/docs/stable/duckdb/advanced_features/sorted_tables) (DuckLake 0.4) by `metaxy_updated_at` — co-locates rows within each Parquet file for better compression and range scans.

Both only affect new data. Existing files keep their original layout, so it is safe to add or change these in later migrations.

!!! tip "Inline partitioning"

    A future DuckDB release is expected to support `PARTITIONED BY` and `SORTED BY` directly in `CREATE TABLE` for DuckLake catalogs ([duckdb/duckdb#20431](https://github.com/duckdb/duckdb/pull/20431)). Once available, the `AlembicDuckDBImpl.create_table()` hook can be replaced with native DDL.

## Generated DDL

Running `alembic upgrade head --sql` produces:

```sql
CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL
);

CREATE TABLE examples__alembic_demo (
    metaxy_provenance_by_field VARIANT NOT NULL,
    metaxy_provenance VARCHAR NOT NULL,
    metaxy_feature_version VARCHAR NOT NULL,
    metaxy_project_version VARCHAR NOT NULL,
    metaxy_data_version_by_field VARIANT NOT NULL,
    metaxy_data_version VARCHAR NOT NULL,
    metaxy_created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metaxy_materialization_id VARCHAR,
    metaxy_updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metaxy_deleted_at TIMESTAMP WITH TIME ZONE,
    sample_uid VARCHAR NOT NULL,
    path VARCHAR NOT NULL
);

ALTER TABLE examples__alembic_demo
    SET PARTITIONED BY (metaxy_feature_version);

ALTER TABLE examples__alembic_demo
    SET SORTED BY (metaxy_updated_at);
```

## Getting Started

```shell
uv sync
```

## Step 1: Define features

```python title="src/example_alembic/definitions.py"
--8<-- "example-alembic/src/example_alembic/definitions.py"
```

## Step 2: Configure Metaxy

`auto_create_tables = false` — tables must exist before the pipeline writes.

```toml title="metaxy.toml"
--8<-- "example-alembic/metaxy.toml"
```

## Step 3: Wire Alembic

```python title="alembic/env.py"
--8<-- "example-alembic/alembic/env.py"
```

## Step 4: Generate and apply migrations

```shell
cd examples/example-alembic
uv run alembic revision --autogenerate -m "create feature tables"
```

Preview the DDL before applying:

```shell
uv run alembic upgrade head --sql
```

!!! danger "Review before applying"

    Always inspect the generated DDL before running `upgrade`. Verify that VARIANT columns, partitioning, and sorted table statements are correct.

```shell
uv run alembic upgrade head
```

## Step 5: Run the pipeline

```python title="src/example_alembic/pipeline.py"
--8<-- "example-alembic/src/example_alembic/pipeline.py"
```

::: metaxy-example output
    example: alembic
    scenario: "Alembic migration and pipeline run"
    step: "run_pipeline"

## Related

- [SQLModel integration](../integrations/plugins/sqlmodel.md)
- [SQLAlchemy integration](../integrations/plugins/sqlalchemy.md)
- [DuckLake example](ducklake.md)
- [DuckDB integration](../integrations/metadata-stores/databases/duckdb.md)
