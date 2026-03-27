---
title: "DuckLake Example"
description: "Example of configuring DuckLake as an open lakehouse format for the DuckDB metadata store."
---

# DuckLake

## Overview

::: metaxy-example source-link
    example: ducklake

This example demonstrates how to run a small Metaxy pipeline against a [DuckLake](https://ducklake.select/) attachment on the DuckDB metadata store.
DuckLake is an open lakehouse format that separates the metadata catalog (table definitions, schema evolution, and transaction history) from data file storage.
This lets you choose independent backends for each layer, for example PostgreSQL for the catalog and S3 for data files. The example keeps the workflow Metaxy-native by writing a small feature dataset and then reading it back through the public API.

Available backend combinations:

| Catalog backend | Storage backend |
|---|---|
| DuckDB, SQLite, PostgreSQL | local filesystem, S3, Cloudflare R2, Google Cloud Storage |
| MotherDuck | managed (no storage backend needed), or BYOB with S3/R2/GCS |

!!! tip

    To use the credential chain (IAM roles, environment variables, etc.) instead of static credentials, set `secret_parameters = { provider = "credential_chain" }` on S3, R2, or GCS storage backends.

!!! note

    MotherDuck supports a "Bring Your Own Bucket" (BYOB) mode where MotherDuck manages the DuckLake catalog while you provide your own S3-compatible storage. Storage secrets are created `IN MOTHERDUCK` so that MotherDuck compute can access your bucket.

!!! tip "Recommended: enable [`Map` datatype](../guide/concepts/metadata-stores.md#map-datatype)"

    DuckLake has native `Map` type support. Enabling [`enable_map_datatype`](../reference/configuration.md#metaxy.config.MetaxyConfig.enable_map_datatype) preserves `Map` columns across read and write operations.

## Getting Started

Install the example's dependencies:

```shell
uv sync
```

For the full list of backend combinations and advanced options, see the [DuckLake integration reference](../integrations/metadata-stores/storage/ducklake.md).

## Step 1: Configure DuckLake

DuckLake is configured with two parts:

1. **Catalog backend**: transaction log and metadata
2. **Storage backend**: data files

The example below is intentionally minimal and runnable out of the box.

```toml title="metaxy.toml"
--8<-- "example-ducklake/metaxy.toml"
```

## Step 2: Initial Run

Let's prepare a small Metaxy pipeline using the configured DuckLake metadata store:

```python title="src/example_ducklake/pipeline.py"
--8<-- "example-ducklake/src/example_ducklake/pipeline.py"
```

## Step 3: Inspect Recorded Metadata

::: metaxy-example output
    example: ducklake
    scenario: "DuckLake pipeline run"
    step: "run_demo"

You should see:

1. A successful Metaxy write
2. The physical DuckLake-backed feature table name Metaxy created
3. The rows read back for `examples/ducklake_demo`

## Related Materials

- [DuckLake integration docs](../integrations/metadata-stores/storage/ducklake.md)
- [DuckDB integration docs](../integrations/metadata-stores/databases/duckdb.md)
