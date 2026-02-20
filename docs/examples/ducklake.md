---
title: "DuckLake Example"
description: "Example of configuring DuckLake as an open lakehouse format for the DuckDB metadata store."
---

# DuckLake

## Overview

::: metaxy-example source-link
    example: ducklake

This example demonstrates how to configure [DuckLake](https://ducklake.select/) with the DuckDB metadata store.
DuckLake is an open lakehouse format that separates the metadata catalog (table definitions, schema evolution, and transaction history) from data file storage.
This lets you choose independent backends for each layer, for example PostgreSQL for the catalog and S3 for data files.

We will set up a DuckLake-backed store via `metaxy.toml` and preview the SQL statements that DuckLake would execute when attaching to a DuckDB connection.

## Getting Started

Install the example's dependencies:

```shell
uv sync
```

## Configuration

DuckLake is configured in `metaxy.toml` with two parts: a **metadata backend** (where the catalog is stored) and a **storage backend** (where data files live).

The active configuration below uses SQLite for the metadata catalog and the local filesystem for data storage. Commented-out sections show alternative backends.

```toml title="metaxy.toml"
--8<-- "example-ducklake/metaxy.toml"
```

Available backend combinations:

| Metadata backend | Storage backend |
|---|---|
| DuckDB, SQLite, PostgreSQL | local filesystem, S3, Cloudflare R2, Google Cloud Storage |
| MotherDuck | managed (no storage backend needed), or BYOB with S3/R2/GCS |

!!! tip

    To use the credential chain (IAM roles, environment variables, etc.) instead of static credentials, set `secret_parameters = { provider = "credential_chain" }` on S3, R2, or GCS storage backends.

!!! note

    MotherDuck supports a "Bring Your Own Bucket" (BYOB) mode where MotherDuck manages the DuckLake catalog while you provide your own S3-compatible storage. Storage secrets are created `IN MOTHERDUCK` so that MotherDuck compute can access your bucket.

## Walkthrough

The demo script initializes the store from configuration and previews the SQL statements that would be executed:

```python title="src/example_ducklake/demo.py"
--8<-- "example-ducklake/src/example_ducklake/demo.py"
```

Run the demo:

```shell
uv run python src/example_ducklake/demo.py
```

The output shows the full sequence of SQL statements: creating secrets for the metadata and storage backends, and attaching the DuckLake database.

## Related Materials

- [DuckDB Metadata Store](../integrations/metadata-stores/databases/duckdb.md)
- [`DuckLakeAttachmentConfig`][metaxy.ext.metadata_stores.ducklake.DuckLakeAttachmentConfig]
