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

## Getting Started

Install the example's dependencies:

```shell
uv sync
```

Overview of the next steps:

1. Configure a DuckDB metadata store with a DuckLake attachment.
2. Use SQLite for the catalog and local filesystem storage.
3. Run a demo script that prints the DuckLake setup SQL (`CREATE SECRET`, `ATTACH`, `USE`).

For the full list of backend combinations and advanced options, see the [DuckLake integration reference](../integrations/metadata-stores/storage/ducklake.md).

## Step 1: Configure DuckLake

DuckLake is configured with two parts:

1. **Catalog backend**: transaction log and metadata
2. **Storage backend**: data files

The example below is intentionally minimal and runnable out of the box.

```toml title="metaxy.toml"
--8<-- "example-ducklake/metaxy.toml"
```

## Step 2: Run The Demo

The demo script loads `metaxy.toml`, validates the configured store, and previews the SQL statements that would be executed:

```python title="src/example_ducklake/demo.py"
--8<-- "example-ducklake/src/example_ducklake/demo.py"
```

Run the demo:

```shell
uv run python src/example_ducklake/demo.py
```

## Step 3: Inspect Output

::: metaxy-example output
    example: ducklake
    scenario: "Preview DuckLake SQL setup"
    step: "run_demo"

You should see the configured backend types plus the setup SQL preview (including `ATTACH IF NOT EXISTS ...`).

## Secret Configuration With Templates

For remote backends (for example, PostgreSQL catalog + S3 storage), use `${...}` templates instead of hardcoding credentials in `metaxy.toml`.

```toml
[stores.dev.config.ducklake.catalog]
type = "postgres"
secret_name = "ducklake_pg"
host = "${DUCKLAKE_PG_HOST}"
user = "${DUCKLAKE_PG_USER}"
password = "${DUCKLAKE_PG_PASSWORD}"

[stores.dev.config.ducklake.storage]
type = "s3"
secret_name = "ducklake_s3"
bucket = "${DUCKLAKE_S3_BUCKET}"
key_id = "${DUCKLAKE_S3_KEY_ID}"
secret = "${DUCKLAKE_S3_SECRET}"
```

Metaxy supports `${VAR}` and `${VAR:-default}` syntax. See [Configuration: Templating Environment Variables](../reference/configuration.md#templating-environment-variables).

## MotherDuck (MD) Configuration

Use `DuckDBMetadataStore` with a MotherDuck connection string and set the DuckLake catalog backend to `motherduck`.

### Option 1: MotherDuck-managed DuckLake

```toml
[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "md:${MD_DATABASE}?motherduck_token=${MOTHERDUCK_TOKEN}"

[stores.dev.config.ducklake]
alias = "ducklake"

[stores.dev.config.ducklake.catalog]
type = "motherduck"
database = "${MD_DUCKLAKE_DATABASE}"
region = "${MOTHERDUCK_REGION:-eu-central-1}"
```

In this mode, MotherDuck manages the DuckLake catalog and storage.

### Option 2: MotherDuck BYOB (Bring Your Own Bucket)

```toml
[stores.dev]
type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "md:${MD_DATABASE}?motherduck_token=${MOTHERDUCK_TOKEN}"

[stores.dev.config.ducklake]
alias = "ducklake"

[stores.dev.config.ducklake.catalog]
type = "motherduck"
database = "${MD_DUCKLAKE_DATABASE}"
region = "${MOTHERDUCK_REGION:-eu-central-1}"

[stores.dev.config.ducklake.storage]
type = "s3"
secret_name = "ducklake_s3"
bucket = "${DUCKLAKE_S3_BUCKET}"
region = "${DUCKLAKE_S3_REGION:-eu-central-1}"
key_id = "${DUCKLAKE_S3_KEY_ID}"
secret = "${DUCKLAKE_S3_SECRET}"
```

In BYOB mode, Metaxy creates storage secrets `IN MOTHERDUCK` so MotherDuck compute can access your bucket.

## Related Materials

- [DuckLake Integration Reference](../integrations/metadata-stores/storage/ducklake.md)
- [DuckDB Metadata Store](../integrations/metadata-stores/databases/duckdb.md)
- [`DuckLakeConfig`][metaxy.ext.metadata_stores.ducklake.DuckLakeConfig]
