# Metaxy + LanceDB

Metaxy implements [`LanceDBMetadataStore`][metaxy.metadata_store.lancedb.LanceDBMetadataStore]. LanceDB keeps one Lance table per feature, writes metadata in append mode, and uses the in-memory Polars versioning engine for provenance calculations. LanceDB handles schema evolution, transactions, and compaction automatically.

It runs embedded (local directory) or against external storage (object stores, HTTP endpoints, LanceDB Cloud), so you can use the same store for local development and remote compute.

See [Configuration Reference](../../../reference/api/metadata-stores/lancedb.md#configuration) for all available options.

## Installation

The backend relies on [`lancedb`](https://lancedb.com/), which is shipped with Metaxy's `lancedb` extras.

## Storage Targets

Point `uri` at any supported URI (`s3://`, `gs://`, `az://`, `db://`, …) and forward credentials with the platform's native mechanism (environment variables, IAM roles, workload identity, etc.). LanceDB supports local filesystem, S3, GCS, Azure, LanceDB Cloud, and remote HTTP/HTTPS endpoints.

## Storage Layout

All tables are stored within a single LanceDB database at the configured URI location.
Each feature gets its own Lance table.

# Reference

- [Configuration Reference](../../../reference/api/metadata-stores/lancedb.md#configuration)
- [API Reference][metaxy.metadata_store.lancedb.LanceDBMetadataStore]
