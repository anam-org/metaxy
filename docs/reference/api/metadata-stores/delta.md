---
title: Delta Lake Metadata Store
---

# DeltaMetadataStore

Deltalake is a object-store-backed metadata backend.
It keeps one Delta table per feature and appends new metadata versions.
Polars is used to write the tables.

## Installation

The backend relies on [`deltalake`](https://delta-io.github.io/delta-rs/python/)
which ships with Metaxy’s `delta` dependency group:

```bash
uv sync --extra delta
```

## Object Stores

Point `root_path` at any supported URI and forward credentials with `storage_options`.
See the [DeltaLake documentation](https://delta-io.github.io/delta-rs/python/usage.html#loading-a-delta-table) for supported URI schemes and storage options.

## Storage Layout

Control how feature keys map to Delta directories with `layout`:

- `flat` (default) keeps the default `<namespace>__<feature>` directory scheme.
- `nested` places every key part into its own directory (e.g.
  `namespace/feature/channel`).

```

::: metaxy.metadata_store.delta.DeltaMetadataStore
