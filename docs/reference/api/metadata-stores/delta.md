---
title: Delta Lake Metadata Store
---

# DeltaMetadataStore

Delta Lake is the recommended object-store-backed metadata backend.
It keeps one Delta table per feature, appends new metadata versions, and leans on Polars / Narwhals for provenance calculations.

## Installation

The backend relies on [`deltalake`](https://delta-io.github.io/delta-rs/python/)
which ships with Metaxy’s `delta` dependency group:

```bash
uv sync --group delta
```

## Object Stores

Point `root_path` at any supported URI (`s3://`, `abfss://`, `gs://`, `file://`,
`local://`, …) and forward credentials with `storage_options`.
The dict is passed
verbatim to `deltalake`, so you can reuse the same keys you would use with delta-rs directly.

## Storage Layout

Control how feature keys map to Delta directories with `layout`:

- `flat` (default) keeps the current `<namespace>__<feature>` directory scheme.
- `nested` places every key part into its own directory (e.g.
  `namespace/feature/channel`), which can make large hierarchies easier to
  inspect in object-store consoles.

Only URI generation changes; table contents remain identical.

## Hashing

`DeltaMetadataStore` defaults to `HashAlgorithm.XXHASH64`, in line with the other non-SQL stores.
Use the constructor arguments to override the algorithm if you need a different hash.

::: metaxy.metadata_store.delta.DeltaMetadataStore
