# DeltaMetadataStore

Delta Lake keeps one Delta table per feature, appends new metadata versions, and uses the in-memory Polars versioning engine for provenance calculations.

## Installation

The backend relies on [`deltalake`](https://delta-io.github.io/delta-rs/python/) which ships with Metaxy’s `delta` extras.

## Object Stores

Point `root_path` at any supported URI (`s3://`, `abfss://`, `gs://`, …) and forward credentials with `storage_options`.
The dict is passed verbatim to [`deltalake`](https://delta-io.github.io/delta-rs/integrations/object-storage/special_configuration/).

## Storage Layout

Control how feature keys map to Delta directories with `layout`:

- `flat` (default) keeps the current `your__feature_key` directory scheme.
- `nested` places every key part into its own directory (e.g. `your/feature/key`).
