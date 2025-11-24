# DeltaMetadataStore

Metaxy implements [`DeltaMetadataStore`][metaxy.metadata_store.delta.DeltaMetadataStore] that stores metadata in [Delta Lake](https://delta.io/) (also known as a LakeHouse format) and uses an in-memory Polars versioning engine.

It supports the local filesystem and remote object stores.

## Installation

This metadata store relies on [`deltalake`](https://delta-io.github.io/delta-rs/python/):

```shell
$ pip install 'metaxy[delta]'
```

## Using Object Stores

Point `root_path` at any supported URI (`s3://`, `abfss://`, `gs://`, …) and forward credentials with `storage_options`.
The dict is passed verbatim to [`deltalake`](https://delta-io.github.io/delta-rs/integrations/object-storage/special_configuration/).

Learn more in the [API docs][metaxy.metadata_store.delta.DeltaMetadataStore].

## Storage Layout

It's possible to control how feature keys map to DeltaLake table locations with the `layout` parameter:

- `nested` (default) places every feature in its own directory: `your/feature/key.delta`
- `flat` stores all of them in the same directory: `your__feature_key.delta`
