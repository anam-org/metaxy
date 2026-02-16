---
title: "Delta Lake Metadata Store"
description: "Delta Lake as a metadata store backend."
---

# Delta Lake

[Delta Lake](https://delta.io/) is an open-source lakehouse storage format with ACID transactions and schema enforcement. To use Metaxy with Delta Lake, configure [`DeltaMetadataStore`][metaxy.ext.metadata_stores.delta.DeltaMetadataStore]. It persists metadata as Delta tables and uses an in-memory Polars engine for versioning computations.

It supports the local filesystem and remote object stores.

!!! tip

    If Polars 1.37 or greater is installed, lazy Polars frames are sinked via
    `LazyFrame.sink_delta`, avoiding unnecessary materialization.

## Installation

```shell
pip install 'metaxy[delta]'
```

## Using Object Stores

Point `root_path` at any supported URI (`s3://`, `abfss://`, `gs://`, ...) and forward credentials with `storage_options`.
The dict is passed verbatim to [`deltalake`](https://delta-io.github.io/delta-rs/integrations/object-storage/special_configuration/).

Learn more in the [API docs][metaxy.ext.metadata_stores.delta.DeltaMetadataStore].

## Storage Layout

It's possible to control how feature keys map to DeltaLake table locations with the `layout` parameter:

- `nested` (default) places every feature in its own directory: `your/feature/key.delta`
- `flat` stores all of them in the same directory: `your__feature_key.delta`

---

<!-- dprint-ignore-start -->
::: metaxy.ext.metadata_stores.delta
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.metadata_stores.delta.DeltaMetadataStore
    options:
      members: false
      heading_level: 2
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.metadata_stores.delta.DeltaMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
