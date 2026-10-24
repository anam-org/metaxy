---
title: "Metaxy + Delta Lake"
description: "Learn how to use Delta Lake to store Metaxy metadata."
---

# Metaxy + Delta Lake

[Delta Lake](https://delta.io/) is an open-source lakehouse storage format with ACID transactions and schema enforcement. To use Metaxy with Delta Lake, configure [`DeltaMetadataStore`][metaxy.ext.polars.handlers.delta.DeltaMetadataStore]. It persists metadata as Delta tables and uses an in-memory Polars engine for versioning computations.

It supports the local filesystem and remote object stores.

!!! tip

    If Polars 1.37 or greater is installed, lazy Polars frames are sinked via
    `LazyFrame.sink_delta`, avoiding unnecessary materialization.

!!! tip "Recommended: enable [`Map` datatype](../../../guide/concepts/metadata-stores.md#map-datatype)"

    Delta Lake supports the `Map` natively. Enabling [`enable_map_datatype`](../../../reference/configuration.md#metaxy.config.MetaxyConfig.enable_map_datatype) preserves `Map` columns across read and write operations.

## Installation

```shell
pip install 'metaxy[delta]'
```

## API Reference

<!-- dprint-ignore-start -->
::: metaxy.ext.polars.handlers.delta
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.polars.handlers.delta.DeltaMetadataStore
    options:
      members: false
      heading_level: 2
<!-- dprint-ignore-end -->

## Configuration

Use JSON arrays for list environment overrides and JSON objects for dictionary overrides. Quote the JSON so the shell passes it unchanged. For example, these overrides clear the configured fallback stores and set the storage region:

```shell
export METAXY_STORES__DEV__CONFIG__FALLBACK_STORES='[]'
export METAXY_STORES__DEV__CONFIG__STORAGE_OPTIONS='{"AWS_REGION": "eu-central-1"}'
```

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.polars.handlers.delta.DeltaMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
