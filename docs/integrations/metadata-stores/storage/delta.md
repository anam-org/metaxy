---
title: "Metaxy + Delta Lake"
description: "Learn how to use Delta Lake to store Metaxy metadata."
---

# Metaxy + Delta Lake

[Delta Lake](https://delta.io/) is an open-source lakehouse storage format with ACID transactions and schema enforcement. To use Metaxy with Delta Lake, configure [`DeltaMetadataStore`][metaxy.ext.metadata_stores.delta.DeltaMetadataStore]. It persists metadata as Delta tables and uses an in-memory Polars engine for versioning computations.

It supports the local filesystem and remote object stores.

!!! tip

    If Polars 1.37 or greater is installed, lazy Polars frames are sinked via
    `LazyFrame.sink_delta`, avoiding unnecessary materialization.

## Installation

```shell
pip install 'metaxy[delta]'
```

## API Reference

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
