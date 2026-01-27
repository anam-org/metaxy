---
title: "BigQuery Metadata Store"
description: "BigQuery as a metadata store backend."
---

# Metaxy + BigQuery

Metaxy implements [`BigQueryMetadataStore`][metaxy.metadata_store.bigquery.BigQueryMetadataStore]. It uses [BigQuery](https://cloud.google.com/bigquery) as metadata storage and versioning engine.

## Installation

```shell
pip install 'metaxy[bigquery]'
```

## API

::: metaxy.metadata_store.bigquery
options:
members: false

<!-- dprint-ignore-start -->
::: metaxy.metadata_store.bigquery.BigQueryMetadataStore
    options:
      inherited_members: false
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.metadata_store.bigquery.BigQueryMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
