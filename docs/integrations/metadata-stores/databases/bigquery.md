---
title: "Metaxy + BigQuery"
description: "Learn how to use BigQuery as a Metaxy metadata store."
---

# Metaxy + BigQuery

!!! warning "Experimental"

    This functionality is experimental.

[BigQuery](https://cloud.google.com/bigquery) is a serverless data warehouse managed by Google Cloud. To use Metaxy with BigQuery, configure [`BigQueryMetadataStore`][metaxy.ext.metadata_stores.bigquery.BigQueryMetadataStore]. Versioning computations run natively in BigQuery.

## Installation

```shell
pip install 'metaxy[bigquery]'
```

## API Reference

<!-- dprint-ignore-start -->
::: metaxy.ext.metadata_stores.bigquery
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.ext.metadata_stores.bigquery.BigQueryMetadataStore
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.ext.metadata_stores.bigquery.BigQueryMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
