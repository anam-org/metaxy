---
title: "BigQuery Metadata Store"
description: "BigQuery as a metadata store backend."
---

# BigQuery

!!! warning "Experimental"

    This functionality is experimental.

[BigQuery](https://cloud.google.com/bigquery) is a serverless data warehouse managed by Google Cloud. To use Metaxy with BigQuery, configure [`BigQueryMetadataStore`][metaxy.metadata_store.bigquery.BigQueryMetadataStore]. Versioning computations run natively in BigQuery.

## Installation

```shell
pip install 'metaxy[bigquery]'
```

---

<!-- dprint-ignore-start -->
::: metaxy.metadata_store.bigquery
    options:
      members: false
      show_root_heading: true
      heading_level: 2

::: metaxy.metadata_store.bigquery.BigQueryMetadataStore
    options:
      members: false
      heading_level: 3
<!-- dprint-ignore-end -->

## Configuration

<!-- dprint-ignore-start -->
::: metaxy-config
    class: metaxy.metadata_store.bigquery.BigQueryMetadataStoreConfig
    path_prefix: stores.dev.config
    header_level: 3
<!-- dprint-ignore-end -->
