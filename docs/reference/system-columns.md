---
title: "System Columns"
description: "Reserved system columns in Metaxy."
---

# System Columns

Metaxy reserves a set of [system-managed columns](./api/constants.md) that it attaches to user-defined feature
metadata tables. These columns are part of the storage interface and are used by
the [metadata store][metaxy.MetadataStore]. Learn more about the storage layout design [here](../metaxy/design.md#storage).

## Canonical column names

| Canonical name                 | Explanation                                                         | Level   | Type   |
| ------------------------------ | ------------------------------------------------------------------- | ------- | ------ |
| `metaxy_provenance_by_field`   | Derived from upstream data versions and code version per field      | sample  | struct |
| `metaxy_provenance`            | Hash of `metaxy_provenance_by_field`                                | sample  | string |
| `metaxy_data_version_by_field` | Defaults to `metaxy_provenance_by_field`, can be user-defined       | sample  | struct |
| `metaxy_data_version`          | Hash of `metaxy_data_version_by_field`                              | sample  | string |
| `metaxy_feature_version`       | Derived from versions of relevant upstream fields                   | feature | string |
| `metaxy_project_version`       | Derived from all Metaxy features which belong to the same Project   | project | string |
| `metaxy_definition_version`    | Hash of the feature spec and Pydantic model schema                  | feature | string |
| `metaxy_created_at`            | Timestamp when the metadata row was created                         | sample  | string |
| `metaxy_updated_at`            | Timestamp when the metadata row was last written to the store       | sample  | string |
| `metaxy_deleted_at`            | Timestamp when the metadata row was soft-deleted (null if active)   | sample  | string |
| `metaxy_materialization_id`    | External orchestration run ID (e.g., Dagster, Airflow) for tracking | run     | string |

All system column names start with the `metaxy_` prefix.

## Example Table

--8<-- "table.md"
