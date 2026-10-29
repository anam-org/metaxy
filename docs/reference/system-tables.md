---
title: "System Tables"
description: "Tables that Metaxy manages in the metadata store."
---

# System Tables

Metaxy keeps its own tables in the [metadata store][metaxy.MetadataStore], next to the feature tables. If the store has a `table_prefix`, the table names start with this prefix.

## `metaxy_system__feature_versions`

This table is typically managed from CI/CD with [`metaxy push`](./cli.md#metaxy-push). It stores serialized [feature definitions][metaxy.FeatureDefinition] instances.

The table is append-only: each push with changes adds rows. For each `feature_key`, the row with the latest `recorded_at` holds the current definition.

| Column                      | Type      | Description                                                                                               |
| --------------------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| `project`                   | string    | The project that defines the feature.                                                                     |
| `feature_key`               | string    | The [feature key][metaxy.FeatureKey].                                                                     |
| `metaxy_feature_version`    | string    | The [feature version](./system-columns.md). Metadata rows of this version have the same value.            |
| `metaxy_definition_version` | string    | The [definition version](./system-columns.md): a hash of the feature spec and the Pydantic model schema.  |
| `metaxy_project_version`    | string    | The [project version](./system-columns.md) at the time of the push.                                       |
| `recorded_at`               | timestamp | The time of the push.                                                                                     |
| `feature_spec`              | string    | The [`FeatureSpec`][metaxy.FeatureSpec] as JSON: ID columns, fields, upstream dependencies and more.      |
| `feature_schema`            | string    | The JSON schema of the Pydantic model of the feature.                                                     |
| `feature_class_path`        | string    | The Python import path of the feature class.                                                              |
| `tags`                      | string    | The snapshot tags as a JSON object.                                                                       |
| `deleted_at`                | timestamp | The time when the feature was removed from the project. Null while the feature exists.                    |
