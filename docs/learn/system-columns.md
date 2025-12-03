# System Column Registry

Metaxy reserves a set of [system-managed columns](../reference/api/constants.md) that it attaches to user-defined feature
metadata tables. These columns are part of the platform contract and are used by
the [metadata store][metaxy.MetadataStore], versioning engine, and migration tooling.

All system column names start with the `metaxy_` prefix.

## Canonical column names

| Canonical name                   | Explanation                                                             | Level   | Type   |
| -------------------------------- | ----------------------------------------------------------------------- | ------- | ------ |
| `metaxy_provenance_by_field`     | Derived from upstream data versions and code version per field          | sample  | struct |
| `metaxy_provenance`              | Hash of `metaxy_provenance_by_field`                                    | sample  | string |
| `metaxy_data_version_by_field`   | Defaults to `metaxy_provenance_by_field`, can be user-defined           | sample  | struct |
| `metaxy_data_version`            | Hash of `metaxy_data_version_by_field`                                  | sample  | string |
| `metaxy_feature_version`         | Derived from versions of relevant upstream fields                       | feature | string |
| `metaxy_snapshot_version`        | Derived from the entire Metaxy feature graph                            | graph   | string |
| `metaxy_feature_spec_version`    | Derived from the part of the feature spec responsible for versioning    | sample  | string |
| `metaxy_full_definition_version` | Hash of the entire feature Pydanitc model schema and the Metaxy project | string  | true   |
| `metaxy_created_at`              | Timestamp when the metadata row was created                             | sample  | string |
| `metaxy_materialization_id`      | External orchestration run ID (e.g., Dagster, Airflow) for tracking     | run     | string |
