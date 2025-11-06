# System Column Registry

Metaxy reserves a small set of [system-managed columns](../reference/api/constants.md) that it attaches to feature
metadata tables. These columns are part of the platform contract and are used by
the [metadata store][metaxy.MetadataStore], versioning engine, and migration tooling to keep track of
feature lineage.

All system column names start with the `metaxy_` prefix to avoid collisions with
user-defined feature fields. Only the prefixed forms are supported.

## Canonical column names

| Canonical name                    | Description                                                             |
| --------------------------------- | ----------------------------------------------------------------------- |
| `metaxy_provenance_by_field`      | Struct capturing per-field provenance hashes. Sample level.             |
| `metaxy_data_version_by_field`    | Optional struct overriding provenance exposed to downstream features.   |
| `metaxy_feature_version`          | Version of the versioned graph upstream to the feature.                 |
| `metaxy_snapshot_version`         | Version of the entire feature graph for the Metaxy project.             |
| `metaxy_feature_spec_version`     | Version of the feature spec part responsible for graph topology.        |
| `metaxy_feature_tracking_version` | Hash that combines spec version and project for multi-project tracking. |
