# System Column Registry

Metaxy reserves a small set of system-managed columns that it attaches to feature
metadata tables. These columns are part of the platform contract and are used by
the metadata store, versioning engine, and migration tooling to keep track of
feature lineage.

All system column names start with the `metaxy_` prefix to avoid collisions with
user-defined feature fields. Only the prefixed forms are supported.

## Canonical column names

| Canonical name | Description |
| --- | --- |
| `metaxy_provenance_by_field` | Struct capturing per-field provenance hashes. Always present for versioning. |
| `metaxy_feature_version` | Current feature implementation hash written on every append. |
| `metaxy_snapshot_version` | Identifier of the deployment snapshot that produced the row. |
| `metaxy_feature_spec_version` | Hash of the entire feature specification (stored in system tables). |
| `metaxy_feature_tracking_version` | Hash that combines spec version and project for multi-project tracking. |

The canonical column constants are defined in `metaxy.models.constants`:

```python
from metaxy.models.constants import (
    METAXY_PROVENANCE_BY_FIELD_COLUMN,
    METAXY_FEATURE_VERSION_COLUMN,
    METAXY_SNAPSHOT_VERSION_COLUMN,
    METAXY_FEATURE_SPEC_VERSION_COLUMN,
    METAXY_FEATURE_TRACKING_VERSION_COLUMN,
)
```

## Working with system columns in code

Utility helpers in `metaxy.models.constants` simplify working with custom pipelines:

```python
from metaxy.models.constants import (
    is_system_column,
    is_droppable_system_column,
)

# Test whether a column is managed by Metaxy
assert is_system_column("metaxy_snapshot_version")

# Determine if a system column can be dropped when pulling data from upstream features
assert is_droppable_system_column("metaxy_feature_version")
```

These helpers ensure new code adopts the `metaxy_` prefix consistently.
