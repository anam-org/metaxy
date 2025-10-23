# Metadata Migration System

## Overview

Metaxy's migration system enables safe, automated updates to feature metadata when feature definitions change. Migrations are explicit, idempotent, and automatically propagate through the entire dependency graph by default. They can be customized and user-defined. Migrations respect custom `align_metadata_with_upstream` method which may be implemented for specific features.

## Core Concepts

### System Tables

Metaxy maintains two system tables in the metadata store:

#### 1. Feature Version History
**Key:** `["__metaxy__", "feature_versions"]`

Tracks when each feature version was recorded:

```python
{
    "feature_key": "video_processing",
    "feature_version": "a3f8b2c1",
    "recorded_at": datetime(2025, 1, 10, 14, 30, 0),
    "fields": ["frames", "audio"],
    "snapshot_id": "9fee05a3",  # Graph snapshot hash (if recorded with serialize_feature_graph)
}
```

**Purpose:**
- Detection of latest version for migrations
- Audit trail of materializations
- Enables `detect_feature_changes()` to work

**Updated explicitly** by calling `store.record_feature_graph_snapshot(Feature)`.
This is typically done in CI after successful materialization.

#### 2. Migration History
**Key:** `["__metaxy__", "migrations"]`

Tracks applied migrations:

```python
{
    "migration_id": "migration_20250113_103000",
    "applied_at": datetime(2025, 1, 13, 10, 30, 0),
    "status": "completed",  # or "partial", "failed"
    "operations_count": 2,
    "affected_features": ["video_processing", "feature_c", "feature_d"],
    "errors": None,  # or dict of errors if partial/failed
}
```

**Purpose:**
- Idempotency - prevent re-running completed migrations
- Recovery tracking - know which migrations are partial
- Audit trail

## Recording Feature Versions

Before migrations can be detected, you must explicitly record feature versions before materialization.
This is typically done in CI/CD pipelines after successful feature computation.

```python
# Explicitly record the version (call this in CI)
store.record_feature_graph_snapshot()

store.write_metadata(MyFeature, metadata_df)
```

**Why explicit?**
- **Performance**: Avoids overhead on every write
- **Control**: Only record "published" versions in production
- **CI integration**: Natural fit with deployment pipelines

**When to call:**
- ✅ Before materializing features in production
- ✅ In CI/CD after tests pass
- ✅ When promoting features from dev to staging/prod
- ❌ Not on every write during development
- ❌ Not for temporary/experimental data


**Snapshot ID:**

The `snapshot_id` returned by `serialize_feature_graph()` is a deterministic hash
representing the entire feature graph state. It's:
- ✅ Idempotent - same graph = same snapshot_id
- ✅ Unique - different graph = different snapshot_id
- ✅ No timestamp - purely based on feature versions
- ✅ Ties all features together in a consistent snapshot

This enables:
- Tracking which features were materialized together
- Identifying complete graph states
- Detecting incomplete materializations (some features missing snapshot_id)

## Migration Workflow

### 1. Developer Changes Feature Code

```python
# Before: code_version = 1
class VideoProcessing(Feature, spec=FeatureSpec(
    fields=[
        FieldSpec(key=FieldKey(["default"]), code_version=1),
    ],
)):
    pass

# After: code_version = 2
class VideoProcessing(Feature, spec=FeatureSpec(
    fields=[
        FieldSpec(key=FieldKey(["default"]), code_version=2),  # Changed!
    ],
)):
    pass
```

## Migration Execution Details

### How Migrations Work (Immutable Copy-on-Write)

**Migrations preserve immutability** - they never modify existing metadata.
Instead, they **copy** old metadata, recalculate data versions, and **append** new rows.

When you run `metaxy migrations apply`:

**Step 1: Check Idempotency**
```python
# Is migration already completed?
if _is_migration_completed(store, migration.id):
    print("Migration already completed, skipping.")
    return
```

**Step 2: Migrate Source Features (Immutable Copy-on-Write)**
```python
for operation in migration.operations:
    # Query rows with OLD feature_version
    old_rows = store.read_metadata(
        feature,
        current_only=False,
        filters=pl.col("feature_version") == operation.from,
    )

    if len(old_rows) == 0:
        # Already migrated (idempotent)
        continue

    # Copy ALL metadata columns (preserves paths, labels, etc.)
    # Exclude only data_version and feature_version (will be recalculated)
    columns_to_keep = [c for c in old_rows.columns
                      if c not in ["data_version", "feature_version"]]
    sample_metadata = old_rows.select(columns_to_keep)

    # Recalculate data versions and APPEND new rows
    # Old rows remain unchanged (immutable!)
    # New rows get:
    # - Same sample_id and user metadata columns
    # - Recalculated data_version (based on new feature definition)
    # - New feature_version
    store.calculate_and_write_data_versions(
        feature=feature,
        sample_df=sample_metadata,
    )
```

**Result:** Metadata store now contains BOTH old and new versions:
- Old rows: `feature_version=abc123`, original `data_version`, original metadata
- New rows: `feature_version=def456`, recalculated `data_version`, **same metadata**

Reading with `current_only=True` returns only new rows.

**Step 3: Propagate Downstream (Automatic)**
```python
# Discover entire downstream DAG
source_keys = [op.feature_key for op in migration.operations]
downstream_keys = store.graph.get_downstream_features(source_keys)

# Process in topological order (dependencies first)
for downstream_key in downstream_keys:
    feature = graph.features_by_key[downstream_key]

    # Read CURRENT metadata
    current_metadata = store.read_metadata(
        feature,
        current_only=True,  # Only current feature_version
    )

    # Read upstream (gets NEW data_versions from step 2)
    upstream_metadata = store.read_upstream_metadata(
        feature,
        current_only=True,
    )

    # User-defined alignment logic
    aligned_metadata = feature.align_metadata_with_upstream(
        current_metadata,
        upstream_metadata,
    )

    # Recalculate and write
    new_metadata = store.calculate_and_write_data_versions(
        feature=feature,
        sample_df=aligned_metadata,
    )
```

**Step 4: Record Completion**
```python
# Write to migration history
migration_record = pl.DataFrame({
    "migration_id": migration.id,
    "applied_at": datetime.now(),
    "status": "completed",
    "operations_count": len(migration.operations),
    "affected_features": [...],
})

store.write_metadata(
    FeatureKey(["__metaxy__", "migrations"]),
    migration_record,
)
```

### Failure Recovery

Migrations are **automatically recoverable** from partial failures.

**How recovery works:**
1. Re-run queries for rows with old `feature_version`
2. If not found: already migrated, skip
3. If found: migrate them
4. Eventually all rows migrated, migration marked completed
5. Future runs are instant no-ops

Migrations can be safely re-run multiple times.

## Migration File Format (YAML)

```yaml
version: 1  # Migration schema version
id: "migration_20250113_103000"  # Unique ID (timestamp-based)
description: "Auto-generated migration for 2 changed features"
created_at: "2025-01-13T10:30:00Z"

operations:
  - type: feature_version_migration
    feature_key: ["video", "processing"]
    from: "abc12345"  # Old version (from metadata)
    to: "def67890"    # New version (from code)
    change_type: "unknown"
    reason: "Updated frame extraction algorithm"

  - type: feature_version_migration
    feature_key: ["audio", "processing"]
    from: "xyz11111"
    to: "xyz22222"
    change_type: "unknown"
    reason: "Added AAC codec support"

# Note: Downstream propagation is automatic
# Not listed explicitly in YAML
```

### Migration File Location

Migrations are stored in a directory (default: `migrations/`) and named by timestamp:

```
migrations/
├── 20250110_120000_initial_setup.yaml
├── 20250113_103000_update_video_processing_audio_processing.yaml
└── 20250115_160000_add_new_features.yaml
```

Files are applied **in sorted order** (timestamp ensures correct ordering).

## Best Practices

### 1. Always Dry-Run First

```bash
$ metaxy migrations apply --dry-run
# Review impact before executing
$ metaxy migrations apply
```

### 2. Commit Migrations to Git

Migrations are part of your codebase:
```bash
$ git add migrations/
$ git commit -m "Add migration for video processing v2"
```

### 3. Document Reasons

Always fill in meaningful `reason` fields:
```yaml
reason: "Updated frame extraction algorithm for better quality"  # ✓ Good
reason: "TODO: Describe what changed"  # ✗ Bad (leftover template)
```

### 4. Test Alignment Logic

If you override `align_metadata_with_upstream()`, test it thoroughly:
```python
def test_custom_alignment():
    current = pl.DataFrame({"sample_id": [1, 2, 3], "custom_field": ["a", "b", "c"]})
    upstream = {"videos": pl.DataFrame({"sample_id": [2, 3, 4]})}

    result = MyFeature.align_metadata_with_upstream(current, upstream)

    # Verify alignment behavior
    assert set(result["sample_id"].to_list()) == {2, 3}  # Inner join
    assert "custom_field" in result.columns  # Preserved
```

### 5. Monitor Migration Progress

For large migrations, monitor progress:
```python
result = apply_migration(store, migration)
print(f"Affected {len(result.affected_features)} features")
print(f"Duration: {result.duration_seconds}s")
if result.errors:
    print(f"Errors: {result.errors}")
```

### 6. Keep Old Versions

Don't manually delete old feature_version rows unless you have a specific cleanup policy. They serve as:
- Historical record
- Debug reference
- Rollback option
