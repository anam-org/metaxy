# Metadata Migration System

## Overview

Metaxy's migration system enables safe, automated updates to feature metadata when feature definitions change. Migrations are explicit, idempotent, and automatically propagate through the entire dependency graph by default. They can be customized and user-defined. Migrations respect custom `align_metadata_with_upstream` method which may be implemented for specific features.

**Prerequisites:** The migration system requires that you record feature graph snapshots in your CD (Continuous Deployment) workflow using `metaxy push`. This command snapshots the complete feature graph state and is essential for migration detection to work.

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

## Recording Feature Graph Snapshots (User Responsibility)

**YOU MUST** record feature graph snapshots in your CD workflow for migrations to work.

### Using `metaxy push` in CD

Add this to your deployment pipeline (BEFORE materializing features):

```bash
# In your CI/CD pipeline (GitHub Actions, Jenkins, etc.)
$ metaxy push
```

This command:
1. Serializes the complete feature graph (all features, dependencies, field definitions)
2. Generates a deterministic `snapshot_id` (hash of all feature_versions)
3. Records the snapshot in the `__metaxy__/feature_versions` system table
4. Returns the snapshot_id for your records

**Example CD Workflow:**
```bash
#!/bin/bash
# deploy.sh - Your deployment script

# 1. Deploy new code to production
git pull origin main
pip install -e .

# 2. Record feature graph snapshot (CRITICAL!)
metaxy push
# Output: "Recorded snapshot: a3f8b2c1... (12 features)"

# 3. Run your feature materialization pipeline
python -m myproject.compute_features

# 4. Generate migration if code changed
metaxy migrations generate
# If features changed, creates: migrations/migration_20250113_103000.yaml

# 5. Apply migration automatically
metaxy migrations apply --latest
```

**Why is `metaxy push` required?**
- Migration detection compares "latest snapshot in store" vs "current code"
- Without snapshots, the system cannot detect what changed
- Provides audit trail: what feature versions were deployed when
- Enables historical graph reconstruction for safe migrations

**What gets recorded:**
```python
# For each feature in the graph:
{
    "feature_key": "video_processing",
    "feature_version": "a3f8b2c1",  # Hash of feature definition
    "snapshot_id": "9fee05a3",      # Hash of entire graph
    "recorded_at": "2025-01-13T10:30:00Z",
    "feature_spec": {...},          # Complete feature definition (serialized)
    "feature_class_path": "myproject.features.VideoProcessing"
}
```

**Snapshot ID Properties:**

The `snapshot_id` is a deterministic hash representing the entire feature graph state:
- ✅ Idempotent - same graph = same snapshot_id every time
- ✅ Unique - any change = different snapshot_id
- ✅ No timestamp - purely content-based hash
- ✅ Groups all features deployed together

**Benefits:**
- Detect exactly which features changed between deployments
- Reconstruct historical feature graphs for safe migrations
- Audit trail of what was deployed when
- Identify incomplete deployments (some features missing snapshot)

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
    # Operations derive old/new feature_versions from migration's snapshot IDs
    # Query feature versions from snapshot metadata
    from_feature_version = get_version_from_snapshot(
        store, operation.feature_key, migration.from_snapshot_id
    )
    to_feature_version = get_version_from_snapshot(
        store, operation.feature_key, migration.to_snapshot_id
    )

    # Query rows with OLD feature_version
    old_rows = store.read_metadata(
        feature,
        current_only=False,
        filters=pl.col("feature_version") == from_feature_version,
    )

    if len(old_rows) == 0:
        # Already migrated (idempotent)
        continue

    # Copy ALL metadata columns (preserves paths, labels, etc.)
    # Exclude only data_version, feature_version, and snapshot_id (will be recalculated)
    columns_to_keep = [c for c in old_rows.columns
                      if c not in ["data_version", "feature_version", "snapshot_id"]]
    sample_metadata = old_rows.select(columns_to_keep)

    # Recalculate data versions and APPEND new rows
    # Old rows remain unchanged (immutable!)
    # New rows get:
    # - Same sample_id and user metadata columns
    # - Recalculated data_version (based on new feature definition)
    # - New feature_version and snapshot_id
    store.calculate_and_write_data_versions(
        feature=feature,
        sample_df=sample_metadata,
    )
```

**Result:** Metadata store now contains BOTH old and new versions:
- Old rows: `feature_version=abc123`, `snapshot_id=snap1`, original `data_version`, original metadata
- New rows: `feature_version=def456`, `snapshot_id=snap2`, recalculated `data_version`, **same metadata**

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
parent_migration_id: "migration_20250110_120000"  # Previous migration (if any)
description: "Auto-generated migration for 2 changed features + 3 downstream"
created_at: "2025-01-13T10:30:00Z"

# Snapshot IDs identify the complete feature graph state (before and after)
from_snapshot_id: "a3f8b2c1..."  # Source snapshot (old state in store)
to_snapshot_id: "def67890..."    # Target snapshot (new state in code)

operations:
  # Root features that changed (code refactoring, not computation changes)
  - id: "reconcile_stt_transcription"
    type: "metaxy.migrations.ops.DataVersionReconciliation"
    feature_key: ["speech", "transcription"]
    reason: "Fixed dependency: now depends only on audio field instead of entire video. Computation logic unchanged, just cleaner graph structure."

  - id: "reconcile_face_detection"
    type: "metaxy.migrations.ops.DataVersionReconciliation"
    feature_key: ["video", "face_detection"]
    reason: "Refactored field structure from single 'default' to 'faces' and 'confidence' fields. Same underlying model, just better schema."

  # Downstream features (auto-generated)
  - id: "reconcile_speaker_diarization"
    type: "metaxy.migrations.ops.DataVersionReconciliation"
    feature_key: ["speech", "diarization"]
    reason: "Reconcile data_versions due to changes in: speech/transcription"

# Feature versions are derived from snapshot IDs at runtime
# Each snapshot uniquely identifies all feature versions in the graph

# IMPORTANT: DataVersionReconciliation is for when CODE changed but COMPUTATION didn't
# Use cases:
# - Dependency graph refactoring (more precise dependencies, merging features)
# - Field structure changes (renaming, splitting fields)
#
# If computation ACTUALLY changed, users must re-run their pipeline
# Migrations cannot recreate lost computation results
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

### 3. Understand When to Use DataVersionReconciliation

**DataVersionReconciliation is ONLY for when:**
- ✅ Code structure changed (refactoring, better dependencies)
- ✅ Graph topology changed (more precise field dependencies)
- ✅ Schema improved (field renaming, splitting)
- ✅ **Computation logic is unchanged** - results would be identical

**Do NOT use DataVersionReconciliation when:**
- ❌ Algorithm actually changed (different output)
- ❌ Model updated (new ML model version)
- ❌ Bug fixed that affects results
- ❌ **Computation would produce different results**

**Example - Use DataVersionReconciliation:**
```yaml
# Before: STT depends on entire video
deps: [FeatureDep(key=["video", "file"])]

# After: STT depends only on audio (more precise)
deps: [FeatureDep(key=["video", "file"], fields=["audio"])]

# Computation is identical, just cleaner graph
reason: "Refined dependency to audio field only. Transcription logic unchanged."
```

**Example - Re-run Pipeline Instead:**
```python
# Before: Whisper v2
model = whisper.load_model("base")

# After: Whisper v3 (DIFFERENT RESULTS!)
model = whisper.load_model("large-v3")

# DO NOT USE MIGRATION - re-run your pipeline!
# Results would be different, existing data is now stale
```

### 4. Document Reasons

Always fill in meaningful `reason` fields explaining what changed:
```yaml
reason: "Corrected dependency from entire video to audio field only. Computation unchanged."  # ✓ Good
reason: "Refactored field structure - split 'default' into 'faces' and 'confidence'. Same detection model."  # ✓ Good
reason: "TODO: Describe what changed"  # ✗ Bad (leftover template)
reason: "Updated algorithm"  # ✗ Bad (vague - did results change?)
```

### 5. Test Alignment Logic

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

### 6. Monitor Migration Progress

For large migrations, monitor progress:
```python
result = apply_migration(store, migration)
print(f"Affected {len(result.affected_features)} features")
print(f"Duration: {result.duration_seconds}s")
if result.errors:
    print(f"Errors: {result.errors}")
```

### 7. Keep Old Versions

Don't manually delete old feature_version rows unless you have a specific cleanup policy. They serve as:
- Historical record
- Debug reference
- Rollback option
