# Metadata Migration System

## Overview

Metaxy's migration system enables safe, automated updates to feature metadata when feature definitions change. Migrations are explicit, idempotent, and automatically propagate through the entire dependency graph.

## Core Concepts

### Feature Version

Every feature has a **feature version** - an 8-character hash of its complete specification:
- Feature key
- Container definitions and code versions
- Dependencies (feature-level and container-level)

```python
class VideoProcessing(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    containers=[
        ContainerSpec(key=ContainerKey(["frames"]), code_version=1),
        ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
    ],
)):
    pass

# Get feature version
print(VideoProcessing.feature_version())  # "a3f8b2c1"

# Change code_version
class VideoProcessing(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    containers=[
        ContainerSpec(key=ContainerKey(["frames"]), code_version=2),  # Changed!
        ContainerSpec(key=ContainerKey(["audio"]), code_version=1),
    ],
)):
    pass

print(VideoProcessing.feature_version())  # "d7e9f4a2" (different!)
```

Every metadata row includes a `feature_version` column:
```python
metadata = pl.DataFrame({
    "sample_id": [1, 2, 3],
    "data_version": [...],
    "feature_version": "a3f8b2c1",  # Auto-added by store
})
```

This enables:
- **Current vs historical**: Filter by `feature_version` to get specific versions
- **Migration tracking**: Know which rows need updating
- **Audit trail**: See what feature definition produced each row

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
    "containers": ["frames", "audio"],
    "snapshot_id": "9fee05a3",  # Graph snapshot hash (if recorded with record_all_feature_versions)
}
```

**Purpose:**
- Detection of latest version for migrations
- Audit trail of materializations
- Enables `detect_feature_changes()` to work

**Updated explicitly** by calling `store.record_feature_version(Feature)`.
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
store.record_feature_version(MyFeature)

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

The `snapshot_id` returned by `record_all_feature_versions()` is a deterministic 8-character hash
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
    containers=[
        ContainerSpec(key=ContainerKey(["default"]), code_version=1),
    ],
)):
    pass

# After: code_version = 2
class VideoProcessing(Feature, spec=FeatureSpec(
    containers=[
        ContainerSpec(key=ContainerKey(["default"]), code_version=2),  # Changed!
    ],
)):
    pass
```

### 2. Auto-Generate Migration

```bash
$ metaxy migrations generate

Scanning feature registry (15 features)...
Querying feature version history...

✓ Detected 2 feature changes:
  - video_processing: abc12345 → def67890 (latest: 2025-01-10 14:30:00)
  - audio_processing: xyz11111 → xyz22222 (latest: 2025-01-12 09:15:00)

✓ Will propagate to 5 downstream features:
  - feature_c (depends on video_processing)
  - feature_d (depends on audio_processing)
  - feature_e (depends on feature_c)
  - feature_f (depends on feature_d)
  - feature_g (depends on feature_e, feature_f)

Generated: migrations/20250113_103000_update_video_processing_audio_processing.yaml

NEXT STEPS:
1. Review and edit 'reason' fields
2. Run: metaxy migrations apply --dry-run
3. Run: metaxy migrations apply
4. Commit migration file to git
```

### 3. Review and Edit Generated Migration

```yaml
# migrations/20250113_103000_update_video_processing_audio_processing.yaml
version: 1
id: "migration_20250113_103000"
description: "Auto-generated migration for 2 changed features"
created_at: "2025-01-13T10:30:00Z"

operations:
  - type: feature_version_migration
    feature_key: ["video", "processing"]
    from: "abc12345"
    to: "def67890"
    change_type: "unknown"
    reason: "TODO: Describe what changed and why"  # ← EDIT THIS

  - type: feature_version_migration
    feature_key: ["audio", "processing"]
    from: "xyz11111"
    to: "xyz22222"
    change_type: "unknown"
    reason: "TODO: Describe what changed and why"  # ← EDIT THIS

# Downstream propagation is automatic (not listed explicitly)
```

**Edit the reasons:**

```yaml
operations:
  - type: feature_version_migration
    feature_key: ["video", "processing"]
    from: "abc12345"
    to: "def67890"
    change_type: "unknown"
    reason: "Updated frame extraction algorithm for better quality"

  - type: feature_version_migration
    feature_key: ["audio", "processing"]
    from: "xyz11111"
    to: "xyz22222"
    change_type: "unknown"
    reason: "Added support for AAC codec"
```

### 4. Dry-Run to Preview

```bash
$ metaxy migrations apply --dry-run

=== DRY RUN MODE ===
Would apply migration: migration_20250113_103000

Source features (2):
✓ video_processing
    From: abc12345 → To: def67890
    Rows to update: 3000
    Reason: Updated frame extraction algorithm for better quality

✓ audio_processing
    From: xyz11111 → To: xyz22222
    Rows to update: 1500
    Reason: Added support for AAC codec

Downstream propagation (5 features):
✓ feature_c
    Current rows: 5000
    Will recalculate based on new video_processing

✓ feature_d
    Current rows: 2000
    Will recalculate based on new audio_processing

✓ feature_e
    Current rows: 8000
    Will recalculate based on new feature_c

✓ feature_f
    Current rows: 3000
    Will recalculate based on new feature_d

✓ feature_g
    Current rows: 10000
    Will recalculate based on new feature_e, feature_f

Total impact:
  Source features: 2 (4500 rows)
  Downstream features: 5 (28000 rows)

No changes made (dry-run).
```

### 5. Apply Migration

```bash
$ metaxy migrations apply

Applying migration_20250113_103000...

[1/2] Migrating source features...
✓ video_processing: 3000 rows migrated (abc12345 → def67890)
✓ audio_processing: 1500 rows migrated (xyz11111 → xyz22222)

[2/2] Propagating to downstream features...
✓ feature_c: 5000 rows updated
✓ feature_d: 2000 rows updated
✓ feature_e: 8000 rows updated
✓ feature_f: 3000 rows updated
✓ feature_g: 10000 rows updated

Migration completed successfully in 45.2s
Affected 7 features, updated 32500 total rows.
```

### 6. Commit to Git

```bash
$ git add migrations/20250113_103000_update_video_processing_audio_processing.yaml
$ git commit -m "Migrate video and audio processing to new versions"
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
downstream_keys = store.registry.get_downstream_features(source_keys)

# Process in topological order (dependencies first)
for downstream_key in downstream_keys:
    feature = registry.features_by_key[downstream_key]

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

Migrations are **automatically recoverable** from partial failures:

```bash
# First attempt - fails midway
$ metaxy migrations apply

Applying migration_20250113_103000...
✓ video_processing: 3000 rows migrated
✓ feature_c: 5000 rows updated
✗ feature_d: FAILED (connection timeout)

Migration marked as partial.
Re-run to continue from failure point.

# Second attempt - picks up where it left off
$ metaxy migrations apply

Applying migration_20250113_103000...
  Checking video_processing...
    No rows with old feature_version found (skipped) ✓
  Checking feature_c...
    Already has correct data_versions (skipped) ✓
  Retrying feature_d...
✓ feature_d: 2000 rows updated
✓ feature_e: 8000 rows updated
✓ feature_f: 3000 rows updated

Migration completed successfully!
```

**How recovery works:**
1. Re-run queries for rows with old `feature_version`
2. If not found: already migrated, skip
3. If found: migrate them
4. Eventually all rows migrated, migration marked completed
5. Future runs are instant no-ops

### Idempotency

Migrations can be safely re-run multiple times:

```python
# Idempotency check
def _is_migration_completed(store: MetadataStore, migration_id: str) -> bool:
    """A migration is complete when:
    1. Marked as 'completed' in migration history, AND
    2. No rows with old feature_version exist for any source feature
    """
    # Check migration history
    try:
        history = store.read_metadata(
            FeatureKey(["__metaxy__", "migrations"]),
            filters=pl.col("migration_id") == migration_id,
        )
        if len(history) > 0 and history["status"][0] != "completed":
            return False  # Partial or failed, not complete
    except FeatureNotFoundError:
        return False

    # Double-check: verify no old rows exist
    for operation in migration.operations:
        old_rows = store.read_metadata(
            FeatureKey(operation.feature_key),
            current_only=False,
            filters=pl.col("feature_version") == operation.from_feature_version,
        )
        if len(old_rows) > 0:
            return False  # Still has old rows

    return True  # Fully complete
```

## User-Defined Metadata Alignment

When feature definitions change, you may need custom logic to align metadata with upstream changes. Override `align_metadata_with_upstream()`:

### Default Behavior (Inner Join)

```python
class MyFeature(Feature, spec=...):
    # Default behavior: inner join on sample_id
    # Only keeps samples present in ALL upstream features
    pass

# During migration:
# - Reads current metadata
# - Inner joins with upstream on sample_id
# - Only samples present in both current AND upstream survive
```

### Custom Alignment Examples

#### One-to-Many Mapping

```python
class VideoFrames(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "frames"]),
    deps=[FeatureDep(key=FeatureKey(["videos"]))],
    ...
)):
    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Each video produces 30 frames.

        When videos upstream change, regenerate all frame sample IDs.
        """
        video_samples = upstream_metadata["videos"]["sample_id"]

        # Generate frame sample IDs
        frames = []
        for video_id in video_samples:
            for frame_idx in range(30):
                frames.append({
                    "sample_id": f"{video_id}_frame_{frame_idx}",
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                })

        return pl.DataFrame(frames)
```

#### Conditional Filtering

```python
class ProcessedVideos(Feature, spec=...):
    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Only process videos longer than 10 seconds."""

        videos = upstream_metadata["videos"]

        # Filter by duration
        valid_videos = videos.filter(pl.col("duration") > 10)

        # Keep custom columns from current_metadata if available
        if len(current_metadata) > 0:
            return current_metadata.join(
                valid_videos.select(pl.col("sample_id")),
                on="sample_id",
                how="inner",  # Only keep samples that pass filter
            )
        else:
            return valid_videos.select(["sample_id", "duration", "path"])
```

#### Outer Join (Keep All Samples)

```python
class MergedFeature(Feature, spec=...):
    @classmethod
    def align_metadata_with_upstream(
        cls,
        current_metadata: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Keep union of all upstream samples (outer join)."""

        all_samples = set()
        for upstream_df in upstream_metadata.values():
            all_samples.update(upstream_df["sample_id"].to_list())

        return pl.DataFrame({"sample_id": sorted(all_samples)})
```

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

## CLI Commands

### Generate Migration

```bash
# Auto-detect all changes
$ metaxy migrations generate

# Specify output directory
$ metaxy migrations generate --output-dir my_migrations

# Specify author
```

### Apply Migrations

```bash
# Dry-run first (recommended)
$ metaxy migrations apply --dry-run

# Apply for real
$ metaxy migrations apply

# Apply from custom directory
$ metaxy migrations apply --migrations-dir my_migrations

# Force re-apply (ignore completion status)
$ metaxy migrations apply --force
```

### Check Status

```bash
$ metaxy migrations status

Migration Status:
┌─────────────────────────────┬───────────┬─────────────────────┐
│ Migration ID                │ Status    │ Applied At          │
├─────────────────────────────┼───────────┼─────────────────────┤
│ migration_20250110_120000   │ completed │ 2025-01-10 12:05:00 │
│ migration_20250113_103000   │ partial   │ 2025-01-13 10:35:00 │
│ migration_20250115_160000   │ pending   │ -                   │
└─────────────────────────────┴───────────┴─────────────────────┘

⚠ Migration migration_20250113_103000 is incomplete.
  Run 'metaxy migrations apply' to continue.
```

## Python API

For programmatic usage:

```python
from metaxy.migrations import (
    detect_feature_changes,
    generate_migration,
    apply_migration,
    apply_migrations_from_directory,
    Migration,
    MigrationResult,
)

# Auto-detect changes
changes = detect_feature_changes(store)
print(f"Detected {len(changes)} changes")

# Generate migration
migration_file = generate_migration(
    store,
    output_dir="migrations",
)

# Load migration
migration = Migration.from_yaml(migration_file)

# Apply with dry-run
result = apply_migration(store, migration, dry_run=True)
print(result.summary())

# Apply for real
result = apply_migration(store, migration)
if result.status == "completed":
    print("Success!")
else:
    print(f"Errors: {result.errors}")

# Apply all migrations from directory
results = apply_migrations_from_directory(store, "migrations")
for result in results:
    print(f"{result.migration_id}: {result.status}")
```

## Key Design Principles

### 1. Always Propagate Downstream

Every migration **automatically cascades** through the entire dependency graph. You never need to manually specify downstream features - Metaxy computes the full DAG and updates everything.

### 2. Idempotent by Design

Migrations can be safely re-run:
- Check migration history for completion status
- Query for rows with old feature_version
- If none found: skip (already done)
- If found: migrate them

### 3. Automatic Recovery from Failures

If a migration fails midway:
- Some features migrated (new feature_version)
- Some not migrated (old feature_version)
- Re-running picks up where it left off
- Eventually converges to completion

No manual intervention needed!

### 4. Never Inconsistent

The metadata store is **never left in an inconsistent state**:
- Old rows with old feature_version remain valid
- New rows with new feature_version are valid
- Re-running migration brings all rows to new version
- During migration: both versions coexist (immutable)

### 5. Explicit and Auditable

- Migrations are explicit YAML files in version control
- Each migration documents what changed and why
- Migration history tracked in metadata store
- Complete audit trail

## Advanced Scenarios

### Multiple Old Versions

If a feature has multiple old versions in metadata:

```python
# Metadata contains rows with:
# - feature_version="v1" (100 rows, materialized 2025-01-01)
# - feature_version="v2" (200 rows, materialized 2025-01-05)
# - feature_version="v3" (500 rows, materialized 2025-01-10)

# Code now has feature_version="v4"

# Migration generator automatically uses LATEST (v3 → v4)
# Older versions (v1, v2) remain in metadata unchanged
# This is fine! They represent historical states.

# If you want to migrate ALL old versions, generate separate migrations:
# Migration 1: v1 → v4
# Migration 2: v2 → v4
# Migration 3: v3 → v4
```

### Custom Migration Logic

For complex scenarios, you can implement custom migration functions:

```python
# migrations/custom.py

def migrate_video_format_change(
    store: MetadataStore,
    operation: FeatureVersionMigration,
) -> None:
    """Custom migration for video format change.

    When migrating from old to new video codec, we need to:
    1. Re-extract frames
    2. Update paths
    3. Recalculate checksums
    """
    # Custom logic here
    old_metadata = store.read_metadata(
        FeatureKey(operation.feature_key),
        current_only=False,
        filters=pl.col("feature_version") == operation.from_feature_version,
    )

    # Transform metadata
    new_metadata = old_metadata.with_columns([
        pl.col("path").str.replace(".mp4", ".mkv"),
        # ... other transformations
    ])

    # Recalculate and write
    result = store.calculate_and_write_data_versions(
        FeatureKey(operation.feature_key),
        new_metadata.drop("data_version", "feature_version"),
    )
```

```yaml
# Use in migration file
operations:
  - type: custom_migration
    feature_key: ["video", "processing"]
    from: "abc12345"
    to: "def67890"
    python_function: "migrations.custom.migrate_video_format_change"
    reason: "Changed video format from MP4 to MKV"
```

## Querying Metadata with Versioning

### Read Current Version (Default)

```python
# Get current metadata (default: current_only=True)
current = store.read_metadata(VideoProcessing)
# Only returns rows with current feature_version
```

### Read Historical Versions

```python
# Get all versions
all_versions = store.read_metadata(VideoProcessing, current_only=False)

# Get specific old version
old_version = store.read_metadata(
    VideoProcessing,
    current_only=False,
    filters=pl.col("feature_version") == "abc12345",
)

# Compare versions
current = store.read_metadata(VideoProcessing, current_only=True)
previous = store.read_metadata(
    VideoProcessing,
    current_only=False,
    filters=pl.col("feature_version") == "abc12345",
)

# See what changed
print(f"Current samples: {len(current)}")
print(f"Previous samples: {len(previous)}")
```

### Query Feature Version History

```python
# Get all materialized versions for a feature
version_history = store.read_metadata(
    FeatureKey(["__metaxy__", "feature_versions"]),
    filters=pl.col("feature_key") == "video_processing",
).sort("recorded_at", descending=True)

print(version_history)
# Shows: feature_version, recorded_at, snapshot_id for each version
```

### Query Migration History

```python
# Get all applied migrations
migration_history = store.read_metadata(
    FeatureKey(["__metaxy__", "migrations"])
).sort("applied_at", descending=True)

# Get specific migration
migration_info = store.read_metadata(
    FeatureKey(["__metaxy__", "migrations"]),
    filters=pl.col("migration_id") == "migration_20250113_103000",
)
```

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

If you need cleanup:
```python
# Delete very old versions (6+ months old)
# This is optional - only if storage is a concern
def cleanup_old_versions(store: MetadataStore, feature: type[Feature], keep_recent: int = 3):
    """Keep only N most recent feature_versions."""
    # Get version history
    version_history = store.read_metadata(
        FeatureKey(["__metaxy__", "feature_versions"]),
        filters=pl.col("feature_key") == feature.spec.key.to_string(),
    ).sort("recorded_at", descending=True)

    # Identify versions to delete (older than keep_recent)
    if len(version_history) > keep_recent:
        old_versions = version_history[keep_recent:]["feature_version"].to_list()

        # Delete old rows (backend-specific, may not be supported)
        for old_version in old_versions:
            store.delete_metadata(
                feature,
                filters=pl.col("feature_version") == old_version,
            )
```

## Implementation Notes

### System Table Schema

System tables use a special naming pattern to avoid conflicts:

```python
# System table keys start with "__metaxy__"
SYSTEM_PREFIX = ["__metaxy__"]

FEATURE_VERSIONS_KEY = FeatureKey(SYSTEM_PREFIX + ["feature_versions"])
MIGRATION_HISTORY_KEY = FeatureKey(SYSTEM_PREFIX + ["migrations"])

# System tables are treated specially:
# - No feature_version filtering applied
# - No automatic feature_version column added
# - Must be explicitly defined in registry (as special Features)
```

### Hash Algorithm

Feature versions use **SHA256 truncated to 8 characters**:

- Same as git short hashes (familiar)
- 8 chars = 4.3 billion combinations (sufficient)
- SHA256 is available everywhere (Python, JS, Java, Rust, SQL)
- Deterministic and collision-resistant

### Topological Sorting

Downstream features are processed in dependency order:

```python
# Example DAG:
# A → B → D
# A → C → D

# Processing order: [A, B, C, D]
# - A first (source)
# - B and C next (depend only on A)
# - D last (depends on B and C)
```

This ensures:
- Dependencies available when needed
- No circular dependency issues
- Predictable, deterministic ordering

## Future Enhancements

### 1. Parallel Execution

For independent features in DAG:
```python
# B and C don't depend on each other, can run in parallel
apply_migration(store, migration, parallel=True)
```

### 2. Migration Rollback

Explicitly rollback to previous version:
```python
rollback_migration(store, migration_id="migration_20250113_103000")
# Migrates back: new_version → old_version
```

### 3. Conflict Detection

Detect if metadata changed between migration generation and application:
```python
# Migration generated on 2025-01-13 10:30
# Applied on 2025-01-13 11:00
# If metadata changed in between, warn user
```

### 4. Progress Callbacks

For long-running migrations:
```python
def progress_callback(feature_key: str, progress: float):
    print(f"{feature_key}: {progress*100:.1f}%")

apply_migration(store, migration, on_progress=progress_callback)
```

## Troubleshooting

### Migration Stuck in "Partial" Status

```bash
# Check what's blocking
$ metaxy migrations status --verbose

Migration migration_20250113_103000: partial
Last applied: 2025-01-13 10:35:00
Completed operations: 3/7
Failed at: feature_d (Connection timeout)

# Re-run to continue
$ metaxy migrations apply
```

### Feature Version Mismatch

```bash
# Code has feature_version="new123" but metadata has "old456"
# No migration exists

$ metaxy migrations generate
# Creates migration automatically

$ metaxy migrations apply
# Applies migration
```

### Custom Alignment Failing

```python
# Debug alignment logic
class MyFeature(Feature, spec=...):
    @classmethod
    def align_metadata_with_upstream(cls, current, upstream):
        print(f"Current samples: {len(current)}")
        print(f"Upstream samples: {list(upstream.keys())}")

        # Your logic here
        result = ...

        print(f"Aligned samples: {len(result)}")
        return result
```

## See Also

- [Metadata Store Tutorial](metadata_store.md) - Core metadata store concepts
- [Sample Data Version](sample_data_version.md) - How data versions are computed
- [Roadmap](roadmap.md) - Future enhancements
