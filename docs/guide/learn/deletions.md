# Metadata Deletion

Metaxy supports two deletion modes for managing metadata lifecycle:

- **Soft deletes** (default): Mark records as deleted while preserving history
- **Hard deletes**: Permanently remove records from storage

All examples assume `import narwhals as nw` and appropriate store access.

## Deletion modes

### Soft deletes

Soft deletes mark records as deleted without physically removing them. When you call `delete_metadata`, Metaxy appends a new row with the `metaxy_deleted_at` [system column](./system-columns.md) set to the deletion timestamp. This preserves full history while hiding deleted records from normal queries.

By default, `read_metadata` automatically filters out soft-deleted records. You can opt-in to see deleted data using `include_soft_deleted=True`.

### Hard deletes

Hard deletes permanently remove rows from storage. Use them for compliance requirements, reclaiming space, or when history is not needed. Pass `soft=False` to `delete_metadata`.

Not all metadata stores support hard deletes. Stores without support will raise `NotImplementedError`.

## Python API

### Basic deletion

Delete specific records using filter expressions:

```python
# Soft delete (default)
with store.open("write"):
    store.delete_metadata(
        PredictionFeature,
        filters=nw.col("model_version") == "v1",
    )

# Hard delete
with store.open("write"):
    store.delete_metadata(
        PredictionFeature,
        filters=nw.col("confidence") < 0.5,
        soft=False,
    )

# Read with different delete visibility
with store:
    active = store.read_metadata(PredictionFeature).collect()
    all_data = store.read_metadata(PredictionFeature, include_soft_deleted=True).collect()
```

### Topological (cascade) deletion

When working with dependent features (such as [expansion relationships](./relationship.md)), you need to delete features in the correct order to maintain referential integrity. Metaxy automatically determines deletion order based on your feature dependency graph.

**Example scenario:** Video processing pipeline with expansion (1:N) relationships:

```python
class VideoRaw(
    BaseFeature,
    spec=FeatureSpec(
        key=["video", "raw"],
        id_columns=["video_id"],
        fields=["frames", "audio"],
    ),
):
    video_id: str


class VideoChunk(
    BaseFeature,
    spec=FeatureSpec(
        key=["video", "chunk"],
        id_columns=["chunk_id"],
        deps=[
            FeatureDep(
                feature=VideoRaw,
                lineage=LineageRelationship.expansion(on=["video_id"]),
            )
        ],
        fields=["frames"],
    ),
):
    video_id: str
    chunk_id: str


class FaceRecognition(
    BaseFeature,
    spec=FeatureSpec(
        key=["video", "faces"],
        id_columns=["chunk_id"],
        deps=[FeatureDep(feature=VideoChunk)],
        fields=["faces"],
    ),
):
    chunk_id: str
```

**Get deletion order and execute:**

```python
# Get deletion order (leaf-first for hard deletes)
with store.open("read"):
    order = store.get_deletion_order(
        VideoRaw,
        direction="downstream",  # dependents first (leaf → root)
        include_self=True,
    )
    # Returns: [FaceRecognition, VideoChunk, VideoRaw]

# Execute hard delete in correct order
with store.open("write"):
    for feature_key in order:
        store.delete_metadata(
            feature_key,
            filters=nw.col("video_id") == "vid_123",
            soft=False,
        )
```

**Direction options:**

- `downstream`: Returns dependents first (leaf → root). Use for hard deletions to avoid foreign key violations.
- `upstream`: Returns dependencies first (root → leaf). Less common for deletions.
- `both`: Returns all connected features in dependency order.

**Partial deletions** (delete individual chunks without affecting the parent):

```python
with store.open("write"):
    store.delete_metadata(
        VideoChunk,
        filters=nw.col("chunk_id") == "chunk_5",
        soft=False,
    )
    # Parent video remains intact
```

## CLI

The `metaxy metadata delete` command provides both basic and cascade deletion workflows.

### Basic deletion

```bash
# Soft delete (default)
metaxy metadata delete --feature predictions --filter "confidence < 0.3"

# Hard delete with confirmation
metaxy metadata delete \
    --feature predictions \
    --filter "created_at < '2024-01-01'" \
    --soft=false \
    --yes
```

### Cascade deletion

Use `--cascade` to automatically delete dependent or dependency features:

```bash
# Preview deletion plan (dry-run)
metaxy metadata delete \
    --feature video/raw \
    --cascade downstream \
    --dry-run

# Execute soft cascade deletion
metaxy metadata delete \
    --feature video/raw \
    --filter "video_id='v1'" \
    --cascade downstream

# Execute hard cascade deletion
metaxy metadata delete \
    --feature video/raw \
    --cascade downstream \
    --soft=false \
    --yes
```

**Cascade options:**

| Option           | Behavior                                                  |
| ---------------- | --------------------------------------------------------- |
| `none` (default) | Delete only the specified feature                         |
| `downstream`     | Delete dependents first, then the feature (leaf → root)   |
| `upstream`       | Delete dependencies first, then the feature (root → leaf) |
| `both`           | Delete all connected features in dependency order         |

The `--dry-run` flag previews which features would be deleted and in what order without executing the operation.

## Dagster integration

The `delete_metadata` op supports both basic and cascade deletion workflows in Dagster pipelines. For more about the Dagster integration, see the [Dagster Integration Guide](../../integrations/orchestration/dagster/index.md).

### Basic deletion

```python
import dagster as dg
from metaxy.ext.dagster import delete_metadata, MetaxyStoreFromConfigResource


@dg.job(resource_defs={"metaxy_store": MetaxyStoreFromConfigResource(name="default")})
def cleanup_job():
    delete_metadata()


# Execute deletion
cleanup_job.execute_in_process(
    run_config={
        "ops": {
            "delete_metadata": {
                "config": {
                    "feature_key": ["customer", "segment"],
                    "filters": ["status = 'inactive'"],
                    "soft": True,
                }
            }
        }
    }
)
```

### Cascade deletion

Add the `cascade` parameter for topological deletions:

```python
cleanup_job.execute_in_process(
    run_config={
        "ops": {
            "delete_metadata": {
                "config": {
                    "feature_key": ["video", "raw"],
                    "filters": ["video_id = 'v1'"],
                    "soft": True,
                    "cascade": "downstream",  # Delete dependent features first
                }
            }
        }
    }
)
```

**Configuration options:**

- `feature_key`: Feature to delete (list of strings)
- `filters`: SQL WHERE clause strings (see [filter expressions](./filters.md))
- `soft`: Soft delete (`true`) or hard delete (`false`)
- `cascade`: Deletion direction (`"none"`, `"downstream"`, `"upstream"`, `"both"`)

The op logs which features are deleted and in what order.

## Related materials

- [Expansion Relationships Example](../../examples/expansion.md) - Complete video processing pipeline
- [Filter Expressions](./filters.md) - Syntax for deletion filters
- [System Columns](./system-columns.md) - Understanding `metaxy_deleted_at`
