---
title: "Deletions"
description: "Learn how to perform soft and hard deletions on feature metadata."
---

# Metadata Deletion

[`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] can be used to identify orphaned samples that no longer exist upstream. Users may want to delete these samples for the current feature from the metadata store.

Metaxy supports two deletion modes: **soft deletes** that preserve history and **hard deletes** that permanently remove records. Soft deletes are the default behavior and preferred for most use cases since they maintain audit trails while allowing records to be filtered out from normal queries.

## Soft deletes

Soft deletes mark records as deleted without physically removing them. When you call `delete`, Metaxy appends a new row with the `metaxy_deleted_at` [system column](../../reference/system-columns.md) set to the deletion timestamp. This preserves your full history—nothing is lost, and you can always query for soft-deleted records if needed.

By default, `read` filters out soft-deleted records automatically. You only see active data. Behind the scenes, Metaxy keeps the latest version of each record by coalescing deletion and creation timestamps, so even if you've updated a record multiple times, queries return only the current state.

```python
import narwhals as nw

with store.open("w"):
    store.delete(
        MyFeature,
        filters=nw.col("status") == "pending",
    )

with store:
    active = store.read(MyFeature)
    all_rows = store.read(MyFeature, include_soft_deleted=True)
```

If you track custom deletion flags in your feature schema, filter them through `read` filters.

## Hard deletes

Hard deletes permanently remove rows from storage. Use them when you need to physically delete data, such as for compliance requirements or to reclaim space. Pass `soft=False` to `delete` and specify which records to remove with a filter expression.

```python
import narwhals as nw

with store.open("w"):
    store.delete(
        MyFeature,
        filters=nw.col("quality") < 0.8,
        soft=False,
    )
```

Not all metadata stores support hard deletes. If your store doesn't support them, you'll get a `NotImplementedError`.

## Topological (cascade) deletion

When working with dependent features (such as [expansion relationships](definitions/relationship.md)), you need to delete features in the correct order to maintain referential integrity. Metaxy automatically determines deletion order based on your feature dependency graph.

**Direction options:**

- `downstream`: Returns dependents first (leaf → root). Use for hard deletions to avoid foreign key violations.
- `upstream`: Returns dependencies first (root → leaf). Less common for deletions.
- `both`: Returns all connected features in dependency order.

## CLI workflows

Run cleanups from the command line using `metaxy metadata delete`:

```bash
# Soft delete by default
metaxy metadata delete predictions --filter "confidence < 0.3"

# Hard delete
metaxy metadata delete predictions --filter "created_at < '2024-01-01'" --hard --yes
```

### Cascade deletion

Use `--cascade` to automatically delete dependent or dependency features:

```bash
# Preview deletion plan (dry-run)
metaxy metadata delete video/raw --cascade downstream --dry-run

# Execute soft cascade deletion
metaxy metadata delete video/raw --filter "video_id='v1'" --cascade downstream

# Execute hard cascade deletion
metaxy metadata delete video/raw --cascade downstream --hard --yes
```

| Option           | Behavior                                                  |
| ---------------- | --------------------------------------------------------- |
| `none` (default) | Delete only the specified feature                         |
| `downstream`     | Delete dependents first, then the feature (leaf → root)   |
| `upstream`       | Delete dependencies first, then the feature (root → leaf) |
| `both`           | Delete both dependencies and dependents                   |

The `--dry-run` flag previews which features would be deleted and in what order without executing the operation.

## Dagster integration

The `delete` op supports both basic and cascade deletion workflows in Dagster pipelines. For more about the Dagster integration, see the [Dagster Integration Guide](../../integrations/orchestration/dagster/index.md).

<!-- skip next -->
```python
import dagster as dg
import metaxy.ext.dagster as mxd


@dg.job(resource_defs={"metaxy_store": mxd.MetaxyStoreFromConfigResource(name="default")})
def cleanup_job():
    mxd.delete()


# Execute with cascade deletion
cleanup_job.execute_in_process(
    run_config={
        "ops": {
            "delete": {
                "config": {
                    "feature_key": ["video", "raw"],
                    "filters": ["video_id = 'v1'"],
                    "soft": True,
                    "cascade": "DOWNSTREAM",
                }
            }
        }
    }
)
```

## Related materials

- [Expansion Relationships Example](../../examples/expansion.md) - Complete video processing pipeline
- [Filter Expressions](filters.md) - Syntax for deletion filters
- [System Columns](../../reference/system-columns.md) - Understanding `metaxy_deleted_at`
