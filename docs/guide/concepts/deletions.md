---
title: "Deletions"
description: "Learn how to perform soft and hard deletions on feature metadata."
---

# Metadata Deletion

Metaxy supports two deletion modes: **soft deletes** that preserve history and **hard deletes** that permanently remove records. Soft deletes are the default behavior and preferred for most use cases since they maintain audit trails while allowing records to be filtered out from normal queries.

## Soft deletes

Soft deletes mark records as deleted without physically removing them. When you call `delete`, Metaxy appends a new row with the `metaxy_deleted_at` [system column](./system-columns.md) set to the deletion timestamp. This preserves your full history—nothing is lost, and you can always query for soft-deleted records if needed.

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

## CLI workflows

Run cleanups from the command line using `metaxy metadata delete`:

```bash
# Soft delete by default
metaxy metadata delete --feature predictions --filter "confidence < 0.3"

# Hard delete
metaxy metadata delete --feature predictions --filter "created_at < '2024-01-01'" --soft=false
```

Learn more in the [CLI reference](../../reference/cli.md#metaxy-metadata-delete)
