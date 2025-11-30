# Data Cleanup

Metaxy exposes a small set of primitives for metadata lifecycle: `read_metadata` to fetch active or soft-deleted rows, `delete_metadata` for hard deletes, and `mutate_metadata` for in-place updates (including soft-delete style flags).

Examples below assume `import narwhals as nw` and `from datetime import datetime, timezone`.

## Soft-delete strategy

Metaxy reserves the `metaxy_deleted_at` column for built-in filtering.
If you want soft deletes, set that column via `mutate_metadata` (or the `soft_delete_metadata` convenience) and read with the options above.
If you already track deletion state in a different column, keep using it—`mutate_metadata` lets you update any fields and you can add your own filters when reading.

Reading after soft deletes

`read_metadata` filters out rows with `metaxy_deleted_at` by default.

- Use `include_deleted=True` to return active and soft-deleted rows together - useful for undeleting
- Use `with_soft_deleted=True` to return only soft-deleted rows - useful to pass deletions to a downstream system or audits

```python
with store:
    deleted = store.read_metadata(
        MyFeature,
        with_soft_deleted=True,
    ).collect()
```

## Hard delete: `delete_metadata`

Permanently removes matching records.
The `filter` argument accepts either a Narwhals expression or a frame of values; when passing a frame, set `match_on` to `"id_columns"` (default), `"all_columns"`, or a list of column names to control the match.

```python
with store.open("write"):
    store.delete_metadata(
        FeatureKey(["user", "sessions"]),
        filter=nw.col("status") == "inactive",
    )

ids = nw.from_dict({"user_id": ["u1", "u2"]})
with store.open("write"):
    store.delete_metadata(
        FeatureKey(["user", "profile"]),
        filter=ids,
        match_on="id_columns",
    )
```

## Mutate metadata: `mutate_metadata`

Update columns for matching rows—useful for anonymization, flags, or implementing your own soft-delete column. The same `filter`/`match_on` semantics as `delete_metadata` apply.

```python
with store.open("write"):
    store.mutate_metadata(
        FeatureKey(["user", "profile"]),
        filter=nw.col("user_id") == "user_123",
        updates={
            "email": "[REDACTED]",
            "metaxy_deleted_at": nw.lit(datetime.now(timezone.utc)),
        },
    )
```

## Asset-style cleanup flow

During orchestration, keep cleanup explicit: let `resolve_update` tell you what to add/change/remove, materialize the first two, then delete the rest (soft or hard).

```python
import dagster as dg


@dg.asset
def my_feature(store):
    added, changed, to_delete = store.resolve_update(MyFeature)

    materialize(added, changed)  # write data + metadata for new/updated rows
    store.delete_metadata(
        MyFeature, filter=to_delete
    )  # or soft_delete_metadata(..., filter=to_delete)

    return {"added": len(added.collect()), "deleted": len(to_delete.collect())}
```

For retention-style jobs, use the CLI helper instead of hand-writing the filters:

```
metaxy metadata cleanup --feature logs --retention-days 90 --confirm  # soft-delete by default
metaxy metadata cleanup --feature logs --retention-days 90 --hard --confirm
```
