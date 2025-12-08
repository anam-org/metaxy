# Metadata Cleanup (hard & soft deletes)

Metaxy gives you a small set of primitives for metadata lifecycle:

- `read_metadata` to fetch active rows (soft-deleted rows are hidden by default)
- `delete_metadata` to delete rows (soft by default, hard when `soft=False`)

All examples assume `import narwhals as nw` and a store opened in the right mode.

## Soft deletes

`metaxy_deleted_at` is reserved for built-in filtering. Soft deletes are append-only: we add a new row with `metaxy_deleted_at` set and leave history intact. Reads hide soft-deleted rows unless you opt in.

```python
from datetime import datetime, timezone

with store.open("write"):
    store.delete_metadata(
        MyFeature,
        filters=nw.col("user_id") == "user_123",
    )

with store:
    active = store.read_metadata(MyFeature).collect()  # soft-deleted hidden
    all_rows = store.read_metadata(MyFeature, include_deleted=True).collect()
    only_deleted = store.read_metadata(MyFeature, with_soft_deleted=True).collect()
```

If you already track a custom deletion flag, you can keep using it and filter via `read_metadata` filters. The built-in column is for the default behavior and convenience.

## Hard deletes

`delete_metadata` permanently removes rows when `soft=False`. Filter with a Narwhals expression (single or combined via `&`).

```python
with store.open("write"):
    store.delete_metadata(
        FeatureKey(["user", "sessions"]),
        filters=nw.col("status") == "inactive",
        soft=False,
    )
```

Backends currently supported for hard delete: in-memory, ibis-based stores (e.g., DuckDB, ClickHouse), Delta, and LanceDB. Unsupported backends will raise `NotImplementedError`.

## CLI workflows

Use the CLI to run cleanups without writing code:

```bash
# Soft delete by default
metaxy metadata delete --feature logs --filter "level = 'warn'"

# Retention-based hard delete
metaxy metadata delete --feature logs --retention-days 90 --mode hard
```

The CLI requires `--filter` (or `--retention-days`) and a feature selection (`--feature` or `--all-features`).

## Dagster ops

For Dagster users, the `metaxy.ext.dagster` package exposes a `delete_metadata` op:

```python
import dagster as dg
import metaxy.ext.dagster as mxd


@dg.job(resource_defs={"store": mxd.MetaxyStoreFromConfigResource(name="default")})
def cleanup_job():
    mxd.delete_metadata(
        feature_key=["logs"],
        retention_days=90,
        hard=False,  # soft delete
    )
```

`retention_days` builds a time-based filter; alternatively pass `filter_expr` (string, evaluated as Narwhals expression) to target specific rows. Use `hard=True` to physically remove rows.
