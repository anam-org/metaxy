---
title: "SQL Views"
description: "Expose current feature rows and feature statistics as SQL views for readers without the Metaxy library."
---

# SQL Views

Metaxy tables keep the full history of every sample: rows of older feature versions, superseded rows and soft-deleted rows. [`MetadataStore.read`][metaxy.MetadataStore.read] hides this history, but services written in other languages, dashboards and SQL tools read the tables directly. To give them the same answer as the Python library, Metaxy can create SQL views in the metadata store.

```shell
mx views sync
```

The command creates three kinds of views:

| View | Contents |
|---|---|
| `<table>__current` | One view per internal feature. It returns the same rows as `read()` with default arguments: only the current feature version, only the latest row per sample, and no soft-deleted rows. The view contains the feature's own columns and `metaxy_data_version`. |
| `metaxy_feature_statistics` | One row per internal feature with its feature version, row count, the missing, stale and orphaned counts of `mx metadata status`, and the time of the last write. The database computes the counts when the view is queried, so they are always up to date. |
| `metaxy_views_manifest` | One row per view with its name, feature key, feature version, id columns, column names and types, and a hash of the view definition. Readers can use it to generate typed models. |

External features do not get a view, and a feature gets a view only once its table exists in the store.

!!! example "Reading current rows from SQL"

    ```sql
    SELECT sample_uid, path, metaxy_data_version
    FROM video__files__current
    ```

## Keeping Views Up to Date

Every view is pinned to the feature versions of the feature graph it was created from. When a feature version changes, run `mx views sync` again to replace the views; readers keep using the same view names. A natural place for this is the deployment pipeline, right after [`mx push`](lifecycle/deployment.md).

The command supports a few options for automation:

- `--dry-run` prints the `CREATE OR REPLACE VIEW` statements without executing them.
- `--check` exits with an error when a view is missing or out of date, which is useful in CI.
- `--manifest PATH` also writes the manifest to a local JSON file.
- `--schema NAME` creates the views in another schema, or another database on ClickHouse.
- `--feature KEY` limits the update to the current views of the given features.

!!! note "Supported stores"

    Views require a store that computes increments natively with Ibis, such as ClickHouse and DuckDB. The statistics view compiles the same queries that `resolve_update` runs, so the counts match `mx metadata status` exactly.
