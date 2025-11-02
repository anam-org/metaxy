# Metadata Stores

Metaxy abstracts interactions with metadata stored in external systems such as databases, files, or object stores, through a unified interface: [MetadataStore][metaxy.MetadataStore].

Metadata stores expose methods for [reading][metaxy.MetadataStore.read_metadata], [writing][metaxy.MetadataStore.write_metadata], deleting metadata, and the most important one: [resolve_update][metaxy.MetadataStore.resolve_update] for receiving a metadata increment.
Metaxy intentionally does not support mutating metadata in-place for performance reasons.
Deletes are not required during normal operations, but they are still supported since users would want to eventually delete stale metadata and data.

Metadata reads/writes **are not guaranteed to be ACID**: Metaxy is designed to interact with analytical databases which lack ACID guarantees by definition and design (for performance reasons).
However, Metaxy guarantees to never attempt to retrieve the same sample version twice, so as long as users do not write it twice (or have deduplication configured inside the metadata store) we should be all good.

When resolving incremental updates for a [feature](feature-definitions.md), Metaxy attempts to perform all computations such as [sample version calculations](data-versioning.md) within the metadata store.
This includes joining upstream features, hashing their versions, and filtering out samples that have already been processed.

There are 3 cases where this is done in-memory instead (with the help of [polars-hash](https://github.com/ion-elgreco/polars-hash)):

1. The metadata store does not have a compute engine at all: for example, [DeltaLake](https://delta.io/) is just a storage format.
2. The user explicitly requested to keep the computations in-memory (`MetadataStore(..., prefer_native=False)`)
3. When having to use a **fallback store** to retrieve one of the parent features.

All 3 cases cannot be accidental and require preconfigured settings or explicit user action. In the third case, Metaxy will also issue a warning just in case the user has accidentally configured a fallback store in production.

Learn about configuring metadata stores [here](../reference/configuration.md/#storeconfig)

## Fallback Stores

Fallback stores are a powerful feature that allow stores to read feature metadata from other stores (only if it's missing in the primary store).
This is very useful for development, as production data can be retrieved immediately without populating the development environment.
This is especially useful for ephemeral environments such as branch/preview deployments (typically created by CI/CD for pull requests) or integration testing environments.

## Project Write Validation

By default, `MetadataStore` raises a `ValueError` when attempting to write to a project that doesn't match the expected project from `MetaxyConfig.get().project`.

For legitimate cross-project operations (such as migrations that need to update features across multiple projects), an escape hatch is provided via the `allow_cross_project_writes()` context manager:

```python
# Normal operation - writes are validated against expected project
with store:
    store.write_metadata(feature_from_my_project, metadata)  # OK
    store.write_metadata(feature_from_other_project, metadata)  # Raises ValueError

# Migration scenario - temporarily allow cross-project writes
with store:
    with store.allow_cross_project_writes():
        store.write_metadata(feature_from_project_a, metadata_a)  # OK
        store.write_metadata(feature_from_project_b, metadata_b)  # OK
```
