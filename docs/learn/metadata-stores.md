# Metadata Stores

Metaxy abstracts interactions with metadata stored in external systems such as databases, files, or object stores, through a unified interface: [`MetadataStore`][metaxy.MetadataStore].

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

## Metadata Store Implementations

Metaxy provides ready `MetadataStore` [implementations](../integrations/metadata-stores] for popular databases and storage systems.

## Configuration

Learn about configuring metadata stores [here](../reference/configuration.md/#stores)
