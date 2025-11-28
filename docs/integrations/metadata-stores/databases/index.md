# Database-Backed Metadata Stores

These metadata stores provide external compute resources.
The most common example of such stores is databases.
Metaxy delegates all versioning computations and operations to external compute as much as possible (typically the entire [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] can be executed externally).
