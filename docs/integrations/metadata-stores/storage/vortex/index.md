# Metaxy + Vortex

The [`VortexMetadataStore`][metaxy.metadata_store.vortex.VortexMetadataStore] stores metadata in [Vortex](https://github.com/vortex-data/vortex) files.
It uses an in-memory Polars versioning engine.

## Installation

The backend relies on [`vortex`](https://vortex.dev/), which is shipped with Metaxy's `vortex` extras.

```shell
pip install 'metaxy[vortex]'
```

## Storage Strategy

VortexMetadataStore currently supports **local filesystem storage only**:

- **Local paths**: Uses vortex-data's native Python API (`vortex.io.write()`)

!!! warning "Remote Storage Not Yet Supported"

    Remote storage (S3, GCS, Azure) is not yet supported due to a bug in the DuckDB vortex extension's COPY TO functionality.
    This will be enabled when the upstream fix is available.

    For remote storage needs, consider using [DeltaLake](../delta/index.md) instead.

## Storage Layout

Control how feature keys map to Vortex file locations with the `layout` parameter:

- `nested` (default): Places every feature in its own directory: `your/feature/key.vortex`
- `flat`: Stores all of them in the same directory: `your__feature__key.vortex`

## Performance Characteristics

- **Compression**: Vortex applies BtrBlocks adaptive compression automatically
- **Append pattern**: Uses read-concat-write (immutable file format)
- **Best for**: Large datasets with infrequent updates, high-performance analytics

For high-frequency updates, consider using [DuckDB](../../databases/duckdb/index.md) or [LanceDB](../../databases/lancedb/index.md) instead.

## Reference

- [Configuration](configuration.md)
- [API](api.md)
