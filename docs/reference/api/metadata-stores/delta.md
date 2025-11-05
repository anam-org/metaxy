---
title: Delta Lake Metadata Store
---

# DeltaMetadataStore

::: metaxy.metadata_store.delta.DeltaMetadataStore

## Installation

The Delta Lake backend relies on [`deltalake`](https://delta-io.github.io/delta-rs/python/), which is provided through Metaxy’s `delta` dependency group. Install it with:

```bash
uv sync --group delta
```

This will pull in the delta-rs bindings alongside the core Metaxy dependencies (if they are not already installed).

## Usage

```py
from pathlib import Path

import polars as pl

from metaxy.metadata_store.delta import DeltaMetadataStore

root = Path("/data/metaxy/metadata")

with DeltaMetadataStore(
    root,
    storage_options={
        # Optional: forward credentials or configuration to the underlying storage backend
        "AWS_REGION": "us-west-2",
    },
) as store:
    # Allow writing metadata from integration tests or scripts
    with store.allow_cross_project_writes():
        store.write_metadata(
            MyFeature,
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "provenance_by_field": [
                        {"frames": "hash1", "audio": "hash1"},
                        {"frames": "hash2", "audio": "hash2"},
                    ],
                }
            ),
        )

    # Later in the same session (or a new one) you can read lazily
    lazy_metadata = store.read_metadata(MyFeature)
    if lazy_metadata is not None:
        df = lazy_metadata.collect().to_polars()
        print(df)
```

### Storage Layout

- Each feature is stored in its own Delta table under `<root>/<namespace>__<feature_name>`.
- Delta transaction logs are managed automatically by `deltalake`.
- `DeltaMetadataStore` uses the Polars/Narwhals fall-back components for hashing and diffs, so no database-specific extensions are required.

### Hashing

`DeltaMetadataStore` defaults to `HashAlgorithm.XXHASH64`, mirroring other non-SQL backends. You can override the algorithm via the constructor arguments, provided the requested hash is supported by the Polars calculator.
