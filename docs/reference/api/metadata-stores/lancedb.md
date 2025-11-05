---
title: LanceDB Metadata Store
---

# LanceDBMetadataStore

::: metaxy.metadata_store.lancedb.LanceDBMetadataStore

## Installation

Install the LanceDB backend alongside Metaxy using the `lancedb` dependency group:

```bash
uv sync --group lancedb
```

This installs [`lancedb`](https://lancedb.com/) and its transitive dependencies.

## Usage

```py
from pathlib import Path

import polars as pl

from metaxy.metadata_store.lancedb import LanceDBMetadataStore

root = Path("/data/metaxy/lancedb")
feature = MyFeature

with LanceDBMetadataStore(root) as store:
    with store.allow_cross_project_writes():
        store.write_metadata(
            feature,
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

    metadata = store.read_metadata(feature)
    if metadata is not None:
        df = metadata.collect().to_polars()
        print(df)
```

### Storage Layout

- Each feature is stored as an individual Lance table inside the database directory.
- LanceDB manages the table schema, updates, and transactions. Metaxy simply appends new rows for each metadata write.

### Hashing

`LanceDBMetadataStore` defaults to `HashAlgorithm.XXHASH64`, identical to other non-SQL backends. You can override the hash algorithm via the constructor if you prefer a different supported variant.
