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

### Object Stores

`DeltaMetadataStore` can operate directly against cloud object stores by pointing `root` at a URI such as `s3://bucket/path`, `gs://bucket/path`, or `az://account/container/path`. When used with remote storage, Metaxy relies on [`obstore`](https://developmentseed.org/obstore) (installed via the `delta` dependency group) so that both Delta Lake operations and metadata management (listing/dropping tables) share the same credentials.

```py
remote_root = "s3://my-metadata-bucket/prod"

store = DeltaMetadataStore(
    remote_root,
    storage_options={
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "AWS_REGION": "us-west-2",
    },
    object_store_kwargs={
        "config": {
            "access_key_id": "...",
            "secret_access_key": "...",
            "region": "us-west-2",
        }
    },
)
```

`storage_options` go straight to `deltalake`, while `object_store_kwargs` are passed to [`obstore.store.from_url`](https://developmentseed.org/obstore/api/store/#obstore.store.from_url) so you can configure the remote backend exactly as required (for example, by providing `config` or `client_options`). When using file-system paths (or `file:///` URIs) the store behaves exactly as before and `object_store_kwargs` are ignored.

### Storage Layout

- Each feature is stored in its own Delta table under `<root>/<namespace>__<feature_name>`.
- Delta transaction logs are managed automatically by `deltalake`.
- `DeltaMetadataStore` uses the Polars/Narwhals fall-back components for hashing and diffs, so no database-specific extensions are required.

### Hashing

`DeltaMetadataStore` defaults to `HashAlgorithm.XXHASH64`, mirroring other non-SQL backends. You can override the algorithm via the constructor arguments, provided the requested hash is supported by the Polars calculator.
