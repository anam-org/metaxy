---
title: "Hello, Metaxy!"
description: "Get started with using Metaxy."
---

# Quickstart

## 1. Install Metaxy

Let's choose a backend for our first [`MetadataStore`](/guide/concepts/metadata-stores.md).
A good option for local development is [DeltaLake](/integrations/metadata-stores/storage/delta.md).
Let's install it:

```shell
pip install 'metaxy[delta]'
```

Now the metadata store can be created as:

```py title="script.py"
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

store = DeltaMetadataStore("/tmp/quickstart.delta")
```

## 2. Define your first Feature

Any Metaxy project must have at least one root feature.

```python title="features.py"
import metaxy as mx
from pydantic import Field


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video",
        id_columns=["id"],
    ),
):
    # define DB columns
    raw_video_path: str = Field(description="Path to the raw video file")
    id: str = Field(description="Unique identifier for the video")
    path: str = Field(description="Path to the processed video file")
```

## 3. Resolve a root increment

Root features are a bit special. They are entry points into the Metaxy world.
Because of that, will have to provide a `samples` argument to [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update],
which is typically not required for non-root features.

!!! tip

    The only requirement for this dataframe is to have a `metaxy_provenance_by_field` column (and to have appropriate ID columns).

<!-- skip: next -->

```py {title="script.py"}
from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

from .features import Video

store = DeltaMetadataStore("/tmp/quickstart.delta")

# somehow prepare a DataFrame with incoming metadata
# this can be a Pandas, Polars, Ibis, or any other DataFrame supported by Narwhals
samples =

with store:
    increment = store.resolve_update(Video, samples=samples)
```

The `increment` object is an instance of [`Increment`][metaxy.Increment] and contains three dataframes:

- `increment.new`: new samples which were not previously recorded

- `increment.stale`: samples which were previously recorded but have now changed

- `increment.orphaned`: samples which were previously recorded but are no longer present in the input `samples` DataFrame

It's up to you how to handle these dataframes.
Usually there will be a processing step iterating over all the rows in `increment.new` and `increment.stale` (possibly in parallel, using something like [Ray](/integrations/compute/ray.md)), while `increment.orphaned` may be used to cleanup the no longer needed data and [metadata](/guide/concepts/deletions.md).

These dataframes have pre-computed provenance columns which **should not be modified** and eventually written to the metadata store.

!!! tip

    The dataframes will have a `metaxy_provenance` column which is recommended to be used for storage paths:
    <!-- skip: next -->
    ```python
    from pathlib import Path

    to_process = pl.concat([increment.new.to_polars(), increment.stale.to_polars()])

    result = []
    for row in to_process.iter_rows(named=True):
        path = Path(row["raw_video_path"]) / row["id"] / row["metaxy_provenance"] / "video.mp4"
        process_video(row["raw_video_path"], path)
        row["path"] = path
        result.append(row)
    ```

### 4. Record metadata for processed samples

Once done, write the metadata for the processed samples:

<!-- skip: next -->

```py {title="script.py"}
import polars as pl

with store.open("w"):
    store.write(VoiceDetection, pl.DataFrame(result))
```

Recorded samples will no longer be returned by `MetadataStore.resolve_update` during future pipeline runs, unless the incoming `metaxy_provenance_by_field` values are updated.


# Feature Dependencies

Now let's add a downstream feature. We can use `deps` field on [`FeatureSpec`][metaxy.FeatureSpec] in order to do that. We will make a simple feature that extracts the audio track from a video.

```py {title="features.py" hl_lines="31"}
import metaxy as mx
from pydantic import Field


class Audio(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="audio",
        id_columns=["id"],
    ),
):
    # define DB columns
    id: str = Field(description="Unique identifier for the audio")
    path: str = Field(description="Path to the audio file")
```


And call the faimiliar `resolve_update` API:

<!-- skip: next -->

```py {title="script.py"}
increment = store.resolve_update(Audio)
```

That's all! The increment can be handled similarly to the `Video` feature.

## Advanced Feature Definitions

Learn how to mark dependencies as optional, specify field-level versions and dependencies, lineage types, and other advanced Metaxy features in [definitions docs](/guide/concepts/definitions/features.md).

## What's Next?

Here are a few more useful links:

--8<-- "whats-next.md"
