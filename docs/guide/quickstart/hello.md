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
import metaxy as mx
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

- `increment.added`: new samples which were not previously recorded

- `increment.changed`: samples which were previously recorded but have now changed

- `increment.removed`: samples which were previously recorded but are no longer present in the input `samples` DataFrame

It's up to you how to handle these dataframes.
Usually there will be a processing step iterating over all the rows in `increment.added` and `increment.changed` (possibly in parallel, using something like [Ray](/integrations/compute/ray.md)), while `increment.removed` may be used to cleanup the no longer needed data and [metadata](/guide/concepts/deletions.md).

These dataframes have pre-computed provenance columns which **should not be modified** and eventually written to the metadata store.

!!! tip

    The dataframes will have a `metaxy_provenance` column which is recommended to be used for storage paths:
    <!-- skip: next -->
    ```python
    from pathlib import Path

    to_process = pl.concat([increment.added.to_polars(), increment.changed.to_polars()])

    result = []
    for row in to_process.iter_rows(named=True):
        path = Path(row["raw_video_path"]) / row["id"] / row["metaxy_provenance"] / "video.mp4"
        process_video(row["raw_video_path"], path)
        result.append(path)
    ```

### 4. Record metadata for processed samples

Once done, write the metadata for the processed samples:

<!-- skip: next -->

```py {title="script.py"}
with store.open("w"):
    store.write(VoiceDetection, result)
```

Recorded samples will no longer be returned by `MetadataStore.resolve_update` during future pipeline runs, unless their `metaxy_provenance_by_field` values are updated.

## Next Steps

Continue to [next section](feature-dependencies.md) to learn how to add more features and define feature dependencies.
