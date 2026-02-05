---
title: "Quickstart Guide"
description: "Get started with Metaxy."
---

# Quickstart

## Installation

Install Metaxy with `deltalake` - an easy way to setup a [`MetadataStore`](../learn/metadata-stores.md) locally:

```shell
pip install 'metaxy[delta]'
```

## Drop a `metaxy.toml` file

```toml {title="metaxy.toml"}
project = "quickstart"
entrypoints = ["features.py"]

[stores.dev]
type = "metaxy.metadata_store.deltalake.DeltaMetadataStore"
config = { root_path = "${HOME}/.metaxy/deltalake" }
```

## Define a root feature

Every Metaxy project must define at least one root feature.
Such features do not have upstream dependencies and act as inputs to the feature graph.

```python {title="features.py"}
import metaxy as mx
from pydantic import Field


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video",
        id_columns=["video_id"],
        fields=[
            "audio",
            "frames",
        ],
    ),
):
    # define DB columns
    video_id: str = Field(description="Unique identifier for the video")
    path: str = Field(description="Path to the video file")
    duration: float = Field(description="Duration of the video in seconds")
```

## Create feature materialization script

Use [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] to compute an increment for materialization:

<!-- skip: next -->

```py {title="script.py"}
import metaxy as mx

from .features import Video

# discover and load Metaxy features
cfg = mx.init_metaxy()

# instantiate the MetadataStore
store = cfg.get_store("dev")

# somehow prepare a DataFrame with incoming metadata
# this can be a Pandas, Polars, Ibis, or any other DataFrame supported by Narwhals
samples = ...

with store:
    increment = store.resolve_update(Video, samples=samples)
```

### 3. Run user-defined computation over the increment

Metaxy is not involved in this step at all.

<!-- skip: next -->

```py {title="script.py"}
if (len(increment.added) + len(increment.changed)) > 0:
    # run your computation, this can be done in a distributed manner
    results = run_custom_pipeline(diff, ...)
```

### 4. Record metadata for processed samples

<!-- skip: next -->

```py {title="script.py"}
with store.open("w"):
    store.write(VoiceDetection, results)
```

We have now successfully recorded the metadata for the computed samples! Processed samples will no longer be returned by `MetadataStore.resolve_update` during future pipeline runs.

---

## Next Steps

Continue to [next section](feature-dependencies.md) to learn how to add more features and define feature dependencies.

---

## Additional info

- Learn more about [feature definitions](../learn/definitions/features.md) or [versioning](../learn/data-versioning.md)

- Explore [Metaxy integrations](../../integrations/index.md)

- Use Metaxy [from the command line](../../reference/cli.md)

- Learn how to [configure Metaxy](../../reference/configuration.md)

- Get lost in our [API Reference](../../reference/api/index.md)
