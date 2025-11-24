# Quickstart

## Installation

Install Metaxy with `deltalake` -- an easy way to setup a [`MetadataStore`](../learn/metadata-stores.md):

```shell
pip install 'metaxy[delta]'
```

## Create a minimal Metaxy config:

```toml filename="metaxy.toml"
entrypoints = ["features.py"]

[stores.dev]
type = "metaxy.metadata_store.deltalake.DeltaMetadataStore"
config = { path = "/tmp/metaxy.delta" }
```

## Define a root feature

Every Metaxy project must define one or more root features -- features without upstream dependencies.

```python filename="features.py"
import metaxy as mx
from pydantic import Field


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video",
        fields=[
            "audio",
            "frames",
        ],
    ),
):
    # define DB columns
    path: str = Field(description="Path to the video file")
    duration: float = Field(description="Duration of the video in seconds")
```

## Create feature materialization script

Use [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update] to compute an increment for materialization:

```py filename="script.py"
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

```py filename="script.py"
if (len(increment.added) + len(increment.changed)) > 0:
    # run your computation, this can be done in a distributed manner
    results = run_custom_pipeline(diff, ...)
```

### 4. Record metadata for processed samples

```py filename="script.py"
with store.open("write"):
    store.write_metadata(VoiceDetection, results)
```

We have now successfully recorded the metadata for the computed samples! Processed samples will no longer be returned by `MetadataStore.resolve_update` during future pipeline runs.

> [!WARNING] No Write Time Uniqueness Checks!
> Metaxy doesn't enforce deduplication or uniqueness checks at **write time** for performance reasons.
> While `MetadataStore.resolve_update` is guaranteed to never return the same versioned sample twice, it's up to the user to ensure that samples are not written multiple times to the metadata store.
> Configuring deduplication or uniqueness checks in the store (database) is a good idea.
> For example, the [SQLModel integration](../integrations/plugins/sqlmodel.md) can inject a composite primary key on `metaxy_data_version`, `metaxy_created_at` and the user-defined ID columns.
> However, Metaxy only uses the latest version (by `metaxy_created_at`) at **read time**.

## What's Next?

- Learn more about [feature definitions](../learn/feature-definitions.md) or [versioning](../learn/data-versioning.md)

- Explore [Metaxy integrations](../integrations/index.md)

- Use Metaxy [from the command line](../reference/cli.md)

- Learn how to [configure Metaxy](../reference/configuration.md)

- Get lost in our [API Reference](../reference/api/index.md)
