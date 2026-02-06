---
title: "Feature Dependencies"
description: "Defining dependencies between features."
---

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
changes = store.resolve_update(Audio)
```

That's all! The changes can be handled similarly to the `Video` feature.

## Advanced Feature Definitions

Learn how to mark dependencies as optional, specify field-level versions and dependencies, lineage types, and other advanced Metaxy features in [definitions docs](/guide/concepts/definitions/features.md).

## What's Next?

Here are a few more useful links:

--8<-- "whats-next.md"
