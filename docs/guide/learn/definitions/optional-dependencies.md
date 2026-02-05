---
title: "Optional Dependencies"
description: "Learn how to define optional feature dependencies."
---

# Optional Dependencies

By default, Metaxy assumes that every upstream feature must have a matching sample in order to materialize a downstream one.
This means that samples with at least one missing upstream sample are not included in the result of [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update].

To customize this behavior, it is possible to mark specific upstream features as optional:

```python hl_lines="21"
import metaxy as mx


class RawVideo(mx.BaseFeature, spec=mx.FeatureSpec(key="raw/video", id_columns=["video_id"])):
    path: str


class AudioTranscript(
    mx.BaseFeature, spec=mx.FeatureSpec(key="audio/transcript", id_columns=["video_id"], deps=[RawVideo])
):
    text: str


class EnrichedVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="enriched/video",
        id_columns=["video_id"],
        deps=[
            mx.FeatureDep(feature=RawVideo),
            mx.FeatureDep(feature=AudioTranscript, optional=True),  # Optional
        ],
    ),
):
    pass
```

In this example, even if some audio transcripts have not been extracted, the downstream feature will still be allowed to process these samples, leaving it to the user to decide how to handle missing data.
