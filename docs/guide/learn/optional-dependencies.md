# Optional Dependencies

By default, feature dependencies use **inner joins** - a sample only exists in the downstream feature if it exists in **all** upstream dependencies.
**Optional dependencies** allow you to preserve samples even when some upstream data is missing by using **left joins** instead.

## Overview

When building features with multiple dependencies, you often want certain dependencies to be "optional" - meaning the downstream feature should still include samples even if the optional dependency has no matching data. This is controlled by the `optional` parameter on [`FeatureDep`][metaxy.FeatureDep].

```python
from metaxy import FeatureSpec, FeatureDep

spec = FeatureSpec(
    key="enriched/video",
    id_columns=["video_id"],
    deps=[
        FeatureDep(feature=RawVideo),  # Required (default)
        FeatureDep(feature=AudioTranscript, optional=True),  # Optional
    ],
    fields=["analysis"],
)
```

## Join Behavior

| Dependency Type             | Join Type  | Behavior                                          |
| --------------------------- | ---------- | ------------------------------------------------- |
| Required (`optional=False`) | Inner join | Sample must exist in both features                |
| Optional (`optional=True`)  | Left join  | Sample preserved even if no match in optional dep |

## When to Use Optional Dependencies

### Enrichment Pattern

Use optional dependencies when you have a **base feature** that should always be present, with **optional enrichment data** that may or may not exist:

```python
class VideoAnalysis(
    BaseFeature,
    spec=FeatureSpec(
        key="video/analysis",
        id_columns=["video_id"],
        deps=[
            FeatureDep(feature=RawVideo),  # Required - defines sample universe
            FeatureDep(
                feature=AudioTranscription, optional=True
            ),  # Optional enrichment
            FeatureDep(feature=ManualAnnotations, optional=True),  # Optional enrichment
        ],
        fields=["analysis"],
    ),
):
    pass
```

In this example:

- All samples from `RawVideo` are preserved
- If `AudioTranscription` exists for a sample, it's included; otherwise, those columns are `NULL`
- If `ManualAnnotations` exists for a sample, it's included; otherwise, those columns are `NULL`

### Multi-Source Fusion

Use optional dependencies when combining data from **multiple sources** where not all sources have all samples:

```python
class UnifiedUserProfile(
    BaseFeature,
    spec=FeatureSpec(
        key="user/unified",
        id_columns=["user_id"],
        deps=[
            FeatureDep(feature=CoreUserData),  # Required base
            FeatureDep(feature=SocialMediaData, optional=True),
            FeatureDep(feature=PurchaseHistory, optional=True),
        ],
        fields=["profile"],
    ),
):
    pass
```

## Rules

### First Dependency Must Be Required

The **first dependency** in the list defines the "sample universe" - it determines which samples exist in the downstream feature.
Therefore, the first dependency **cannot be optional**:

```python
# INVALID - first dependency cannot be optional
spec = FeatureSpec(
    key="my/feature",
    id_columns=["id"],
    deps=[
        FeatureDep(feature=OptionalData, optional=True),  # Error!
        FeatureDep(feature=RequiredData),
    ],
    fields=["output"],
)
# Raises: ValueError: The first dependency cannot be optional...

# VALID - first dependency is required
spec = FeatureSpec(
    key="my/feature",
    id_columns=["id"],
    deps=[
        FeatureDep(feature=RequiredData),  # Required (first)
        FeatureDep(feature=OptionalData, optional=True),  # Optional (subsequent)
    ],
    fields=["output"],
)
```

### Join Order Follows Declaration Order

Dependencies are joined in the order they are declared in the `deps` list.
This ensures deterministic behavior:

1. First dependency becomes the base table
2. Subsequent dependencies are joined one by one
3. Required deps use inner join, optional deps use left join

## Handling NULL Values

When an optional dependency has no matching data, its columns will contain `NULL` values.
Your feature's `load_input()` method must handle these NULLs appropriately:

```python
import polars as pl


class EnrichedVideo(BaseFeature, spec=...):
    def load_input(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Handle NULL from optional AudioTranscription
        return df.with_columns(
            pl.col("transcript").fill_null("NO_TRANSCRIPT"),
            pl.col("confidence_score").fill_null(0.0),
        )
```

## Provenance with Optional Dependencies

Metaxy tracks provenance correctly even when optional dependencies have missing data:

- When an optional dependency has **no match**, a sentinel value `"__NULL__"` is used in provenance calculation
- This means samples **with** vs **without** optional data will have **different provenance hashes**
- This is correct behavior: the sample's lineage genuinely differs based on what upstream data contributed to it

### Provenance Determinism

Provenance is deterministic: the same combination of upstream data (including missing optional data) always produces the same provenance hash.

```python
# These samples will have DIFFERENT provenance:
# - Sample A: has RawVideo + AudioTranscription data
# - Sample B: has only RawVideo data (AudioTranscription missing)

# These samples will have SAME provenance:
# - Sample B: has only RawVideo data
# - Sample C: has only RawVideo data (assuming same upstream provenance)
```

## Example: Video Processing Pipeline

Here's a complete example showing optional dependencies in a video processing pipeline:

```python
import metaxy as mx
from metaxy import FeatureSpec, FeatureDep
import polars as pl


class RawVideo(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="raw/video",
        id_columns=["video_id"],
        fields=["duration", "format"],
    ),
):
    video_id: str
    duration: float
    format: str


class AudioTranscription(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="audio/transcription",
        id_columns=["video_id"],
        fields=["transcript", "language"],
    ),
):
    video_id: str
    transcript: str
    language: str


class FaceDetection(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="vision/faces",
        id_columns=["video_id"],
        fields=["face_count", "confidence"],
    ),
):
    video_id: str
    face_count: int
    confidence: float


class VideoAnalysis(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="video/analysis",
        id_columns=["video_id"],
        deps=[
            FeatureDep(feature=RawVideo),  # Required
            FeatureDep(feature=AudioTranscription, optional=True),  # Optional
            FeatureDep(feature=FaceDetection, optional=True),  # Optional
        ],
        fields=["summary"],
    ),
):
    video_id: str
    duration: float
    format: str
    transcript: str | None  # Can be NULL
    language: str | None  # Can be NULL
    face_count: int | None  # Can be NULL
    summary: str

    def load_input(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Handle NULLs from optional dependencies
        return df.with_columns(
            pl.col("transcript").fill_null(""),
            pl.col("language").fill_null("unknown"),
            pl.col("face_count").fill_null(0),
        )
```

## Best Practices

1. **Order dependencies intentionally**: Place the most "core" required dependency first, as it defines the sample universe.

2. **Handle NULLs explicitly**: Always account for NULL values in columns from optional dependencies in your `load_input()` method.

3. **Use type hints**: Annotate fields from optional dependencies as `| None` to make NULL handling explicit.

4. **Consider provenance implications**: Remember that adding/removing optional data changes provenance. This is usually desired behavior but should be understood.

5. **Test edge cases**: Test your feature with scenarios where optional dependencies have full, partial, and no matches.
