# Optional Dependencies

By default, feature dependencies use **inner joins** - a sample only exists in the downstream feature if it exists in **all** upstream dependencies.
**Optional dependencies** allow you to preserve samples even when some upstream data is missing by using **left joins** instead.

## Overview

When building features with multiple dependencies, you often want certain dependencies to be "optional" - meaning the downstream feature should still include samples even if the optional dependency has no matching data.
This is controlled by the `optional` parameter on [`FeatureDep`][metaxy.FeatureDep].

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
| All optional                | Outer join | Any row from any dependency can pass through      |

Required dependencies are joined first (inner join), then optional dependencies are joined (left join).
If all dependencies are optional, outer joins are used instead to allow any row to pass through.

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

## Provenance with Optional Dependencies

Metaxy tracks provenance correctly even when optional dependencies have missing data. When an optional dependency has no match, an empty string is used in provenance calculation.
This means samples with vs without optional data will have different provenance hashes, which is correct behavior since the sample's lineage genuinely differs based on what upstream data contributed to it.
