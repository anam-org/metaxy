# Feature Definitions

See full documentation: https://anam-org.github.io/metaxy/guide/learn/feature-definitions/

!!! critical

    Feature definitions are the core of Metaxy. It's extremely important to define them correctly and carefully.
    Make sure to understand the different between **metadata columns** (class attributes / pydantic fields)
    and Metaxy's `mx.FieldSpec` (purely logical, describe the **data contents**).

## Basic Feature

```python
import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="my/feature",
        id_columns=["video_id"],
        fields=[
            "audio",
            "frames",
        ],
    ),
):
    video_id: str
    path: str
    duration: float
    height: int
    width: int
```

## Feature with Dependencies

```python
import metaxy as mx


class ChildFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="child/feature",
        id_columns=["sample_id"],
        deps=[ParentFeature],  # Simple dependency
        fields=["predictions"],
    ),
):
    sample_id: str
```

## Versioned Fields

```python
import metaxy as mx


class VersionedFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        fields=[
            mx.FieldSpec(key="versioned_field", code_version="1"),
        ],
        ...
    ),
):
    ...
```

!!! critical

    Changing the `code_version` of a field will invalidate downstream feature samples that depend on this field.
