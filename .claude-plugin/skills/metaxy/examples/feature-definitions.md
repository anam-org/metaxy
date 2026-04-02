# Feature Definitions

See full documentation: https://docs.metaxy.io/stable/guide/concepts/definitions/features/

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
        key="media/video",
        id_columns=["video_id"],
        fields=[
            "audio",   # Logical field: describes a data content aspect
            "frames",  # Logical field: not a metadata column
        ],
    ),
):
    # Metadata columns: stored in the metadata store
    video_id: str
    path: str
    duration: float
    height: int
    width: int
```

**Logical fields** (`fields` in `FeatureSpec`) describe the data contents for versioning and lineage. They do not correspond to metadata columns. **Metadata columns** (class attributes) are the actual data stored in the metadata store.

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
    model_name: str
    confidence: float
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

## Field Dependencies via Mapping

Use `FieldsMapping` on `FeatureDep` to automatically resolve field dependencies:

```python
import metaxy as mx
from metaxy.models.fields_mapping import FieldsMapping


class DownstreamFeature(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="downstream",
        deps=[
            # Default mapping: auto-matches fields by name
            mx.FeatureDep(feature=UpstreamFeature),
            # Suffix matching: "french" matches "audio/french"
            mx.FeatureDep(
                feature=AudioFeature,
                fields_mapping=FieldsMapping.default(match_suffix=True),
            ),
        ],
        fields=["audio", "french", "combined"],
        id_columns=["sample_id"],
    ),
):
    sample_id: str
```

Consult with documentation for more details and mapping options.

## Direct Field Dependencies (FieldDep)

Use `FieldDep` on `FieldSpec` to explicitly specify which upstream fields a field depends on:

```python
import metaxy as mx


class Crop(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="example/crop",
        deps=[mx.FeatureDep(feature=Video)],
        fields=[
            mx.FieldSpec(
                key="audio",
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=Video,
                        fields=["audio"],  # Depend only on Video's audio field
                    )
                ],
            ),
            mx.FieldSpec(
                key="frames",
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=Video,
                        fields=["frames"],  # Depend only on Video's frames field
                    )
                ],
            ),
        ],
        id_columns=["sample_uid"],
    ),
):
    sample_uid: str
```
