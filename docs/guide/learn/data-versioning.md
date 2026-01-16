---
title: "Data Versioning"
description: "How Metaxy calculates and tracks versions."
---

# Versioning

Metaxy calculates a few types of versions at [feature](feature-definitions.md), [field](feature-definitions.md), and [sample](#samples) levels.

Metaxy's versioning system is declarative, static, deterministic and idempotent.

## Versioning

Feature and field versions are defined by the feature graph topology and the user-provided code versions of fields. Sample versions are defined by upstream sample versions and the code versions of the fields defined on the sample's feature.

All versions are computed ahead of time: feature and field versions can be immediately derived from code (and we keep historical graph snapshots for them), and calculating sample versions requires access to the metadata store.

Metaxy uses hashing algorithms to compute all versions. The algorithm and the hash [length](../../reference/configuration.md#hash_truncation_length) can be configured.

Here is how these versions are calculated, from bottom to top.

### Definitions

These versions can be computed from Metaxy definitions (e.g. Python code or historical snapshots of the feature graph). We don't need to access the metadata store in order to calculate them.

#### Field Level

- **Field Code Version** is defined on the field and is provided by the user (defaults to `"__metaxy_initial__"`)

> [!NOTE] Code Version Value
> The value can be arbitrary, but in the future we might implement something around semantic versioning.

- **Field Version** is computed from the code version of this field, the [fully qualified field path](feature-definitions.md#fully-qualified-field-key) and from the field versions of its [parent fields](feature-definitions.md#field-level-dependencies) (if any exist, for example, fields on root features do not have dependencies).

#### Feature Level

- **Feature Version**: is computed from the **Field Versions** of all fields defined on the feature and the key of the feature.
- **Feature Code Version** is computed from the **Field Code Versions** of all fields defined on the feature. Unlike _Feature Version_, this version does not change when dependencies change. The value of this version is determined entirely by user input.

#### Graph Level

- **Snapshot Version**: is computed from the **Feature Versions** of all features defined on the graph.

??? "How is snapshot version used?"

    This value is used to uniquely encode versioned feature graph topology. `metaxy graph push` CLI can be used to keep track of previous versions of the feature graph, enabling features such as data version reconciliation migrations.

### Samples

These versions are sample-level and require access to the metadata store in order to compute them.

- **Provenance By Field** is computed from the upstream **Provenance By Field** (with respect to defined [field-level dependencies](feature-definitions.md#field-level-dependencies) and the code versions of the current fields. This is a dictionary mapping sample field names to their respective versions. This is how this looks like in the metadata store (database):

| sample_uid | metaxy_provenance_by_field                    |
| ---------- | --------------------------------------------- |
| video_001  | `{"audio": "a7f3c2d8", "frames": "b9e1f4a2"}` |
| video_002  | `{"audio": "d4b8e9c1", "frames": "f2a6d7b3"}` |
| video_003  | `{"audio": "c9f2a8e4", "frames": "e7d3b1c5"}` |
| video_004  | `{"audio": "b1e4f9a7", "frames": "a8c2e6d9"}` |

- **Sample Version** is derived from the **Provenance By Field** by simply hashing it.

Computing this value is the goal of the entire versioning engine. It ensures that only the necessary samples are recomputed when a feature version changes. It acts as source of truth for resolving incremental updates for feature metadata.

!!! tip "Customizing Sample Versions"

    Users can override the computed sample-level versions by setting `metaxy_data_version_by_field` on their metadata. This can be used for eliminating false-positives (e.g. content-based hashing), when sometimes data stays the same even after upstream has changed. This customization only affects how downstream increments are calculated.

## Practical Example

Consider a video processing pipeline with these features:

??? tip "Simplified Metaxy Definitions"

    This example uses Metaxy's [syntactic sugar](syntactic-sugar.md) for cleaner code.
    Feature classes can be passed directly to `deps` instead of wrapping in `FeatureDep`,
    and field names matching upstream fields automatically create field-level dependencies.

```python
from metaxy import Feature, FeatureSpec, FieldDep, FieldSpec


class Video(
    Feature,
    spec=FeatureSpec(
        key="example/video",
        fields=[
            FieldSpec(key="audio", code_version="1"),
            FieldSpec(key="frames", code_version="1"),
        ],
    ),
):
    """Video metadata feature (root)."""

    frames: int
    duration: float
    size: int


class Crop(
    Feature,
    spec=FeatureSpec(
        key="example/crop",
        deps=[Video],
        fields=[
            FieldSpec(key="audio", code_version="1"),  # (1)!
            FieldSpec(key="frames", code_version="1"),  # (2)!
        ],
    ),
):
    pass  # omit columns for the sake of simplicity


class FaceDetection(
    Feature,
    spec=FeatureSpec(
        key="example/face_detection",
        deps=[Crop],
        fields=[
            FieldSpec(
                key="faces",
                code_version="1",
                deps=[FieldDep(feature=Crop, fields=["frames"])],
            ),
        ],
    ),
):
    pass


class SpeechToText(
    Feature,
    spec=FeatureSpec(
        key="example/stt",
        deps=[Video],
        fields=[
            FieldSpec(
                key="transcription",
                code_version="1",
                deps=[FieldDep(feature=Video, fields=["audio"])],
            ),
        ],
    ),
):
    pass
```

{ .annotated }

1. This `audio` field [automatically depends][metaxy.models.fields_mapping.FieldsMapping.default] on the `audio` field of the `Video` feature, because their names match.

2. This `frames` field [automatically depends][metaxy.models.fields_mapping.FieldsMapping.default] on the `frames` field of the `Video` feature, because their names match.

Running `metaxy graph render --format mermaid` produces this graph:

::: metaxy-example graph
example: overview
scenario: "Initial feature graph"
:::

## Tracking Definitions Changes

Imagine the `audio` field of the `Video` feature changes (1):
{ .annotate }

1. Perhaps, something like denoising has been applied externally

::: metaxy-example patch
example: overview
path: patches/01_update_audio_version.patch
:::

Run `metaxy graph diff` to see what changed:

::: metaxy-example graph-diff
example: overview
scenario: "Code change - audio field"
step: "update_audio_version"
:::

!!! info

    - `Video`, `Crop`, and `SpeechToText` have changed

    - `FaceDetection` remained unchanged (depends only on `frames` and not on `audio`)

    - Audio field versions have changed throughout the graph

    - Frame field versions have stayed the same

## Incremental Computation

The single most important piece of code in Metaxy is the [`resolve_update`][metaxy.MetadataStore.resolve_update] method.
It handles the following:

1. Joins upstream feature metadata

2. Computes sample versions

3. Compares against existing metadata

4. Returns diff: added, changed, removed samples

Typically, steps 1-3 can be run directly in the database. Analytical databases such as ClickHouse or Snowflake can efficiently handle these operations.

The Python pipeline then only handles the increment.

```python
with store:  # MetadataStore
    # Metaxy computes provenance_by_field and identifies changes
    increment = store.resolve_update(MyFeature)

    # Process only changed samples
```

The `increment` object has attributes for new upstream samples, samples with new versions, and samples that have been removed from upstream metadata.
