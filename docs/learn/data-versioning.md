# Data Versioning

Metaxy computes data versions as hashes of upstream dependencies, enabling automatic change detection and incremental computation.

## How It Works

Consider a video processing pipeline with these features:

```python
from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)


class Video(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "video"]),
        deps=None,  # Root feature
        fields=[
            FieldSpec(key=FieldKey(["audio"]), code_version=1),
            FieldSpec(key=FieldKey(["frames"]), code_version=1),
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
        key=FeatureKey(["example", "crop"]),
        deps=[FeatureDep(key=Video.spec.key)],
        fields=[
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version=1,
                deps=[
                    FieldDep(feature_key=Video.spec.key, fields=[FieldKey(["audio"])])
                ],
            ),
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version=1,
                deps=[
                    FieldDep(feature_key=Video.spec.key, fields=[FieldKey(["frames"])])
                ],
            ),
        ],
    ),
):
    pass


class FaceDetection(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "face_detection"]),
        deps=[FeatureDep(key=Crop.spec.key)],
        fields=[
            FieldSpec(
                key=FieldKey(["faces"]),
                code_version=1,
                deps=[
                    FieldDep(feature_key=Crop.spec.key, fields=[FieldKey(["frames"])])
                ],
            ),
        ],
    ),
):
    pass


class SpeechToText(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "stt"]),
        deps=[FeatureDep(key=Video.spec.key)],
        fields=[
            FieldSpec(
                key=FieldKey(["transcription"]),
                code_version=1,
                deps=[
                    FieldDep(feature_key=Video.spec.key, fields=[FieldKey(["audio"])])
                ],
            ),
        ],
    ),
):
    pass
```

Running `metaxy graph render --format mermaid` produces this graph:

```mermaid
---
title: Feature Graph
---
flowchart TB
    %% Snapshot version: 8468950d
    %%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '14px'}}}%%
        example_video["<div style="text-align:left"><b>example/video</b><br/><small>(v: bc9ca835)</small><br/><font
color="#999">---</font><br/>• audio <small>(v: 22742381)</small><br/>• frames <small>(v: 794116a9)</small></div>"]
        example_crop["<div style="text-align:left"><b>example/crop</b><br/><small>(v: 3ac04df8)</small><br/><font
color="#999">---</font><br/>• audio <small>(v: 76c8bdc9)</small><br/>• frames <small>(v: abc79017)</small></div>"]
        example_face_detection["<div style="text-align:left"><b>example/face_detection</b><br/><small>(v: 1ac83b07)</small><br/><font
color="#999">---</font><br/>• faces <small>(v: 2d75f0bd)</small></div>"]
        example_stt["<div style="text-align:left"><b>example/stt</b><br/><small>(v: c83a754a)</small><br/><font
color="#999">---</font><br/>• transcription <small>(v: ac412b3c)</small></div>"]
        example_video --> example_crop
        example_crop --> example_face_detection
        example_video --> example_stt
```

## Tracking Changes

Imagine the `audio` field of the `Video` feature changes (perhaps denoising was applied):

```diff
         key=FeatureKey(["example", "video"]),
         deps=None,
         fields=[
             FieldSpec(
                 key=FieldKey(["audio"]),
-                code_version=1,
+                code_version=2,
             ),
```

Run `metaxy graph diff` to see what changed:

```mermaid
---
title: Merged Graph Diff
---
flowchart TB
    %%{init: {'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'themeVariables': {'fontSize': '14px'}}}%%

    example_video["<div style="text-align:left"><b>example/video</b><br/><font color="#CC0000">bc9ca8</font> → <font
color="#00AA00">6db302</font><br/><font color="#999">---</font><br/>- <font color="#FFAA00">audio</font> (<font
color="#CC0000">227423</font> → <font color="#00AA00">09c839</font>)<br/>- frames (794116)</div>"]
    style example_video stroke:#FFA500,stroke-width:3px
    example_crop["<div style="text-align:left"><b>example/crop</b><br/><font color="#CC0000">3ac04d</font> → <font
color="#00AA00">54dc7f</font><br/><font color="#999">---</font><br/>- <font color="#FFAA00">audio</font> (<font
color="#CC0000">76c8bd</font> → <font color="#00AA00">f3130c</font>)<br/>- frames (abc790)</div>"]
    style example_crop stroke:#FFA500,stroke-width:3px
    example_face_detection["<div style="text-align:left"><b>example/face_detection</b><br/>1ac83b<br/><font
color="#999">---</font><br/>- faces (2d75f0)</div>"]
    example_stt["<div style="text-align:left"><b>example/stt</b><br/><font color="#CC0000">c83a75</font> → <font
color="#00AA00">066d34</font><br/><font color="#999">---</font><br/>- <font color="#FFAA00">transcription</font> (<font
color="#CC0000">ac412b</font> → <font color="#00AA00">058410</font>)</div>"]
    style example_stt stroke:#FFA500,stroke-width:3px

    example_video --> example_crop
    example_crop --> example_face_detection
    example_video --> example_stt
```

Notice:

- `Video`, `Crop`, and `SpeechToText` changed (highlighted)
- `FaceDetection` remained unchanged (depends only on `frames`, not `audio`)
- Audio field versions changed throughout the graph
- Frame field versions stayed the same

Metaxy's static graph analysis identifies features out of sync after topology changes or code version bumps. Beyond feature and field-level versions, Metaxy computes sample-level versions ahead of computation through the entire graph, enabling processing cost prediction and automatic migrations.

## Sample-Level Versioning

For each sample (row) in your dataset, Metaxy computes a data version by hashing upstream dependency versions. This happens before the actual computation.

Example metadata row:

```python
{
    "sample_uid": "video_001",
    "data_version": {
        "audio": "a2ha72a",
        "frames": "ja812hp",
    },
    "feature_version": "nasdh1a",
    "snapshot_version": "def456",
    # User columns
    "path": "/data/video_001.mp4",
    "duration": 120.5,
}
```

When upstream dependencies change, Metaxy recalculates data versions and identifies which samples need recomputation by comparing old versus new versions.

## Incremental Computation

The metadata store's `calculate_and_write_data_versions()` method:

1. Joins upstream feature metadata
2. Computes data version hashes for each sample
3. Compares against existing metadata
4. Returns diff: added, changed, removed samples

Your pipeline processes only the samples that changed:

```python
with store:  # MetadataStore
    # Metaxy computes data_version and identifies changes
    diff = store.resolve_update(MyFeature)

    # Process only changed samples
```

This approach avoids expensive recomputation when nothing changed, while ensuring correctness when dependencies update.

## Version vs Data Version

**Feature version**: Hash of feature definition (code, deps, fields). Static, deterministic from code.

**Data version**: Hash of upstream data versions for a specific sample. Depends on actual data.

**Snapshot version**: Hash of all feature versions in the graph. Represents entire graph state.

A feature version change doesn't necessarily mean all samples need recomputation. Metaxy compares data versions to identify only the affected samples.
