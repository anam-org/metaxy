---
title: "Expansion Example"
description: "Example of one-to-many expansion relationships."
---

# Expansion

## Overview

::: metaxy-example source-link
    example: one-to-many

This example demonstrates how to implement expansion (`1:N`) transformations with Metaxy.
In such relationships a single parent sample can map into multiple child samples.

These relationships can be modeled with [LineageRelationship.expansion][metaxy.models.lineage.LineageRelationship.expansion] lineage type.

We will use a hypothetical video chunking pipeline as an example.

## The Pipeline

We are going to define a typical video processing pipeline with three features:

::: metaxy-example graph
    example: one-to-many
    scenario: "Initial pipeline run"
    direction: LR

### Defining features: `"video/raw"`

Each video-like feature in our pipeline is going to have two fields: `audio` and `frames`.

Let's set the code version of `audio` to `"1"` in order to change it in the future.
`frames` field will have a default version.

<!-- dprint-ignore-start -->
```python title="src/example_one_to_many/features.py" hl_lines="10"
--8<-- "example-one-to-many/src/example_one_to_many/features.py:video"
```
<!-- dprint-ignore-end -->

### Defining features: `"video/chunk"`

`"video/chunk"` represents a piece of the upstream `"video/raw"` feature. Since each `"video/raw"` sample can be split into multiple chunks, we need to tell Metaxy how to map each chunk to its parent video.

<!-- dprint-ignore-start -->
```python title="src/example_one_to_many/features.py" hl_lines="9"
--8<-- "example-one-to-many/src/example_one_to_many/features.py:video_chunk"
```
<!-- dprint-ignore-end -->

We do not specify custom versions on its fields. Metaxy will automatically assign field-level lineage by [matching on field names](../reference/api/definitions/fields-mapping.md): `"video/chunk:frames"` depends on `"video/raw:frames"` and `"video/chunk:audio"` depends on `"video/raw:audio"`.

::: metaxy-example graph
    example: one-to-many
    scenario: "Initial pipeline run"
    direction: LR
    show_field_deps: true
    features: ["video/raw", "video/chunk"]

### Defining features: `"video/faces"`

`"video/faces"` processes video chunks and **only depends on the `frames` field**. This can be expressed with a [`FieldDep`][metaxy.FieldDep].

<!-- dprint-ignore-start -->
```python title="src/example_one_to_many/features.py" hl_lines="9"
--8<-- "example-one-to-many/src/example_one_to_many/features.py:face_recognition"
```
<!-- dprint-ignore-end -->

::: metaxy-example graph
    example: one-to-many
    scenario: "Initial pipeline run"
    direction: LR
    show_field_deps: true
    features: ["video/chunk", "video/faces"]

This completes the feature definitions. Let's proceed to running the pipeline.

## Walkthrough

Here is a toy pipeline for computing the feature graph described above:

::: metaxy-example file
    example: one-to-many
    path: pipeline.py

### Step 1: Launch Initial Run

Run the pipeline to create videos, chunks, and face recognition results:

::: metaxy-example output
    example: one-to-many
    scenario: "Initial pipeline run"
    step: "initial_run"

All three features have been materialized. Note that the `"video/chunk"` feature may dynamically create as many samples as needed: Metaxy doesn't need to know anything about this in advance, except the relationship type.

### Step 2: Verify Idempotency

Run the pipeline again without any changes:

::: metaxy-example output
    example: one-to-many
    scenario: "Idempotent rerun"
    step: "idempotent_run"

Nothing needs recomputation - the system correctly detects no changes.

### Step 3: Change Audio Code Version

Now let's bump the code version on the `audio` field of `"video/raw"` feature:

::: metaxy-example patch-with-diff
    example: one-to-many
    path: patches/01_update_video_code_version.patch
    scenario: "Code change - audio field only"
    step: "update_audio_version"

This represents updating the audio processing algorithm, and therefore the audio data, while frame data is kept the same.

### Step 4: Observe Field-Level Tracking

Run the pipeline again after the code change:

::: metaxy-example output
    example: one-to-many
    scenario: "Code change - audio field only"
    step: "recompute_after_audio_change"

**Key observation:**

- `"video/chunk"` has been recomputed since the `audio` field on it has been affected by the upstream change
- `"video/faces"` did not require a recompute, because it only depends on the `frames` field (which did not change)

## Conclusion

Metaxy provides a convenient API for modeling expansion relationships: [LineageRelationship.expansion][metaxy.models.lineage.LineageRelationship.expansion]. Other Metaxy features such as field-level versioning continue to work seamlessly when declaring expansion relationships.

## Related materials

Learn more about:

- [Features and Fields](../guide/concepts/definitions/features.md)
- [Relationships](/guide/concepts/definitions/relationship.md)
- [Fields Mapping](../guide/concepts/syntactic-sugar.md#fields-mapping)
