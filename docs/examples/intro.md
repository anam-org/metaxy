---
title: "Introduction"
description: "Video/audio ML pipeline demonstrating incremental recomputation."
---

# Introduction

## Overview

::: metaxy-example source-link
    example: intro

This example walks through building a multimodal ML pipeline that processes video files. It demonstrates Metaxy's core workflow: declaring features with dependencies, resolving incremental updates, and detecting when code changes require selective recomputation.

The pipeline extracts faces and transcriptions from videos, then summarizes the transcriptions:

::: metaxy-example graph
    example: intro
    scenario: "Initial pipeline run"
    direction: LR

## The Pipeline

### Defining features: `"video"`

The root feature represents raw video files. It has no dependencies and no versioned fields since it tracks external data.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py"
--8<-- "example-intro/src/example_intro/features.py:video_feature"
```
<!-- dprint-ignore-end -->

### Defining features: `"video/faces"`

Depends on `"video"`. The `faces` field has a `code_version` that tracks the algorithm version.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py" hl_lines="4"
--8<-- "example-intro/src/example_intro/features.py:face_detection_feature"
```
<!-- dprint-ignore-end -->

The `deps=[Video]` declaration tells Metaxy:

1. `"video/faces"` depends on `"video"`
2. When the parent's provenance changes, face detection must be recomputed
3. Changes to sibling features like `"video/transcription"` do not affect this feature

### Defining features: `"video/transcription"`

Also depends on `"video"`, but is independent of `"video/faces"`. The `transcript` field tracks the transcription algorithm version.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py" hl_lines="4"
--8<-- "example-intro/src/example_intro/features.py:audio_transcription_feature"
```
<!-- dprint-ignore-end -->

### Defining features: `"video/summary"`

Depends on `"video/transcription"`. When the transcription algorithm changes, summaries are automatically recomputed. Changes to `"video/faces"` do not affect summaries because there is no dependency path between them.

<!-- dprint-ignore-start -->
```python title="src/example_intro/features.py" hl_lines="4"
--8<-- "example-intro/src/example_intro/features.py:summary_feature"
```
<!-- dprint-ignore-end -->

## Getting Started

Install the example's dependencies:

```shell
uv sync
```

## Walkthrough

### Step 1: Initial Run

Run the pipeline to process all videos through face detection, transcription, and summarization:

::: metaxy-example output
    example: intro
    scenario: "Initial pipeline run"
    step: "initial_run"

All four features were materialized for the 3 video samples.

### Step 2: Verify Idempotency

Run the pipeline again without any changes:

::: metaxy-example output
    example: intro
    scenario: "Idempotent rerun"
    step: "idempotent_run"

**Key observation:** No recomputation occurred. Metaxy detected that all provenances match the stored state and skipped every feature.

### Step 3: Update Transcription Algorithm

Now let's simulate an improved transcription model by changing `"video/transcription"`'s `code_version` from `"1"` to `"2"`:

::: metaxy-example patch-with-diff
    example: intro
    path: patches/01_update_transcription.patch
    scenario: "Code evolution"
    step: "update_transcription_version"

This change means that the existing transcriptions and the downstream summaries have to be recomputed.

### Step 4: Observe Selective Recomputation

Run the pipeline again after the algorithm change:

::: metaxy-example output
    example: intro
    scenario: "Code evolution"
    step: "recompute_after_change"

**Key observation:**

- `"video/transcription"` and its downstream `"video/summary"` were recomputed
- `"video/faces"` was **not** recomputed because it depends only on `"video"`, which did not change

## How It Works

The branching graph in this example has two independent paths from `"video"`:

- `"video"` → `"video/faces"` (face detection branch)
- `"video"` → `"video/transcription"` → `"video/summary"` (transcription branch)

When the transcription `code_version` changes, Metaxy recomputes the feature version for `"video/transcription"`. This changes its provenance, which propagates to `"video/summary"`. The face detection branch is unaffected because its version depends only on `"video"`, which did not change.

This is the core mechanism: [`resolve_update`][metaxy.MetadataStore.resolve_update] computes what the provenance _would be_ given the current definitions and compares it to what is stored. Only samples where these differ are returned for recomputation.

## Related Materials

- [Basic Example](basic.md) -- two-feature pipeline with the same workflow
- [Features and Fields](../guide/concepts/definitions/features.md) -- field-level lineage with [`FieldDep`][metaxy.FieldDep]
- [Data Versioning](../guide/concepts/versioning.md) -- how versions are calculated
- [Relationships](../guide/concepts/definitions/relationship.md) -- aggregation and expansion patterns
