---
title: "Aggregation Example"
description: "Example of many-to-one aggregation relationships."
---

# Aggregation

## Overview

::: metaxy-example source-link
example: aggregation
:::

This example demonstrates how to implement aggregation (`N:1`) relationships with Metaxy.
In such relationships multiple parent samples produce a single child sample.

These relationships can be modeled with [LineageRelationship.aggregation][metaxy.models.lineage.LineageRelationship.aggregation] lineage type.

We will use a speaker embedding pipeline as an example, where multiple audio recordings from the same speaker are aggregated to compute a single speaker embedding.

## The Pipeline

Let's define a pipeline with two features:

::: metaxy-example graph
example: aggregation
scenario: "Initial pipeline run"
:::

### Defining features: `Audio`

Each audio recording has an `audio_id` (unique identifier) and a `speaker_id` (which speaker it belongs to). Multiple audio recordings can belong to the same speaker.

<!-- dprint-ignore-start -->
```python title="src/example_aggregation/features.py"
--8<-- "example-aggregation/src/example_aggregation/features.py:1:18"
```
<!-- dprint-ignore-end -->

### Defining features: `SpeakerEmbedding`

`SpeakerEmbedding` aggregates all audio recordings from a speaker into a single embedding. The key configuration is the `lineage` parameter which tells Metaxy that multiple `Audio` records with the same `speaker_id` are aggregated into one `SpeakerEmbedding`.

<!-- dprint-ignore-start -->
```python title="src/example_aggregation/features.py" hl_lines="6-11"
--8<-- "example-aggregation/src/example_aggregation/features.py:20:45"
```
<!-- dprint-ignore-end -->

The `LineageRelationship.aggregation(on=["speaker_id"])` declaration is the key part. It tells Metaxy:

1. Multiple `Audio` rows are aggregated into one `SpeakerEmbedding` row
2. The aggregation is keyed on `speaker_id` - all audio with the same speaker_id contributes to one embedding
3. When **any** audio for a speaker changes, the aggregated provenance changes, triggering recomputation of that speaker's embedding

## Walkthrough

Here is the pipeline code that processes audio and computes speaker embeddings:

::: metaxy-example file
example: aggregation
path: pipeline.py
:::

### Step 1: Initial Run

Run the pipeline to create audio recordings and speaker embeddings:

::: metaxy-example output
example: aggregation
scenario: "Initial pipeline run"
step: "run_pipeline"
:::

All features have been materialized:

- 4 audio recordings (2 per speaker)
- 2 speaker embeddings (one per speaker)

### Step 2: Verify Idempotency

Run the pipeline again without any changes:

::: metaxy-example output
example: aggregation
scenario: "Idempotent rerun"
:::

Nothing needs recomputation - the system correctly detects no changes.

### Step 3: Update One Audio Recording

Now let's update the provenance of audio `a1` (belonging to speaker `s1`):

::: metaxy-example patch
example: aggregation
path: patches/01_update_audio_provenance.patch
:::

This represents a change to one audio recording (perhaps it was re-processed or updated).

### Step 4: Observe Selective Recomputation

Run the pipeline again after the audio change:

::: metaxy-example output
example: aggregation
scenario: "Update one audio - only affected speaker recomputed"
:::

**Key observation:**

- Only speaker `s1`'s embedding is recomputed (because audio `a1` belongs to `s1`)
- Speaker `s2`'s embedding is **not** recomputed (none of their audio changed)

This demonstrates that Metaxy correctly tracks aggregation lineage - when any audio for a speaker changes, only that speaker's embedding needs recomputation.

### Step 5: Add New Audio

Now let's add a new audio recording for speaker `s1`:

::: metaxy-example patch
example: aggregation
path: patches/02_add_audio.patch
:::

### Step 6: Observe Aggregation Update

Run the pipeline again:

::: metaxy-example output
example: aggregation
scenario: "Add new audio - only affected speaker recomputed"
:::

**Key observation:**

- Only speaker `s1`'s embedding is recomputed (the new audio belongs to `s1`)
- Speaker `s1` now has 3 audio recordings (up from 2)
- Speaker `s2` remains unchanged

## How It Works

Metaxy uses window functions to compute aggregated provenance without reducing rows. When resolving updates for `SpeakerEmbedding`:

1. All audio rows for the same speaker get identical aggregated provenance values
2. The aggregated provenance is computed from the individual audio provenances
3. When any audio for a speaker changes, the aggregated provenance changes
4. This triggers recomputation of only the affected speaker's embedding

## Conclusion

Metaxy provides a convenient API for modeling aggregation relationships: [LineageRelationship.aggregation][metaxy.models.lineage.LineageRelationship.aggregation]. Other Metaxy features continue to seamlessly work with aggregation relationships.

## Related Materials

Learn more about:

- [Features and Fields](../guide/concepts/definitions/features.md)
- [Relationships](/guide/concepts/definitions/relationship.md)
- [One-to-Many Expansion](./expansion.md) (the inverse relationship)
