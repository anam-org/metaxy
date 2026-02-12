---
title: "Basic Example"
description: "Basic example of upstream change detection and recomputation."
---

# Basic Example

## Overview

::: metaxy-example source-link
example: basic
:::

This example demonstrates how Metaxy automatically detects changes in upstream features and triggers recomputation of downstream features. It shows the core value proposition of Metaxy: avoiding unnecessary recomputation while ensuring data consistency.

We will build a simple two-feature pipeline where a child feature depends on a parent feature. When the parent's algorithm changes (represented by `code_version`), the child feature is automatically recomputed.

## The Pipeline

Let's define a pipeline with two features:

::: metaxy-example graph
example: basic
scenario: "Setup upstream data"
:::

### Defining features: `ParentFeature`

The parent feature represents raw embeddings computed from source data. It has a single field `embeddings` with a `code_version` that tracks the algorithm version.

<!-- dprint-ignore-start -->
```python title="src/example_basic/features.py"
--8<-- "example-basic/src/example_basic/features.py:parent_feature"
```
<!-- dprint-ignore-end -->

### Defining features: `ChildFeature`

The child feature depends on the parent and produces predictions. The key configuration is the `FeatureDep` which declares that `ChildFeature` depends on `ParentFeature`.

<!-- dprint-ignore-start -->
```python title="src/example_basic/features.py" hl_lines="5"
--8<-- "example-basic/src/example_basic/features.py:child_feature"
```
<!-- dprint-ignore-end -->

The `FeatureDep` declaration tells Metaxy:

1. `ChildFeature` depends on `ParentFeature`
2. When the parent's field provenance changes, the child must be recomputed
3. This dependency is tracked automatically, enabling incremental recomputation

## Walkthrough

### Step 1: Initial Run

Run the pipeline to create parent embeddings and child predictions:

::: metaxy-example output
example: basic
scenario: "Initial pipeline run"
step: "initial_run"
:::

The pipeline materialized 3 samples for the child feature. Each sample has its provenance tracked.

### Step 2: Verify Idempotency

Run the pipeline again without any changes:

::: metaxy-example output
example: basic
scenario: "Idempotent rerun"
step: "idempotent_run"
:::

**Key observation:** No recomputation occurred.

### Step 3: Update Parent Algorithm

Now let's simulate an algorithm improvement by changing the parent's `code_version` from `"1"` to `"2"`:

::: metaxy-example patch-with-diff
example: basic
path: patches/01_update_parent_algorithm.patch
scenario: "Code evolution"
step: "update_parent_version"
:::

This change means that the existing embeddings and the downstream feature have to be recomputed.

### Step 4: Observe Automatic Recomputation

Run the pipeline again after the algorithm change:

::: metaxy-example output
example: basic
scenario: "Code evolution"
step: "recompute_after_change"
:::

**Key observation:** The child feature was automatically recomputed because:

1. The parent's `code_version` changed from `"1"` to `"2"`
2. This changed the parent's `metaxy_feature_version`
3. The child's field dependency on `embeddings` detected the change
4. All child samples were marked for recomputation

## How It Works

Metaxy tracks provenance at the field level using:

1. **Field Version**: A hash combining the field's `code_version` and provenances of upstream fields
2. **Feature Version**: A hash combining the field versions of all fields in the feature
3. **Dependency Resolution**: When resolving updates, Metaxy computes what the provenance _would be_ and compares it to what's stored

This enables precise, incremental recomputation without re-processing unchanged data.

## Conclusion

Metaxy provides automatic change detection and incremental recomputation through:

- Feature dependency tracking via [`FeatureDep`][metaxy.FeatureDep]
- Algorithm versioning via [`FieldSpec.code_version`][metaxy.FieldSpec]
- Provenance-based change detection via [`MetadataStore.resolve_update`][metaxy.MetadataStore.resolve_update]

This mechanism ensures your pipelines are both efficient and keep relevant data up to date.

## Related Materials

Learn more about:

- [Features and Fields](../guide/concepts/definitions/features.md)
- [Data Versioning](../guide/concepts/data-versioning.md)
- [Relationships](/guide/concepts/definitions/relationship.md)
