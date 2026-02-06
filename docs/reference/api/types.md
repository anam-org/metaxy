---
title: "Types API"
description: "Type definitions used in Metaxy."
---

# Types

## Versioning Engine

::: metaxy.versioning.types.LazyChanges

::: metaxy.versioning.types.Changes

::: metaxy.versioning.types.PolarsChanges
    options:
      members: true
      show_if_no_docstring: true

::: metaxy.versioning.types.PolarsLazyChanges
    options:
      members: true
      show_if_no_docstring: true

::: metaxy.HashAlgorithm
    options:
      show_if_no_docstring: true

## Keys

Types for working with feature and field keys.

### Canonical Keys

::: metaxy.FeatureKey

::: metaxy.FieldKey

### Type Annotations

These are typically used to annotate function parameters. Most APIs in Metaxy accepts them and perform type coercion into [canonical types](#canonical-keys).

::: metaxy.CoercibleToFeatureKey

::: metaxy.CoercibleToFieldKey

### Pydantic Type Annotations

These types are used for type coercion into [canonical types](#canonical-keys) with Pydantic.

::: metaxy.ValidatedFeatureKey

::: metaxy.ValidatedFieldKey

::: metaxy.ValidatedFeatureKeySequence

::: metaxy.ValidatedFieldKeySequence

### Adapters

These can perform type coercsion into [canonical types](#canonical-keys) in non-pydantic code.

::: metaxy.ValidatedFeatureKeyAdapter

::: metaxy.ValidatedFeatureKeySequenceAdapter

::: metaxy.ValidatedFieldKeyAdapter

::: metaxy.ValidatedFieldKeySequenceAdapter

## Other Types

::: metaxy.models.types.SnapshotPushResult

::: metaxy.IDColumns
