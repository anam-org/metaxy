---
title: "Definitions API"
description: "API reference for feature definitions."
---

# Definitions

Metaxy's dependency specification system allows users to express dependencies between their features and their fields.

## `FeatureSpec`

[FeatureSpec](./feature-spec.md) is the core of Metaxy's dependency specification system: it stores all the information about the parents, field mappings, and other metadata associated with a feature.

## Feature

A [feature](./feature.md) in Metaxy is used to model user-defined metadata. It must have a `FeatureSpec` instance associated with it. A `Feature` class is typically associated with a single table in the [MetadataStore][metaxy.metadata_store.base.MetadataStore].

## FieldSpec

A [field](./field.md) in Metaxy is a logical slices of the **data** represented by feature metadata. Users are free to define their own fields as is suitable for them.

Dependencies between fields are modeled with [FieldDep][metaxy.models.field.FieldDep] and can be automatic (via field mappings) or explicitly set by users.

## Graph

All features live on a [FeatureGraph](./graph.md) object. The users don't typically interact with it outside of advanced use cases.
