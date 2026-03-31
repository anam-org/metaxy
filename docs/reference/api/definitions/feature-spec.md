---
title: "Feature Spec API"
description: "API reference for FeatureSpec and FeatureDep."
---

# Feature Spec

Feature specs act as source of truth for all metadata related to features: their dependencies, fields, code versions, and so on.

::: metaxy.FeatureSpec

## Unique

`FeatureSpec.unique` defines a read-time uniqueness constraint. `subset` groups
records that represent the same value or entity. `keep="any"` selects a stable
representative. `keep="latest"` requires `order_by`; the row with the
lexicographically greatest values in those columns is selected, with the
feature's ID columns providing final logical tie-breakers. Together they must
define a total logical order; conflicting rows tied on the full key are invalid.
NULL sorts before every non-NULL value. Repeated column names are ignored after
their first occurrence. Physical metadata stabilizes selection among identical
re-appends and resolves conflicting full-key ties; detecting those conflicts is
currently the writer's responsibility.

Default current reads resolve each `id_columns` group and remove soft-deleted
samples before applying uniqueness. With `include_soft_deleted=True`,
deleted current rows participate normally and may win. An unregistered history
read cannot apply the feature contract. To read its raw rows, request both
feature and sample history and set `apply_unique=False`.

Dependency-provided columns are accepted but may produce a warn-only validation
message because they are not part of the feature's declared schema.

::: metaxy.Unique

::: metaxy.UniqueKeep

## Feature Dependencies

::: metaxy.FeatureDep
