---
title: "Syntactic Sugar"
description: "Shorthand syntax for feature definitions."
---

# Syntactic Sugar

## Type Coercion For Input Types

Internally, Metaxy uses strongly typed Pydantic models to represent features, feature keys, their fields, and dependencies between them. But specifying all of these models can be very verbose and cumbersome.

Because Metaxy loves its users, we provide syntactic sugar for simplified construction of these models, and various Metaxy APIs typically accept unions of equivalent types.
Metaxy coerces them into canonical internal models.

This is fully typed and only affects constructor arguments. Attributes on Metaxy objects will always return only the canonical type.

## Features

APIs that require feature references accept the following equivalent objects:

- **slash-separated strings**: `"my/feature"`

- **sequence of strings**: `["my", "feature"]`

- [`FeatureKey`][metaxy.FeatureKey]: `mx.FeatureKey("my/feature")`

- [`FeatureSpec`][metaxy.FeatureSpec]: `mx.FeatureSpec(key="my/feature", ...)`

- [`BaseFeature`] types: `Myfeature`, where `Myfeature` is a subclass of `BaseFeature`

## Keys

Both `FeatureKey` and `FieldKey` can be constructed from:

- **slash-separated strings**: `FeatureKey("prefix/feature")`

- **sequence of strings**: `FeatureKey(["prefix", "feature"])`

Internally they are represented as a sequence of parts.

## Fields

`fields` argument of `FeatureSpec` can omit the full `FieldsSpec`:

```python
import metaxy as mx

spec = mx.FeatureSpec(
    key="example/fields",
    id_columns=["id"],
    fields=["my/field", mx.FieldSpec(key="field/with/version", code_version="v1.2.3")],
)
```

## Fields Mapping

Metaxy uses a bunch of common sense heuristics [automatically find parent fields](../../reference/api/definitions/fields-mapping.md) by matching on their names. This is enabled by default. For example, using the same field names in upstream and downstream features will automatically create a dependency between these fields:

```py
import metaxy as mx


class Parent(mx.BaseFeature, spec=mx.FeatureSpec(key="parent/feature", id_columns=["id"], fields=["my_field"])):
    id: str


class Child(
    mx.BaseFeature, spec=mx.FeatureSpec(key="child/feature", id_columns=["id"], deps=[Parent], fields=["my_field"])
):
    id: str
```

is equivalent to:

```py
import metaxy as mx


class Grandchild(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="grandchild/feature",
        id_columns=["id"],
        deps=[Child],
        fields=[mx.FieldSpec(key="my_field", deps=[mx.FieldDep(feature=Parent, fields=["my_field"])])],
    ),
):
    id: str
```
