---
title: "Syntactic Sugar"
description: "Shorthand syntax for feature definitions."
---

# Syntactic Sugar

## Type Coercion For Input Types

Internally, Metaxy uses strongly typed Pydantic models to represent feature keys, their fields, and the dependencies between them.

To avoid boilerplate, Metaxy also has syntactic sugar for construction of these classes.
Different ways to provide them are automatically coerced into canonical internal models.
This is fully typed and only affects **constructor arguments**, so accessing **attributes** on Metaxy models will always return only the canonical types.

Some examples:

```py
import metaxy as mx

key = mx.FeatureKey("prefix/feature")
key = mx.FeatureKey(["prefix", "feature"])
same_key = mx.FeatureKey(key)
```

Metaxy really loves you, the user!

## Keys

Both `FeatureKey` and `FieldKey` accept:

- **String format**: `FeatureKey("prefix/feature")`

- **Sequence format**: `FeatureKey(["prefix", "feature"])`

- **Same type**: `FeatureKey(another_feature_key)` -- for full Inception mode

All formats produce equivalent keys, internally represented as a sequence of parts.

## Feature Dep

[`FeatureDep`][metaxy.FeatureDep] accepts types coercible to `FeatureKey` and additionally subclasses of `BaseFeature`:

```py
import metaxy as mx

dep = mx.FeatureDep(feature=MyFeature)
```

## Feature Spec

[`FeatureSpec`][metaxy.FeatureSpec] has some syntactic sugar implemented as well.

### Deps

The `deps` argument accepts a sequence of types coercible to `FeatureDep`:

```py
import metaxy as mx

spec = mx.FeatureSpec(
    key="example/spec",
    id_columns=["id"],
    deps=[
        MyFeature,
        mx.FeatureDep(feature=["my", "feature", "key"]),  # sequence format
        ["another", "key"],  # also sequence format
        "very/nice",  # string format with slash separator
    ],
)
```

### Fields

`fields` elements can omit the full `FieldsSpec` and be strings (field keys) instead:

```python
import metaxy as mx

spec = mx.FeatureSpec(
    key="example/fields",
    id_columns=["id"],
    fields=["my/field", mx.FieldSpec(key="field/with/version", code_version="v1.2.3")],
)
```

### Fields Mapping

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
