# Syntactic Sugar

## Type Coercion For Input Types

Internally, Metaxy uses strongly typed Pydantic models to represent feature keys, their fields, and the dependencies between them.

To avoid boilerplate, Metaxy also has syntactic sugar for construction of these classes.
Different ways to provide them are automatically coerced into canonical internal models.
This is fully typed and only affects **constructor arguments**, so accessing **attributes** on Metaxy models will always return only the canonical types.

Some examples:

```py
from metaxy import FeatureKey

key = FeatureKey("prefix/feature")
key = FeatureKey(["prefix", "feature"])
key = FeatureKey("prefix", "feature")
same_key = FeatureKey(key)
```

Metaxy really loves you, the user!

## Keys

Both `FeatureKey` and `FieldKey` accept:

- **String format**: `FeatureKey("prefix/feature")`
- **Sequence format**: `FeatureKey(["prefix", "feature"])`
- **Variadic format**: `FeatureKey("prefix", "feature")`
- **Same type**: `FeatureKey(another_feature_key)` -- for full Inception mode

All formats produce equivalent keys, internally represented as a sequence of parts.

## `FeatureSpec`

### Fields

[`FieldSpec`][metaxy.FieldSpec] can be passed to [`FeatureSpec`][metaxy.FeatureSpec] as a string that represents the field key:

```python
spec = FeatureSpec(
    ..., fields=["my/field", FieldSpec(key="field/with/version", code_version="v1.2.3")]
)
```

### Fields Mapping

Metaxy uses a bunch of common sense heuristics [automatically find parent fields](../reference/api/definitions/fields-mapping.md) by matching on their names. This is enabled by default. For example, using the same field names in upstream and downstream features will automatically create a dependency between these fields:

```py
class Parent(BaseFeature, spec=FeatureSpec(fields=["my_field"], ...):
    ...

class Child(Parent, spec=FeatureSpec(fields=["my_field"], ...):
    ...
```

is equivalent to:

```py
class Child(Parent, spec=FeatureSpec(fields=["my_field"], ...):
    ...

class Grandchild(Child, spec=FeatureSpec(fields=[FieldSpec(key="my_field", deps=[FieldDep(feature=Parent.spec().key, field="my_field")])], ...):
    ...
```
