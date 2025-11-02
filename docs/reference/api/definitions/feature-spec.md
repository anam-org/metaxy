# Feature Spec

Feature specs act as source of truth for all metadata related to features: their dependencies, fields, code versions, and so on.

!!! tip "Accessing Code Versions"
    Use `MyFeature.spec().code_version` to inspect the deterministic hash for a feature definition. This value is only exposed via the spec to keep the `Feature` API focused on user-defined fields.

::: metaxy.BaseFeatureSpec

::: metaxy.FeatureSpec

# Feature Dependencies

::: metaxy.FeatureDep
