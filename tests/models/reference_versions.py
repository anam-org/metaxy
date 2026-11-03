"""Reference computation of field and feature versions.

The functions compute versions with a plain recursion over the feature graph and do not use
the version cache of `FeatureGraph`. Tests use them as an independent oracle for the cached computation.
The hash truncation length of the active config applies.

The functions look up parent fields with `FeaturePlan.get_parent_fields_for_field`, so tests that record
these lookups must call them only after their assertions on the recorded lookups.
"""

import hashlib

from metaxy._hashing import truncate_hash
from metaxy.models.feature import FeatureGraph
from metaxy.models.plan import FQFieldKey
from metaxy.models.types import CoercibleToFeatureKey, ValidatedFeatureKeyAdapter


def reference_field_version(graph: FeatureGraph, key: FQFieldKey) -> str:
    """Version of one field, from its code version and the versions of its parent fields."""
    definition = graph.feature_definitions_by_key.get(key.feature)

    # A field of an external feature with a provenance override uses the override as its version.
    if definition and definition.is_external and definition.has_provenance_override:
        provenance = definition.provenance_by_field_override
        field_str = key.field.to_string()
        if provenance and field_str in provenance:
            return provenance[field_str]

    hasher = hashlib.sha256()

    plan = graph.get_feature_plan(key.feature)
    field = plan.feature.fields_by_key[key.field]

    hasher.update(key.to_string().encode())
    hasher.update(str(field.code_version).encode())

    for parent_key, _parent_field in sorted(plan.get_parent_fields_for_field(key.field).items()):
        hasher.update(reference_field_version(graph, parent_key).encode())

    return truncate_hash(hasher.hexdigest())


def reference_feature_version_by_field(graph: FeatureGraph, key: CoercibleToFeatureKey) -> dict[str, str]:
    """Version of each field of a feature, in the field order of the feature spec."""
    validated_key = ValidatedFeatureKeyAdapter.validate_python(key)
    plan = graph.get_feature_plan(validated_key)
    return {
        field_key.to_string(): reference_field_version(graph, FQFieldKey(field=field_key, feature=validated_key))
        for field_key in plan.feature.fields_by_key
    }


def reference_feature_version(graph: FeatureGraph, key: CoercibleToFeatureKey) -> str:
    """Version of a feature, from the versions of its fields."""
    hasher = hashlib.sha256()
    version_by_field = reference_feature_version_by_field(graph, key)
    for field_key in sorted(version_by_field):
        hasher.update(field_key.encode())
        hasher.update(version_by_field[field_key].encode())

    return truncate_hash(hasher.hexdigest())
