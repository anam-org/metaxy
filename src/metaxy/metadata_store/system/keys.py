"""System table keys and constants."""

from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldKey, FieldSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey

METAXY_SYSTEM_KEY_PREFIX = "metaxy-system"

# System table keys
FEATURE_VERSIONS_KEY = FeatureKey([METAXY_SYSTEM_KEY_PREFIX, "feature_versions"])
EVENTS_KEY = FeatureKey([METAXY_SYSTEM_KEY_PREFIX, "events"])


def _create_system_spec(key: FeatureKey, id_columns: tuple[str, ...]) -> FeatureSpec:
    """Create a minimal FeatureSpec for a system table."""
    return FeatureSpec(
        key=key,
        id_columns=id_columns,
        fields=[FieldSpec(key=FieldKey([col]), code_version="1") for col in id_columns],
        # No deps - system tables are root features
    )


def _create_system_plan(spec: FeatureSpec) -> FeaturePlan:
    """Create a minimal FeaturePlan for a system table (no dependencies)."""
    return FeaturePlan(feature=spec, deps=None)


# System FeatureSpecs (for versioning engine operations)
# feature_versions uses compound ID: (feature_key, full_definition_version)
# This preserves history while allowing latest lookups
FEATURE_VERSIONS_SPEC = _create_system_spec(
    FEATURE_VERSIONS_KEY, ("feature_key", "metaxy_full_definition_version")
)
EVENTS_SPEC = _create_system_spec(EVENTS_KEY, ("event_id",))

# System FeaturePlans (for versioning engine context)
FEATURE_VERSIONS_PLAN = _create_system_plan(FEATURE_VERSIONS_SPEC)
EVENTS_PLAN = _create_system_plan(EVENTS_SPEC)
