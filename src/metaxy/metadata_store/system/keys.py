"""System table keys and constants."""

from metaxy.models.types import FeatureKey

METAXY_SYSTEM_KEY_PREFIX = "metaxy-system"

# System table keys
FEATURE_VERSIONS_KEY = FeatureKey([METAXY_SYSTEM_KEY_PREFIX, "feature_versions"])
EVENTS_KEY = FeatureKey([METAXY_SYSTEM_KEY_PREFIX, "events"])
