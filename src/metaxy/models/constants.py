"""Shared constants for system column names."""

DEFAULT_CODE_VERSION = "__metaxy_initial__"

# Essential system columns that must always be preserved for joining/versioning
# Note: ID columns are defined by BaseFeatureSpec.id_columns (default: ["sample_uid"])
# provenance_by_field is always required for versioning (stored as metaxy_provenance_by_field in DB)
ESSENTIAL_SYSTEM_COLUMNS = frozenset(
    {
        "provenance_by_field",  # Always required for versioning (Python name)
        "metaxy_provenance_by_field",  # Database storage name
    }
)

# System columns that should be dropped to avoid conflicts when joining upstream features
# These will be recalculated for the target feature, so keeping them from upstream causes conflicts
DROPPABLE_SYSTEM_COLUMNS = frozenset(
    {
        "feature_version",
        "snapshot_version",
        "metaxy_feature_version",
        "metaxy_snapshot_version",
    }
)

# All system columns (essential + droppable)
ALL_SYSTEM_COLUMNS = ESSENTIAL_SYSTEM_COLUMNS | DROPPABLE_SYSTEM_COLUMNS
