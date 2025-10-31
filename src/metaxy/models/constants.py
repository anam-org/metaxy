"""Shared constants for system column names."""

# Essential system columns that must always be preserved for joining/versioning
# Note: ID columns are defined by BaseFeatureSpec.id_columns() (default: ["sample_uid"])
# data_version is always required for versioning
ESSENTIAL_SYSTEM_COLUMNS = frozenset(
    {
        "data_version",  # Always required for versioning
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
        "metaxy_data_version",
    }
)

# All system columns (essential + droppable)
ALL_SYSTEM_COLUMNS = ESSENTIAL_SYSTEM_COLUMNS | DROPPABLE_SYSTEM_COLUMNS
