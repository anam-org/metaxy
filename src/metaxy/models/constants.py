"""Shared constants for system-managed column names.

All system columns use the metaxy_ prefix to avoid conflicts with user columns.
"""

from __future__ import annotations

# Default code version for initial feature definitions
DEFAULT_CODE_VERSION = "__metaxy_initial__"

# System column prefix
SYSTEM_COLUMN_PREFIX = "metaxy_"

# --- System Column Names -----------------------------------------------------------
# All system columns that Metaxy manages internally. These columns are automatically
# added to metadata DataFrames and should not be defined by users.

METAXY_PROVENANCE_BY_FIELD = f"{SYSTEM_COLUMN_PREFIX}provenance_by_field"
"""Field-level provenance hashes (struct column mapping field names to hashes)."""

METAXY_PROVENANCE = f"{SYSTEM_COLUMN_PREFIX}provenance"
"""Hash of`metaxy_provenance_by_field` -- a single string value."""

METAXY_FEATURE_VERSION = f"{SYSTEM_COLUMN_PREFIX}feature_version"
"""Hash of the feature definition (dependencies + fields + code_versions)."""

METAXY_SNAPSHOT_VERSION = f"{SYSTEM_COLUMN_PREFIX}snapshot_version"
"""Hash of the entire feature graph snapshot (recorded during deployment)."""

METAXY_DEFINITION_VERSION = f"{SYSTEM_COLUMN_PREFIX}definition_version"
"""Hash of the complete feature definition including Pydantic schema and feature spec.

This comprehensive hash captures the feature definition (excluding project):
- Pydantic model schema (field types, descriptions, validators, serializers, etc.)
- Feature specification (dependencies, fields, code_versions, metadata)

Project is stored separately. Used in system tables to detect when ANY part of a feature changes."""

METAXY_DATA_VERSION_BY_FIELD = f"{SYSTEM_COLUMN_PREFIX}data_version_by_field"
"""Field-level data version hashes (struct column mapping field names to version hashes).

Similar to provenance_by_field, but can be user-overridden to implement custom versioning
(e.g., content hashes, timestamps, semantic versions)."""

METAXY_DATA_VERSION = f"{SYSTEM_COLUMN_PREFIX}data_version"
"""Hash of metaxy_data_version_by_field -- a single string value."""

METAXY_CREATED_AT = f"{SYSTEM_COLUMN_PREFIX}created_at"
"""Timestamp when the metadata row was created."""

METAXY_UPDATED_AT = f"{SYSTEM_COLUMN_PREFIX}updated_at"
"""Timestamp when the metadata row was last updated (written to the store)."""

METAXY_DELETED_AT = f"{SYSTEM_COLUMN_PREFIX}deleted_at"
"""Timestamp when the metadata row was soft-deleted."""

METAXY_MATERIALIZATION_ID = f"{SYSTEM_COLUMN_PREFIX}materialization_id"
"""External orchestration run ID (e.g., Dagster Run ID, Airflow Run ID) for tracking pipeline executions."""

# --- System Column Sets ------------------------------------------------------------

ALL_SYSTEM_COLUMNS = frozenset(
    {
        METAXY_PROVENANCE_BY_FIELD,
        METAXY_PROVENANCE,
        METAXY_FEATURE_VERSION,
        METAXY_SNAPSHOT_VERSION,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_DATA_VERSION,
        METAXY_CREATED_AT,
        METAXY_UPDATED_AT,
        METAXY_DELETED_AT,
        METAXY_MATERIALIZATION_ID,
    }
)
"""All Metaxy-managed column names that are injected into feature tables."""

# Columns that should be dropped when joining upstream features (will be recalculated)
_DROPPABLE_COLUMNS = frozenset(
    {
        METAXY_FEATURE_VERSION,
        METAXY_SNAPSHOT_VERSION,
        METAXY_CREATED_AT,
        METAXY_UPDATED_AT,
        METAXY_DELETED_AT,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_DATA_VERSION,
        METAXY_MATERIALIZATION_ID,
    }
)

# Columns that should be dropped before joining upstream features in FeatureDepTransformer
# These are NOT needed for provenance calculation and would cause column name conflicts
# when joining 3+ upstream features (e.g., metaxy_created_at_right already exists error)
_COLUMNS_TO_DROP_BEFORE_JOIN = frozenset(
    {
        METAXY_FEATURE_VERSION,
        METAXY_SNAPSHOT_VERSION,
        METAXY_CREATED_AT,
        METAXY_UPDATED_AT,
        METAXY_MATERIALIZATION_ID,
    }
)


# --- Utility Functions -------------------------------------------------------------


def is_system_column(name: str) -> bool:
    """Check whether a column name is a system-managed column.

    Args:
        name: Column name to check

    Returns:
        True if the column is a system column, False otherwise

    Examples:
        >>> is_system_column("metaxy_feature_version")
        True
        >>> is_system_column("my_column")
        False
    """
    return name in ALL_SYSTEM_COLUMNS


def is_droppable_system_column(name: str) -> bool:
    """Check whether a column should be dropped when joining upstream features.

    Droppable columns (feature_version, snapshot_version) are recalculated for
    each feature, so keeping them from upstream would cause conflicts.

    Args:
        name: Column name to check

    Returns:
        True if the column should be dropped during joins, False otherwise

    Examples:
        >>> is_droppable_system_column("metaxy_feature_version")
        True
        >>> is_droppable_system_column("metaxy_provenance_by_field")
        False
    """
    return name in _DROPPABLE_COLUMNS


# System columns that have lineage from upstream features
# These columns are computed from corresponding upstream columns (same column name)
# With 5 parents, each of these columns will have 5 dependencies
SYSTEM_COLUMNS_WITH_LINEAGE: frozenset[str] = frozenset(
    {
        METAXY_PROVENANCE_BY_FIELD,
        METAXY_PROVENANCE,
        METAXY_DATA_VERSION_BY_FIELD,
        METAXY_DATA_VERSION,
    }
)
