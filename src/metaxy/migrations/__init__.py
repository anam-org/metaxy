"""Migration system for metadata version updates."""

from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations.detector import detect_diff_migration
from metaxy.migrations.executor import MigrationExecutor
from metaxy.migrations.models import (
    DiffMigration,
    FullGraphMigration,
    Migration,
    MigrationResult,
)
from metaxy.migrations.ops import (
    BaseOperation,
    DataVersionReconciliation,
    MetadataBackfill,
)

__all__ = [
    # Core migration types
    "Migration",
    "DiffMigration",
    "FullGraphMigration",
    "MigrationResult",
    # Operations (for custom migrations)
    "BaseOperation",
    "DataVersionReconciliation",
    "MetadataBackfill",
    # Migration workflow
    "detect_diff_migration",
    "MigrationExecutor",
    "SystemTableStorage",
]
