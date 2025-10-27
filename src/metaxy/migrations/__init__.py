"""Migration system for metadata version updates."""

from metaxy.metadata_store.system_tables import SystemTableStorage
from metaxy.migrations.detector import detect_migration
from metaxy.migrations.executor import MigrationExecutor
from metaxy.migrations.models import (
    CustomMigration,
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
    "CustomMigration",
    "MigrationResult",
    # Operations (for custom migrations)
    "BaseOperation",
    "DataVersionReconciliation",
    "MetadataBackfill",
    # Migration workflow
    "detect_migration",
    "MigrationExecutor",
    "SystemTableStorage",
]
