"""Migration system for metadata version updates."""

from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations.detector import detect_diff_migration
from metaxy.migrations.executor import MigrationExecutor
from metaxy.migrations.loader import (
    build_migration_chain,
    find_latest_migration,
    find_migration_file,
    find_migration_yaml,
    list_migrations,
    load_migration_from_file,
    load_migration_from_python,
    load_migration_from_yaml,
)
from metaxy.migrations.models import (
    DiffMigration,
    FullGraphMigration,
    Migration,
    MigrationResult,
    PythonMigration,
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
    "PythonMigration",
    "MigrationResult",
    # Operations (for custom migrations)
    "BaseOperation",
    "DataVersionReconciliation",
    "MetadataBackfill",
    # Migration workflow
    "detect_diff_migration",
    "MigrationExecutor",
    "SystemTableStorage",
    # Loader functions
    "load_migration_from_yaml",
    "load_migration_from_python",
    "load_migration_from_file",
    "find_migration_file",
    "find_migration_yaml",
    "list_migrations",
    "find_latest_migration",
    "build_migration_chain",
]
